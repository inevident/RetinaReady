"""Fail-closed llama.cpp adapter for the RetinaPriority Gemma LoRA.

This is an optional, uncalibrated research-demo path.  It deliberately accepts
only the fixed quality-passing DeepDRiD presentation images, verifies the local
server and LoRA identity before every inference, and never treats free-
generation output as a calibrated probability.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from workflow import (
    EscalationAssessment,
    EscalationDecision,
    EscalationReason,
    uncertain_escalation,
)


# These prompts and the target keys are intentionally byte-for-byte aligned
# with the escalation task in ml/train_qlora.py.  Keep changes synchronized.
ESCALATION_SYSTEM_PROMPT = """You are RetinaPriority, an offline retinal review-priority assistant.
Assess only whether a clinically usable conventional color fundus photograph should
enter ROUTINE or PRIORITY under the declared diabetic-retinopathy
screening threshold. Never claim a diagnosis, infer that an eye is healthy, recommend
treatment, or delay human review. PRIORITY corresponds to the released dataset's
referable threshold (DR grade 2-4); ROUTINE corresponds to grade 0-1. Return
exactly one JSON object. Unsupported or uncertain inputs must be handled by the
application's fail-closed safety policy rather than released as routine."""
ESCALATION_USER_PROMPT = """Assign review priority for this clinically usable conventional
color fundus photograph. Return the RetinaPriority JSON object only, with no markdown
or additional prose."""
ESCALATION_DISCLAIMER = (
    "Review-priority support only; not a diagnosis or treatment recommendation."
)


def _response_schema_branch(decision: str, next_step: str) -> dict[str, object]:
    """Bind each internal decision to its one permitted policy-authored action."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "confidence": {"type": "null"},
            "decision": {"type": "string", "const": decision},
            "disclaimer": {"type": "string", "const": ESCALATION_DISCLAIMER},
            "next_step": {"type": "string", "const": next_step},
        },
        "required": ["confidence", "decision", "disclaimer", "next_step"],
    }


ESCALATION_RESPONSE_SCHEMA: dict[str, object] = {
    "oneOf": [
        _response_schema_branch(
            "ROUTINE", "Keep in the routine clinician review queue."
        ),
        _response_schema_branch(
            "PRIORITY", "Route for priority clinician review."
        ),
    ]
}

GEMMA_ESCALATION_SCOPE = "fixed-deepdrid-quality-pass-research-demo"
GEMMA_ESCALATION_PROFILE = "gemma-lora-free-generation-uncalibrated-experimental"
GEMMA_ESCALATION_MODEL_LABEL = (
    "RetinaPriority Gemma 4 LoRA · uncalibrated experimental research demo"
)
RESEARCH_DEMO_OPT_IN = "RETINA_ENABLE_ESCALATION_RESEARCH_DEMO"
MAX_RUNTIME_JSON_BYTES = 1024 * 1024
RUNTIME_VERIFY_ATTEMPTS = 2
RUNTIME_VERIFY_RETRY_SECONDS = 0.15
RUNTIME_ADAPTER_TIMEOUT_SECONDS = 10.0

# 146_l2 is a ROUTINE training-split calibration/demo example, not held-out
# evidence. 296_l2 remains the READY-keyed PRIORITY validation example.
# RETAKE/LIMITED examples are intentionally not accepted by this adapter, even
# in ESCALATION_ONLY mode.
GEMMA_ESCALATION_DEMO_IMAGE_SHA256 = frozenset(
    {
        "21ef6838c18ccfe8697a1e2f4a31d2cce2cb11eb2627995a977d5aaaa9aeeda7",
        "b154932d70e281d2b7e2998c52c4c6a4631095f90f48900660c32535b020efd9",
    }
)


class GemmaEscalationError(RuntimeError):
    """A recoverable local runtime, identity, or output-contract failure."""

    def __init__(
        self,
        message: str,
        *,
        reason: EscalationReason,
        executed: bool = False,
        model_available: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.executed = executed
        self.model_available = model_available


class _RejectRedirects(HTTPRedirectHandler):
    """Never let a loopback request redirect image bytes off device."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None


def _open_loopback(request: Request, timeout: float):
    # Ignore proxy environment variables as well as redirects. The configured
    # host is separately required to be a numeric loopback address.
    return build_opener(ProxyHandler({}), _RejectRedirects()).open(
        request, timeout=timeout
    )


def _runtime_url(api_url: str, path: str) -> str:
    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/v1/chat/completions"):
        trimmed = trimmed[: -len("/v1/chat/completions")]
    elif trimmed.endswith("/v1"):
        trimmed = trimmed[:-3]
    return f"{trimmed}/{path.lstrip('/')}"


def _chat_completions_url(api_url: str) -> str:
    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/v1/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


@dataclass(frozen=True)
class LocalGemmaEscalationAdapter:
    """Call one loopback llama.cpp server with one exact RetinaPriority LoRA."""

    api_url: str
    model_id: str
    lora_path: Path
    lora_sha256: str
    timeout_seconds: float = 90.0
    input_allowlist: frozenset[str] = field(
        default=GEMMA_ESCALATION_DEMO_IMAGE_SHA256
    )
    model_label: str = GEMMA_ESCALATION_MODEL_LABEL
    _runtime_identity_verified: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _runtime_lock: Any = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        parsed = urlparse(self.api_url)
        try:
            host_is_loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            host_is_loopback = False
        if (
            parsed.scheme != "http"
            or not host_is_loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path
            not in {"", "/", "/v1", "/v1/chat/completions"}
        ):
            raise ValueError(
                "RetinaPriority Gemma URL must be an uncredentialed HTTP loopback URL."
            )
        if not self.model_id or self.model_id.strip() != self.model_id:
            raise ValueError("RetinaPriority model alias must be explicit and exact.")
        resolved_lora = self.lora_path.expanduser().resolve()
        object.__setattr__(self, "lora_path", resolved_lora)
        if not resolved_lora.is_file():
            raise ValueError("RetinaPriority LoRA file is missing.")
        if (
            len(self.lora_sha256) != 64
            or self.lora_sha256.lower() != self.lora_sha256
            or any(character not in "0123456789abcdef" for character in self.lora_sha256)
        ):
            raise ValueError("RetinaPriority LoRA SHA-256 pin is invalid.")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("RetinaPriority timeout must be positive and finite.")
        if not self.input_allowlist or any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.input_allowlist
        ):
            raise ValueError("RetinaPriority input hash allowlist is invalid.")
        self._validate_opt_in()
        self._verify_lora_file()

    @staticmethod
    def _validate_opt_in() -> None:
        if os.getenv(RESEARCH_DEMO_OPT_IN) != "1":
            raise GemmaEscalationError(
                "RetinaPriority research-demo opt-in is not active.",
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_lora_file(self) -> None:
        try:
            observed = self._sha256(self.lora_path)
        except OSError as error:
            raise GemmaEscalationError(
                "RetinaPriority LoRA file is unavailable.",
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            ) from error
        if observed != self.lora_sha256:
            raise GemmaEscalationError(
                "RetinaPriority LoRA checksum mismatch.",
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("RETINA_ESCALATION_GEMMA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _fetch_json(
        self,
        request: Request,
        *,
        timeout: float,
        output_request: bool = False,
    ) -> Any:
        try:
            with _open_loopback(request, timeout=timeout) as response:
                raw = response.read(MAX_RUNTIME_JSON_BYTES + 1)
            if len(raw) > MAX_RUNTIME_JSON_BYTES:
                raise GemmaEscalationError(
                    "The loopback RetinaPriority response exceeded the size limit.",
                    reason=(
                        EscalationReason.INVALID_OUTPUT
                        if output_request
                        else EscalationReason.ARTIFACT_UNAVAILABLE
                    ),
                    executed=output_request,
                    model_available=output_request,
                )
            return json.loads(raw.decode("utf-8"))
        except HTTPError as error:
            raise GemmaEscalationError(
                "The loopback RetinaPriority server returned an HTTP error.",
                reason=EscalationReason.ADAPTER_ERROR,
                executed=output_request,
                model_available=output_request,
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise GemmaEscalationError(
                "The loopback RetinaPriority server is unreachable.",
                reason=EscalationReason.ADAPTER_ERROR,
                executed=output_request,
                model_available=False,
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise GemmaEscalationError(
                "The loopback RetinaPriority server returned invalid JSON.",
                reason=(
                    EscalationReason.INVALID_OUTPUT
                    if output_request
                    else EscalationReason.ARTIFACT_UNAVAILABLE
                ),
                executed=output_request,
                model_available=output_request,
            ) from error

    def _verify_runtime_once(self) -> None:
        headers = self._headers()
        health = self._fetch_json(
            Request(_runtime_url(self.api_url, "health"), headers=headers),
            timeout=min(3.0, self.timeout_seconds),
        )
        if not isinstance(health, dict) or health.get("status") != "ok":
            raise GemmaEscalationError(
                "RetinaPriority health contract failed.",
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            )

        models = self._fetch_json(
            Request(_runtime_url(self.api_url, "v1/models"), headers=headers),
            timeout=min(3.0, self.timeout_seconds),
        )
        try:
            data = models["data"]
            model_ids = [item["id"] for item in data]
        except (KeyError, TypeError) as error:
            raise GemmaEscalationError(
                "RetinaPriority model-list schema failed.",
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            ) from error
        if (
            not isinstance(data, list)
            or any(not isinstance(item, dict) or set(item).isdisjoint({"id"}) for item in data)
            or self.model_id not in model_ids
        ):
            raise GemmaEscalationError(
                "The exact RetinaPriority model alias is not loaded.",
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            )

        adapters = self._fetch_json(
            Request(_runtime_url(self.api_url, "lora-adapters"), headers=headers),
            # This identity endpoint restores an idle llama.cpp allocation on
            # the current server build. Cold wake regularly takes just over
            # three seconds on a 24-GB Mac, so do not use the short metadata
            # timeout here.
            timeout=min(RUNTIME_ADAPTER_TIMEOUT_SECONDS, self.timeout_seconds),
        )
        if (
            not isinstance(adapters, list)
            or len(adapters) != 1
            or not isinstance(adapters[0], dict)
            or adapters[0].get("path") != str(self.lora_path)
            or not isinstance(adapters[0].get("scale"), (int, float))
            or isinstance(adapters[0].get("scale"), bool)
            or float(adapters[0]["scale"]) != 1.0
        ):
            raise GemmaEscalationError(
                "The exact RetinaPriority LoRA path is not active at scale 1.",
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            )

    def _verify_runtime(self) -> None:
        # Re-check the opt-in and bound file for each health check and release
        # attempt so a resident process cannot continue after revocation or
        # artifact tampering. llama.cpp may briefly reject an identity endpoint
        # while its idle model allocation is waking, so repeat the *complete*
        # identity contract once. A persistent mismatch still fails closed.
        self._validate_opt_in()
        self._verify_lora_file()
        for attempt in range(RUNTIME_VERIFY_ATTEMPTS):
            try:
                self._verify_runtime_once()
                break
            except GemmaEscalationError as error:
                if (
                    error.reason is not EscalationReason.ADAPTER_ERROR
                    or attempt + 1 >= RUNTIME_VERIFY_ATTEMPTS
                ):
                    raise
                time.sleep(RUNTIME_VERIFY_RETRY_SECONDS)
        object.__setattr__(self, "_runtime_identity_verified", True)

    def _cached_runtime_is_sleeping(self) -> bool:
        """Check an already-verified sleeping server without waking its model."""

        if not self._runtime_identity_verified:
            return False
        self._validate_opt_in()
        self._verify_lora_file()
        props = self._fetch_json(
            Request(
                _runtime_url(self.api_url, "props"),
                headers=self._headers(),
            ),
            timeout=min(3.0, self.timeout_seconds),
        )
        return (
            isinstance(props, dict)
            and props.get("is_sleeping") is True
            and props.get("model_alias") == self.model_id
        )

    def _request_assessment(
        self, image_bytes: bytes, content_type: str
    ) -> dict[str, object]:
        body = {
            "model": self.model_id,
            "lora": [{"id": 0, "scale": 1.0}],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            # JSON-object mode still lets the model choose arbitrary keys (the
            # converted LoRA can otherwise emit {"priority": ...}). Constrain
            # decoding to the training contract, then independently revalidate
            # every value and the decision/next-step pairing in ``_normalize``.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "retinapriority_assessment",
                    "strict": True,
                    "schema": ESCALATION_RESPONSE_SCHEMA,
                },
            },
            "messages": [
                {"role": "system", "content": ESCALATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ESCALATION_USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{content_type};base64,"
                                    f"{base64.b64encode(image_bytes).decode('ascii')}"
                                )
                            },
                        },
                    ],
                },
            ],
        }
        envelope = self._fetch_json(
            Request(
                _chat_completions_url(self.api_url),
                data=json.dumps(body).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            ),
            timeout=self.timeout_seconds,
            output_request=True,
        )
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GemmaEscalationError(
                "RetinaPriority response envelope is invalid.",
                reason=EscalationReason.INVALID_OUTPUT,
                executed=True,
                model_available=True,
            ) from error
        if not isinstance(content, str):
            raise GemmaEscalationError(
                "RetinaPriority response content is not text.",
                reason=EscalationReason.INVALID_OUTPUT,
                executed=True,
                model_available=True,
            )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise GemmaEscalationError(
                "RetinaPriority output is not one strict JSON object.",
                reason=EscalationReason.INVALID_OUTPUT,
                executed=True,
                model_available=True,
            ) from error
        if not isinstance(payload, dict):
            raise GemmaEscalationError(
                "RetinaPriority output is not a JSON object.",
                reason=EscalationReason.INVALID_OUTPUT,
                executed=True,
                model_available=True,
            )
        return payload

    def _normalize(self, payload: dict[str, object]) -> EscalationAssessment:
        if set(payload) != {"confidence", "decision", "disclaimer", "next_step"}:
            raise GemmaEscalationError(
                "RetinaPriority output fields do not match the training contract.",
                reason=EscalationReason.INVALID_OUTPUT,
                executed=True,
                model_available=True,
            )
        decision = payload.get("decision")
        expected_next_step = {
            "ROUTINE": "Keep in the routine clinician review queue.",
            "PRIORITY": "Route for priority clinician review.",
        }
        if decision not in expected_next_step:
            raise GemmaEscalationError(
                "RetinaPriority decision is not ROUTINE or PRIORITY.",
                reason=EscalationReason.INVALID_OUTPUT,
                executed=True,
                model_available=True,
            )
        if (
            payload.get("confidence") is not None
            or payload.get("next_step") != expected_next_step[decision]
            or payload.get("disclaimer") != ESCALATION_DISCLAIMER
        ):
            raise GemmaEscalationError(
                "RetinaPriority output values do not match the training contract.",
                reason=EscalationReason.INVALID_OUTPUT,
                executed=True,
                model_available=True,
            )
        mapped = (
            EscalationDecision.PRIORITY_REVIEW
            if decision == "PRIORITY"
            else EscalationDecision.ROUTINE_REVIEW
        )
        return EscalationAssessment(
            decision=mapped,
            confidence=None,
            executed=True,
            model_available=True,
            release_allowed=True,
            reason=EscalationReason.COMPLETED,
            summary="Uncalibrated local RetinaPriority research inference completed.",
            instruction="A clinician makes the final review-order decision.",
            model=self.model_label,
        )

    def _assess_sync(
        self, image_bytes: bytes, content_type: str
    ) -> EscalationAssessment:
        # Health checks and inference share a single-slot llama.cpp runtime.
        # Serializing them prevents a browser health request from colliding
        # with the first completion while an idle model is being restored.
        with self._runtime_lock:
            self._verify_runtime()
            return self._normalize(self._request_assessment(image_bytes, content_type))

    async def assess(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        allow_experimental_input: bool = False,
    ) -> EscalationAssessment:
        del filename
        if content_type != "image/jpeg":
            return uncertain_escalation(
                reason=EscalationReason.ADAPTER_ERROR,
                summary="Review priority is uncertain for this unsupported input type.",
                instruction="Route the image to human prioritization.",
                model=self.model_label,
            )
        if (
            not allow_experimental_input
            and hashlib.sha256(image_bytes).hexdigest() not in self.input_allowlist
        ):
            return uncertain_escalation(
                reason=EscalationReason.ADAPTER_ERROR,
                summary="Review priority is uncertain outside the fixed DeepDRiD demo scope.",
                instruction="Use one of the fixed quality-passing DeepDRiD demo images.",
                model=self.model_label,
            )
        try:
            return await asyncio.to_thread(
                self._assess_sync, image_bytes, content_type
            )
        except GemmaEscalationError as error:
            return uncertain_escalation(
                reason=error.reason,
                summary="Review priority could not be assigned reliably.",
                instruction="Route the image to human prioritization.",
                model=self.model_label,
                executed=error.executed,
                model_available=error.model_available,
            )
        except Exception:
            return uncertain_escalation(
                reason=EscalationReason.ADAPTER_ERROR,
                summary="Review priority could not be assigned reliably.",
                instruction="Route the image to human prioritization.",
                model=self.model_label,
            )

    def runtime_status(self) -> dict[str, object]:
        try:
            with self._runtime_lock:
                if not self._cached_runtime_is_sleeping():
                    self._verify_runtime()
        except Exception:
            return {
                "status": "unavailable",
                "profile": GEMMA_ESCALATION_PROFILE,
                "model_verified": False,
                "lora_verified": False,
                "adapter_hash_verified": False,
                "release_enabled": False,
                "scope": GEMMA_ESCALATION_SCOPE,
                "input_scope": "fixed-deepdrid-quality-pass-demo-samples",
                "calibration": "uncalibrated-free-generation-experimental",
                "clinical_use": False,
                "model": self.model_label,
                "network_required": False,
                "loopback_http_required": True,
            }
        return {
            "status": "ready",
            "profile": GEMMA_ESCALATION_PROFILE,
            "model_verified": True,
            "lora_verified": True,
            "adapter_hash_verified": True,
            "release_enabled": True,
            "scope": GEMMA_ESCALATION_SCOPE,
            "input_scope": "fixed-deepdrid-quality-pass-demo-samples",
            "calibration": "uncalibrated-free-generation-experimental",
            "clinical_use": False,
            "model": self.model_label,
            "network_required": False,
            "loopback_http_required": True,
        }
