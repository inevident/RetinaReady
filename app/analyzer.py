"""Analysis engine boundary for demo and local Gemma inference.

The browser-facing response stays stable while the implementation can switch
between deterministic demo data and a llama.cpp-style OpenAI-compatible server.
Image bytes live in memory only and are never written by this module.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from demo_engine import analyze_demo


DECISIONS = frozenset({"READY", "LIMITED", "RETAKE"})
ISSUE_CODES = frozenset(
    {"artifact", "blur", "field_cutoff", "unsupported_modality", "uncertain"}
)
DISCLAIMER = "Technical image-quality assessment only; not a diagnosis."
QUALITY_ATTENTION_LABEL = (
    "Model quality attention \u2014 not pathology localization."
)
READY_MIN_CONFIDENCE = 0.90
READY_MIN_SCORE = 70

# Independent pins make the colocated bundle manifest an integrity description,
# not its own trust root. Updating any quality artifact requires an intentional
# code review and a matching pin change here.
QUALITY_SPECIALIST_MANIFEST_SHA256 = (
    "d1e51bea40dcbfee170920e73369c76ef10c8f49e20604869e91e996e53870a2"
)
QUALITY_SPECIALIST_ARTIFACT_SHA256 = (
    "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395",
    "84081ad06122a0354d0bd4c31cdc53052f1bdb4999fb706b7babcfe72b94d936",
    "84081ad06122a0354d0bd4c31cdc53052f1bdb4999fb706b7babcfe72b94d936",
)

# This specialist has no independently validated modality/OOD gate. The safe
# live profile is therefore intentionally limited to the four fixed DeepDRiD
# examples served by the app, rather than accepting arbitrary uploads.
SPECIALIST_DEMO_IMAGE_SHA256 = frozenset(
    {
        "21ef6838c18ccfe8697a1e2f4a31d2cce2cb11eb2627995a977d5aaaa9aeeda7",
        "b154932d70e281d2b7e2998c52c4c6a4631095f90f48900660c32535b020efd9",
        "70fb1638c1ea9e06b92b354d3e5e4891d9470832898bc7386a74fae803e7cff8",
        "24ab77726e2a0dc2eda976ecf3a852e59dcd5f361cdfd719c34f2c1fef62e589",
    }
)

SYSTEM_PROMPT = """You are RetinaReady, an offline technical image-quality assistant.
Assess only whether the supplied image is a color fundus photograph of sufficient
technical quality. Never diagnose disease, infer that an eye is healthy, recommend
treatment, or interpret retinal pathology. Return exactly one JSON object matching
the requested schema. Use LIMITED for an unsupported modality, an uncertain case,
or any case where you cannot assess quality safely."""

USER_PROMPT = """Assess this image for capture quality.
Return exactly:
{
  "decision": "READY" | "LIMITED" | "RETAKE",
  "confidence": number from 0 to 1 or null,
  "issues": array containing only "artifact", "blur", "field_cutoff",
            "unsupported_modality", or "uncertain",
  "scores": {
    "artifact": integer 0-100 or null,
    "clarity": integer 0-100 or null,
    "field_definition": integer 0-100 or null
  },
  "retake_instruction": string or null,
  "disclaimer": "Technical image-quality assessment only; not a diagnosis."
}
For every score, 100 means best technical quality. Do not add markdown or prose."""


class AnalyzerError(RuntimeError):
    """A recoverable local analysis failure."""


class Analyzer(Protocol):
    mode: str
    model_label: str

    def runtime_status(self) -> dict[str, object]: ...

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class DemoAnalyzer:
    mode: str = "demo"
    model_label: str = "Deterministic demo engine · no model loaded"

    def runtime_status(self) -> dict[str, object]:
        return {"status": "ready", "profile": "demo", "model_verified": False}

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> dict[str, object]:
        del allow_experimental_input
        return analyze_demo(
            image_bytes,
            filename=filename,
            content_type=content_type,
            scenario=scenario,
        )


def _chat_completions_url(api_url: str) -> str:
    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


def _runtime_url(api_url: str, path: str) -> str:
    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/v1/chat/completions"):
        trimmed = trimmed[: -len("/v1/chat/completions")]
    elif trimmed.endswith("/v1"):
        trimmed = trimmed[:-3]
    return f"{trimmed}/{path.lstrip('/')}"


def _extract_json_object(content: object) -> dict[str, object]:
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise AnalyzerError("The local model returned an unexpected response.")

    # Gemma 4 may prefix the answer with a thought/reasoning channel even when
    # thinking is disabled. Scanning for a decodable object also accepts a
    # ```json code fence without trusting or trying to interpret the prefix.
    decoder = json.JSONDecoder()
    fallback: dict[str, object] | None = None
    for start, character in enumerate(content):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if "decision" in parsed:
            return parsed
        if fallback is None:
            fallback = parsed
    if fallback is not None:
        return fallback
    raise AnalyzerError("The local model did not return a valid JSON object.")


def _number(value: object, *, field: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AnalyzerError(f"The local model returned an invalid {field}.")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise AnalyzerError(f"The local model returned an invalid {field}.")
    return numeric


def _score(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return round(_number(value, field=field, minimum=0, maximum=100))


def _limited_result(*, unsupported: bool = False) -> dict[str, object]:
    if unsupported:
        eyebrow = "Unsupported image"
        summary = "This does not appear to be a supported color fundus photograph."
        issues = ["Unsupported image type"]
        instruction = (
            "Upload a color fundus photograph. OCT and angiography images are not "
            "supported."
        )
    else:
        eyebrow = "Unable to assess"
        summary = "Capture quality could not be assessed reliably."
        issues = ["Assessment uncertain"]
        instruction = "Review the image or retake it before continuing."
    return {
        "status": "LIMITED",
        "eyebrow": eyebrow,
        "summary": summary,
        "confidence": None,
        "issues": issues,
        "instruction": instruction,
        "scores": None,
        "disclaimer": DISCLAIMER,
        "mode": "local-model",
    }


def _safe_instruction(decision: str, issue_codes: list[str]) -> str:
    """Derive capture guidance from validated codes, never model-authored prose."""

    issue_set = set(issue_codes)
    if decision == "READY":
        return "No retake needed. Continue to the normal human-review workflow."
    if "unsupported_modality" in issue_set:
        return (
            "Upload a color fundus photograph. OCT and angiography images are not "
            "supported."
        )
    if decision == "LIMITED":
        return "Have a trained operator review or recapture the image before continuing."

    actions: list[str] = []
    if "blur" in issue_set:
        actions.append("stabilize the camera and refocus")
    if "field_cutoff" in issue_set:
        actions.append("recenter the retinal field")
    if "artifact" in issue_set:
        actions.append("remove glare or other capture artifacts")
    if not actions:
        actions.append("refocus and recenter the retinal field")
    return f"{'; '.join(actions).capitalize()}, then retake the image."


def normalize_model_result(payload: dict[str, object]) -> dict[str, object]:
    """Convert the strict training contract into the browser result contract."""

    decision = str(payload.get("decision", "")).upper()
    if decision not in DECISIONS:
        raise AnalyzerError("The local model returned an invalid decision.")

    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list) or any(
        not isinstance(issue, str) or issue not in ISSUE_CODES for issue in raw_issues
    ):
        raise AnalyzerError("The local model returned invalid issue codes.")
    raw_issues = list(dict.fromkeys(raw_issues))

    confidence = payload.get("confidence")
    if confidence is not None:
        confidence = _number(confidence, field="confidence", minimum=0, maximum=1)

    # Unsupported and uncertain outputs are never allowed to become READY or
    # RETAKE through a model formatting mistake.
    if "unsupported_modality" in raw_issues:
        decision = "LIMITED"
    if "uncertain" in raw_issues:
        decision = "LIMITED"
    if decision == "LIMITED" and not {
        "unsupported_modality",
        "uncertain",
    }.intersection(raw_issues):
        raw_issues.append("uncertain")

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise AnalyzerError("The local model returned invalid quality scores.")
    normalized_scores = {
        "Artifact quality": _score(
            raw_scores.get("artifact"), field="artifact quality score"
        ),
        "Clarity": _score(raw_scores.get("clarity"), field="clarity score"),
        "Field": _score(
            raw_scores.get("field_definition"), field="field-definition score"
        ),
    }

    # Free-generation confidence is not the calibrated direct-logit
    # probability. Still, an obvious contradiction must never become READY.
    # READY is permitted only when the model reports no issue, high confidence,
    # and complete quality scores above a conservative floor; otherwise abstain.
    if decision == "READY" and (
        raw_issues
        or confidence is None
        or confidence < READY_MIN_CONFIDENCE
        or any(
            score is None or score < READY_MIN_SCORE
            for score in normalized_scores.values()
        )
    ):
        decision = "LIMITED"
        if "uncertain" not in raw_issues:
            raw_issues.append("uncertain")

    issue_labels = {
        "artifact": "Capture artifact",
        "blur": "Motion or focus blur",
        "field_cutoff": "Field not centered",
        "unsupported_modality": "Unsupported image type",
        "uncertain": "Assessment uncertain",
    }
    issues = [issue_labels[issue] for issue in raw_issues]

    scores = {
        label: value for label, value in normalized_scores.items() if value is not None
    } or None
    if decision == "LIMITED":
        # LIMITED means unsupported or uncertain in the model contract, so
        # displaying quality scores would imply a confidence we do not have.
        scores = None

    if decision == "READY":
        eyebrow = "Capture ready"
        summary = "This image is technically ready for clinical review."
    elif decision == "RETAKE":
        eyebrow = "Retake recommended"
        summary = "Technical quality is too low for a dependable review."
    elif "unsupported_modality" in raw_issues:
        eyebrow = "Unsupported image"
        summary = "This does not appear to be a supported color fundus photograph."
    else:
        eyebrow = "Unable to assess"
        summary = "Capture quality could not be assessed reliably."

    model_instruction = payload.get("retake_instruction")
    if model_instruction is not None and not isinstance(model_instruction, str):
        raise AnalyzerError("The local model returned an invalid retake instruction.")
    instruction = _safe_instruction(decision, raw_issues)

    return {
        "status": decision,
        "eyebrow": eyebrow,
        "summary": summary,
        "confidence": confidence,
        "issues": issues,
        "instruction": instruction,
        "scores": scores,
        "disclaimer": DISCLAIMER,
        "mode": "local-model",
    }


@dataclass(frozen=True)
class LocalOpenAIAnalyzer:
    """Call an OpenAI-compatible multimodal endpoint on this machine."""

    api_url: str
    model_id: str
    display_label: str | None = None
    model_profile: str = "local-unknown"
    timeout_seconds: float = 90.0
    mode: str = "local-model"

    @property
    def model_label(self) -> str:
        return self.display_label or self.model_id

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("GEMMA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def runtime_status(self) -> dict[str, object]:
        """Verify both llama.cpp readiness and the configured model identity."""

        lora_verified = False
        try:
            health_request = Request(
                _runtime_url(self.api_url, "health"), headers=self._headers()
            )
            with urlopen(health_request, timeout=3) as response:
                health = json.loads(response.read().decode("utf-8"))
            if not isinstance(health, dict) or health.get("status") != "ok":
                raise ValueError("runtime is not ready")

            models_request = Request(
                _runtime_url(self.api_url, "v1/models"), headers=self._headers()
            )
            with urlopen(models_request, timeout=3) as response:
                models = json.loads(response.read().decode("utf-8"))
            model_ids = {
                item.get("id")
                for item in models.get("data", [])
                if isinstance(item, dict)
            }
            if self.model_id not in model_ids:
                raise ValueError("configured model alias is not loaded")
            if self.model_profile == "tuned-lora":
                lora_request = Request(
                    _runtime_url(self.api_url, "lora-adapters"),
                    headers=self._headers(),
                )
                with urlopen(lora_request, timeout=3) as response:
                    adapters = json.loads(response.read().decode("utf-8"))
                if not isinstance(adapters, list) or not any(
                    isinstance(adapter, dict) and adapter.get("path")
                    for adapter in adapters
                ):
                    raise ValueError("tuned profile has no loaded LoRA adapter")
                lora_verified = True
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return {
                "status": "unavailable",
                "profile": self.model_profile,
                "model_verified": False,
                "lora_verified": False,
            }
        return {
            "status": "ready",
            "profile": self.model_profile,
            "model_verified": True,
            "lora_verified": lora_verified,
        }

    def __post_init__(self) -> None:
        hostname = (urlparse(self.api_url).hostname or "").lower()
        allow_remote = os.getenv("RETINA_ALLOW_REMOTE_MODEL") == "1"
        if hostname not in {"localhost", "127.0.0.1", "::1"} and not allow_remote:
            raise ValueError(
                "GEMMA_API_URL must use localhost unless RETINA_ALLOW_REMOTE_MODEL=1."
            )

    def _request(
        self,
        image_bytes: bytes,
        content_type: str,
        quality_context: str | None = None,
    ) -> dict[str, object]:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        user_prompt = USER_PROMPT
        if quality_context:
            user_prompt = f"{quality_context}\n\n{USER_PROMPT}"
        body = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": 384,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
        }
        request = Request(
            _chat_completions_url(self.api_url),
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise AnalyzerError(
                f"The local model endpoint returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise AnalyzerError("The local Gemma endpoint is not reachable.") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AnalyzerError("The local model endpoint returned invalid JSON.") from error

        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise AnalyzerError("The local model response is missing its output.") from error
        try:
            return normalize_model_result(_extract_json_object(content))
        except AnalyzerError:
            # A malformed or contradictory model answer is uncertainty, not a
            # capture failure and not a reason to invent numeric scores.
            return _limited_result()

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> dict[str, object]:
        del filename, scenario, allow_experimental_input
        return await asyncio.to_thread(self._request, image_bytes, content_type)


@dataclass(frozen=True)
class HybridLocalAnalyzer:
    """Use a frozen conservative visual gate and Gemma as a safety veto."""

    gemma: LocalOpenAIAnalyzer
    specialist: Any
    mode: str = "hybrid-local"

    @property
    def model_label(self) -> str:
        return f"{self.gemma.model_label} + quality specialist"

    def runtime_status(self) -> dict[str, object]:
        status = dict(self.gemma.runtime_status())
        status["specialist_verified"] = bool(
            getattr(self.specialist, "bundle_verified", False)
        )
        status["profile"] = "hybrid-tuned-lora"
        return status

    @staticmethod
    def _decision_trace(
        specialist: str, gemma: str, policy: str
    ) -> dict[str, str]:
        """Return a deterministic trace; model-authored prose never enters it."""

        return {
            "specialist": specialist,
            "gemma": gemma,
            "policy": policy,
        }

    @staticmethod
    def _validated_quality_attention(
        assessment: Any,
    ) -> dict[str, str] | None:
        """Expose only the bounded, local quality-attention payload."""

        attention = getattr(assessment, "quality_attention", None)
        if not isinstance(attention, dict):
            return None
        factor = attention.get("factor")
        factor_label = attention.get("factor_label")
        method = attention.get("method")
        image_data_url = attention.get("image_data_url")
        if (
            attention.get("label") != QUALITY_ATTENTION_LABEL
            or factor not in {"artifact", "clarity", "field_definition"}
            or not isinstance(factor_label, str)
            or method
            not in {"factor-grad-cam", "factor-gradient-sensitivity"}
            or not isinstance(image_data_url, str)
            or not image_data_url.startswith("data:image/png;base64,")
        ):
            return None
        return {
            "label": QUALITY_ATTENTION_LABEL,
            "factor": factor,
            "factor_label": factor_label,
            "method": method,
            "image_data_url": image_data_url,
        }

    @staticmethod
    def _specialist_result(assessment: Any) -> dict[str, object]:
        issue_labels = {
            "artifact": "Capture artifact",
            "blur": "Motion or focus blur",
            "field_cutoff": "Field not centered",
            "uncertain": "Assessment uncertain",
        }
        decision = assessment.decision
        issues = [issue_labels[code] for code in assessment.issue_codes]
        if decision == "READY":
            eyebrow = "Capture ready"
            summary = "This image is technically ready for clinical review."
        elif decision == "RETAKE":
            eyebrow = "Retake recommended"
            summary = "Technical quality is too low for a dependable review."
        else:
            eyebrow = "Unable to assess"
            summary = "The local models disagree or confidence is insufficient."
        # The sigmoid is a ranking score, not a calibrated probability. Keep
        # the API confidence null until reliability calibration is evaluated.
        confidence: float | None = None
        scores = (
            None
            if decision == "LIMITED"
            else {
                "Artifact quality": assessment.scores["artifact"],
                "Clarity": assessment.scores["clarity"],
                "Field": assessment.scores["field_definition"],
            }
        )
        result: dict[str, object] = {
            "status": decision,
            "eyebrow": eyebrow,
            "summary": summary,
            "confidence": confidence,
            "issues": issues,
            "instruction": _safe_instruction(decision, assessment.issue_codes),
            "scores": scores,
            "disclaimer": DISCLAIMER,
            "mode": "hybrid-local",
        }
        quality_attention = HybridLocalAnalyzer._validated_quality_attention(
            assessment
        )
        if quality_attention is not None:
            result["quality_attention"] = quality_attention
        return result

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> dict[str, object]:
        del filename, scenario, allow_experimental_input
        try:
            assessment = await asyncio.to_thread(self.specialist.assess, image_bytes)
        except (ValueError, OSError):
            result = _limited_result(unsupported=True)
            result["mode"] = "hybrid-local"
            result["decision_trace"] = self._decision_trace(
                "Input rejected", "Skipped", "LIMITED"
            )
            return result
        specialist_result = self._specialist_result(assessment)

        # Gemma can veto an unsupported, uncertain, or conflicting specialist
        # result, but free generation can never promote a LIMITED gate.
        if assessment.decision == "LIMITED":
            specialist_result["decision_trace"] = self._decision_trace(
                "Abstained", "Skipped", "LIMITED"
            )
            return specialist_result

        try:
            gemma_result = await asyncio.to_thread(
                self.gemma._request,
                image_bytes,
                content_type,
                assessment.prompt_context(),
            )
        except AnalyzerError:
            # The frozen specialist remains available, but the hybrid
            # contract requires Gemma confirmation before releasing READY or
            # RETAKE. A local-server timeout therefore abstains instead of
            # turning a transient infrastructure failure into a decision.
            specialist_result.update(
                {
                    "status": "LIMITED",
                    "eyebrow": "Unable to assess",
                    "summary": (
                        "Gemma confirmation is unavailable, so this image needs "
                        "human review."
                    ),
                    "confidence": None,
                    "issues": ["Assessment uncertain"],
                    "instruction": _safe_instruction("LIMITED", ["uncertain"]),
                    "scores": None,
                }
            )
            specialist_result.pop("quality_attention", None)
            specialist_result["decision_trace"] = self._decision_trace(
                f"{assessment.decision} candidate", "No confirmation", "LIMITED"
            )
            return specialist_result

        if gemma_result["status"] != assessment.decision:
            specialist_result.update(
                {
                    "status": "LIMITED",
                    "eyebrow": "Unable to assess",
                    "summary": "The local models disagree, so this image needs review.",
                    "confidence": None,
                    "issues": ["Assessment uncertain"],
                    "instruction": _safe_instruction("LIMITED", ["uncertain"]),
                    "scores": None,
                }
            )
            specialist_result.pop("quality_attention", None)
            specialist_result["decision_trace"] = self._decision_trace(
                f"{assessment.decision} candidate", "No confirmation", "LIMITED"
            )
            return specialist_result

        # For a confirmed RETAKE, Gemma's normalized issue labels can add
        # explanation while the frozen specialist retains final authority.
        if assessment.decision == "RETAKE":
            specialist_result["issues"] = list(
                dict.fromkeys(
                    [*specialist_result["issues"], *gemma_result.get("issues", [])]
                )
            )
        specialist_result["decision_trace"] = self._decision_trace(
            f"{assessment.decision} candidate", "Confirmed", assessment.decision
        )
        return specialist_result


@dataclass(frozen=True)
class SpecialistLocalAnalyzer:
    """Run only the frozen, exact-hash retinal quality specialist locally.

    This mode is deliberately independent of Gemma so the complete quality
    gate can run on a memory-constrained laptop.  A missing or changed bundle,
    malformed assessment, decode failure, or inference failure is an
    abstention (`LIMITED`), never a fabricated READY/RETAKE decision.
    """

    specialist: Any | None
    bundle_paths: tuple[Path, Path, Path] | None = None
    input_allowlist: frozenset[str] | None = None
    display_label: str | None = None
    mode: str = "specialist-local"

    @property
    def model_label(self) -> str:
        return self.display_label or "RetinaReady frozen quality specialist · Local"

    def _bundle_is_verified(self) -> bool:
        """Re-verify the allowlisted files instead of trusting a stale flag."""

        if self.specialist is None or not bool(
            getattr(self.specialist, "bundle_verified", False)
        ):
            return False
        if self.bundle_paths is None:
            # Test doubles can omit paths. Production construction always pins
            # all three paths and therefore executes the verifier below.
            return True
        backbone_path, decision_head_path, factor_head_path = self.bundle_paths
        try:
            manifest_path = decision_head_path.parent / "manifest.json"
            pinned_paths = (
                manifest_path,
                backbone_path,
                decision_head_path,
                factor_head_path,
            )
            pinned_hashes = (
                QUALITY_SPECIALIST_MANIFEST_SHA256,
                *QUALITY_SPECIALIST_ARTIFACT_SHA256,
            )
            return all(
                path.is_file() and self._sha256(path) == expected
                for path, expected in zip(pinned_paths, pinned_hashes, strict=True)
            )
        except Exception:
            return False

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def runtime_status(self) -> dict[str, object]:
        verified = self._bundle_is_verified()
        return {
            "status": "ready" if verified else "unavailable",
            "profile": "quality-specialist",
            "model_verified": verified,
            "lora_verified": False,
            "specialist_verified": verified,
            "privacy": "local-only",
            "network_required": False,
            "input_scope": (
                "fixed-deepdrid-demo-samples"
                if self.input_allowlist is not None
                else "caller-managed"
            ),
            "detail": (
                "Exact-hash quality-specialist bundle loaded."
                if verified
                else "Exact-hash quality-specialist bundle unavailable."
            ),
        }

    @staticmethod
    def _assessment_is_valid(assessment: Any) -> bool:
        decision = getattr(assessment, "decision", None)
        issue_codes = getattr(assessment, "issue_codes", None)
        scores = getattr(assessment, "scores", None)
        ready_score = getattr(assessment, "ready_score", None)
        ready_threshold = getattr(assessment, "ready_threshold", None)
        retake_threshold = getattr(assessment, "retake_threshold", None)

        if decision not in DECISIONS:
            return False
        if not isinstance(issue_codes, list) or any(
            code not in {"artifact", "blur", "field_cutoff", "uncertain"}
            for code in issue_codes
        ):
            return False
        if len(issue_codes) != len(set(issue_codes)):
            return False
        if decision == "READY" and issue_codes:
            return False
        if decision == "LIMITED" and issue_codes != ["uncertain"]:
            return False
        if decision == "RETAKE" and "uncertain" in issue_codes:
            return False

        if not isinstance(scores, dict) or set(scores) != {
            "artifact",
            "clarity",
            "field_definition",
        }:
            return False
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 100
            for value in scores.values()
        ):
            return False

        numeric_policy = (ready_score, ready_threshold, retake_threshold)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
            for value in numeric_policy
        ):
            return False
        retake = float(retake_threshold)
        ready = float(ready_threshold)
        score = float(ready_score)
        if not retake < ready:
            return False
        expected_decision = (
            "READY" if score > ready else "RETAKE" if score < retake else "LIMITED"
        )
        return decision == expected_decision

    def _limited(
        self,
        *,
        unsupported: bool = False,
        unavailable: bool = False,
        outside_scope: bool = False,
        trace: str | None = None,
    ) -> dict[str, object]:
        result = _limited_result(unsupported=unsupported)
        result["mode"] = self.mode
        if unavailable:
            result.update(
                {
                    "eyebrow": "Quality gate unavailable",
                    "summary": (
                        "The exact-hash local quality specialist is unavailable, "
                        "so this image needs human review."
                    ),
                }
            )
        elif outside_scope:
            result.update(
                {
                    "eyebrow": "Outside demo dataset",
                    "summary": (
                        "Specialist-only mode accepts only the fixed DeepDRiD "
                        "demo images because no modality detector is loaded."
                    ),
                    "issues": ["Outside validated demo set"],
                    "instruction": "Choose one of the fixed DeepDRiD sample buttons.",
                }
            )
        elif not unsupported:
            result["summary"] = (
                "The local quality specialist could not assess this image reliably."
            )
        result["decision_trace"] = {
            "specialist": trace or ("Unavailable" if unavailable else "Input rejected"),
            "gemma": "Not used",
            "policy": "LIMITED",
        }
        return result

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> dict[str, object]:
        del filename, scenario
        if content_type not in {"image/jpeg", "image/png", "image/webp"}:
            return self._limited(unsupported=True)
        if not self._bundle_is_verified():
            return self._limited(unavailable=True)
        if (
            not allow_experimental_input
            and self.input_allowlist is not None
            and hashlib.sha256(image_bytes).hexdigest() not in self.input_allowlist
        ):
            return self._limited(
                unsupported=True,
                outside_scope=True,
                trace="Outside dataset scope",
            )

        try:
            assessment = await asyncio.to_thread(self.specialist.assess, image_bytes)
        except ValueError as error:
            # QualitySpecialist uses this exact prefix only for PIL decode
            # failures; other ValueErrors are inference/policy failures.
            decode_failure = str(error).startswith("specialist could not decode")
            return self._limited(
                unsupported=decode_failure,
                trace="Input rejected" if decode_failure else "Inference failed",
            )
        except Exception:
            return self._limited(trace="Inference failed")

        try:
            assessment_valid = self._assessment_is_valid(assessment)
        except Exception:
            assessment_valid = False
        if not assessment_valid:
            return self._limited(trace="Invalid output")
        try:
            result = HybridLocalAnalyzer._specialist_result(assessment)
        except Exception:
            return self._limited(trace="Invalid output")

        result["mode"] = self.mode
        if assessment.decision == "LIMITED":
            result["summary"] = (
                "The quality specialist abstained because confidence is insufficient."
            )
            specialist_trace = "Abstained"
        else:
            specialist_trace = f"{assessment.decision} decision"
        result["decision_trace"] = {
            "specialist": specialist_trace,
            "gemma": "Not used",
            "policy": assessment.decision,
        }
        return result


def build_analyzer() -> Analyzer:
    """Create the configured engine once at app startup."""

    analyzer_mode = os.getenv("RETINA_ANALYZER", "auto").strip().lower()
    api_url = os.getenv("GEMMA_API_URL", "").strip()
    if analyzer_mode == "specialist":
        from quality_specialist import QualitySpecialist

        project_root = Path(__file__).resolve().parents[1]
        specialist_dir = Path(
            os.getenv(
                "RETINA_SPECIALIST_DIR",
                project_root / "models" / "retinaready-quality-specialist",
            )
        )
        bundle_paths = (
            specialist_dir / "densenet121-a639ec97.pth",
            specialist_dir / "decision-head.pt",
            specialist_dir / "factor-head.pt",
        )
        specialist: Any | None
        try:
            specialist = QualitySpecialist(
                backbone_path=bundle_paths[0],
                decision_head_path=bundle_paths[1],
                factor_head_path=bundle_paths[2],
                device=os.getenv("RETINA_SPECIALIST_DEVICE", "cpu"),
            )
        except Exception:
            specialist = None
        return SpecialistLocalAnalyzer(
            specialist=specialist,
            bundle_paths=bundle_paths,
            input_allowlist=SPECIALIST_DEMO_IMAGE_SHA256,
            display_label=os.getenv("RETINA_MODEL_LABEL") or None,
        )
    if analyzer_mode == "local" or (analyzer_mode == "auto" and api_url):
        if not api_url:
            raise ValueError("GEMMA_API_URL is required when RETINA_ANALYZER=local.")
        gemma = LocalOpenAIAnalyzer(
            api_url=api_url,
            model_id=os.getenv("MODEL_ID", "retinaready-gemma4-26b"),
            display_label=os.getenv("RETINA_MODEL_LABEL") or None,
            model_profile=os.getenv("RETINA_MODEL_PROFILE", "local-unknown"),
            timeout_seconds=float(os.getenv("GEMMA_TIMEOUT_SECONDS", "90")),
        )
        if os.getenv("RETINA_HYBRID", "0") == "1":
            from quality_specialist import QualitySpecialist

            project_root = Path(__file__).resolve().parents[1]
            specialist_dir = Path(
                os.getenv(
                    "RETINA_SPECIALIST_DIR",
                    project_root / "models" / "retinaready-quality-specialist",
                )
            )
            specialist = QualitySpecialist(
                backbone_path=specialist_dir / "densenet121-a639ec97.pth",
                decision_head_path=specialist_dir / "decision-head.pt",
                factor_head_path=specialist_dir / "factor-head.pt",
                device=os.getenv("RETINA_SPECIALIST_DEVICE", "cpu"),
            )
            return HybridLocalAnalyzer(gemma=gemma, specialist=specialist)
        return gemma
    return DemoAnalyzer(
        model_label=os.getenv(
            "RETINA_MODEL_LABEL", "Deterministic demo engine · no model loaded"
        )
    )
