"""Exact-hash local runtime for the opted-in escalation research demo.

This adapter does not promote or modify the source experimental artifact. Its
authority comes only from a separate, exact-hash promotion manifest scoped to
a nonclinical hackathon research demonstration. Every integrity or inference
failure returns UNCERTAIN and releases no queue decision.
"""

from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from workflow import (
    EscalationAssessment,
    EscalationDecision,
    EscalationReason,
    uncertain_escalation,
)


PROMOTION_SCOPE = "nonclinical-hackathon-research-demo-only"
PROMOTION_PROFILE = "exact-hash-local-research-demo"
PREPROCESSING_CONTRACT = (
    "crop-black-border-threshold15-buffer20; square-pad; resize-512-antialias; "
    "to-tensor; normalize-0.5"
)


class EscalationIntegrityError(ValueError):
    """An allowlist, checksum, artifact, or report contract failed."""


class LocalEscalationSpecialistAdapter:
    """Frozen DenseNet-121 plus the allowlisted escalation MLP ensemble."""

    model_label = "RetinaPriority DenseNet research demo · nonclinical"

    def __init__(
        self,
        *,
        project_root: Path,
        promotion_manifest_path: Path,
        device: str = "cpu",
    ) -> None:
        self._project_root = project_root.resolve()
        self._promotion_manifest_path = promotion_manifest_path.resolve()
        if not self._promotion_manifest_path.is_relative_to(self._project_root):
            raise EscalationIntegrityError("promotion manifest must stay inside project")
        self._promotion = self._load_promotion_manifest()
        self._validate_opt_in(self._promotion)
        self._bound_paths = self._resolve_bound_paths(self._promotion)
        self._verify_bound_files()

        report = self._load_json(self._bound_paths["report"])
        self._validate_report(report, self._promotion)

        import torch
        from torchvision import models

        self._torch = torch
        self._device = torch.device(device)
        artifact = torch.load(
            self._bound_paths["artifact"], map_location="cpu", weights_only=True
        )
        self._validate_artifact(artifact, self._promotion, torch)

        backbone = models.densenet121(weights=None)
        backbone_state = torch.load(
            self._bound_paths["backbone"], map_location="cpu", weights_only=True
        )
        self._migrate_legacy_densenet_keys(backbone_state)
        backbone.load_state_dict(backbone_state)
        backbone.classifier = torch.nn.Identity()
        self._backbone = backbone.eval().to(self._device)

        self._heads = self._load_heads(artifact)
        self._feature_mean = artifact["feature_mean"].to(self._device)
        self._feature_std = artifact["feature_std"].to(self._device)
        self._policy = artifact["policy"]
        for module in [self._backbone, *self._heads]:
            module.requires_grad_(False)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EscalationIntegrityError(f"invalid JSON binding: {path.name}") from error
        if not isinstance(payload, dict):
            raise EscalationIntegrityError(f"JSON binding is not an object: {path.name}")
        return payload

    def _load_promotion_manifest(self) -> dict[str, Any]:
        if not self._promotion_manifest_path.is_file():
            raise EscalationIntegrityError("promotion manifest is missing")
        manifest = self._load_json(self._promotion_manifest_path)
        try:
            if manifest["schema_version"] != 1:
                raise EscalationIntegrityError("unsupported promotion manifest schema")
            if manifest["scope"] != PROMOTION_SCOPE:
                raise EscalationIntegrityError("promotion scope is not research-demo-only")
            if manifest["network_required"] is not False:
                raise EscalationIntegrityError("promotion manifest permits network use")
            if manifest["fail_closed_decision"] != "UNCERTAIN":
                raise EscalationIntegrityError("promotion manifest is not fail closed")
            if manifest["allowed_released_decisions"] != [
                "ROUTINE_REVIEW",
                "PRIORITY_REVIEW",
            ]:
                raise EscalationIntegrityError("unexpected released-decision allowlist")
            if manifest["preprocessing_contract"] != PREPROCESSING_CONTRACT:
                raise EscalationIntegrityError("preprocessing contract mismatch")
            opt_in = manifest["required_opt_in"]
            if opt_in != {
                "environment_variable": "RETINA_ENABLE_ESCALATION_RESEARCH_DEMO",
                "value": "1",
            }:
                raise EscalationIntegrityError("unexpected opt-in contract")
        except (KeyError, TypeError) as error:
            raise EscalationIntegrityError("invalid promotion manifest") from error
        return manifest

    @staticmethod
    def _validate_opt_in(manifest: dict[str, Any]) -> None:
        opt_in = manifest["required_opt_in"]
        if os.getenv(opt_in["environment_variable"]) != opt_in["value"]:
            raise EscalationIntegrityError("research-demo opt-in is not active")

    def _resolve_bound_paths(self, manifest: dict[str, Any]) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        try:
            bindings = manifest["bindings"]
            if set(bindings) != {"artifact", "report", "backbone"}:
                raise EscalationIntegrityError("unexpected promotion bindings")
            for name, binding in bindings.items():
                relative = Path(binding["file"])
                checksum = binding["sha256"]
                if relative.is_absolute() or ".." in relative.parts:
                    raise EscalationIntegrityError("binding must be project-relative")
                if (
                    not isinstance(checksum, str)
                    or len(checksum) != 64
                    or any(character not in "0123456789abcdef" for character in checksum)
                ):
                    raise EscalationIntegrityError("invalid SHA-256 allowlist value")
                path = (self._project_root / relative).resolve()
                if not path.is_relative_to(self._project_root):
                    raise EscalationIntegrityError("binding escapes project root")
                paths[name] = path
        except (KeyError, TypeError) as error:
            raise EscalationIntegrityError("invalid promotion bindings") from error
        return paths

    def _verify_bound_files(self) -> None:
        for name, path in self._bound_paths.items():
            if not path.is_file():
                raise EscalationIntegrityError(f"bound {name} file is missing")
            expected = self._promotion["bindings"][name]["sha256"]
            if self._sha256(path) != expected:
                raise EscalationIntegrityError(f"bound {name} checksum mismatch")

    @staticmethod
    def _validate_report(
        report: dict[str, Any], promotion: dict[str, Any]
    ) -> None:
        try:
            artifact_contract = promotion["artifact_contract"]
            policy_contract = promotion["policy_contract"]
            if report["schema_version"] != 1:
                raise EscalationIntegrityError("unsupported escalation report schema")
            if report["status"] != "experimental-only; not integrated":
                raise EscalationIntegrityError("report status is not experimental-only")
            if report["recommendation"]["runtime_integration"] != "do-not-integrate":
                raise EscalationIntegrityError("report runtime recommendation changed")
            if report["recommendation"]["clinical_use"] != "not-authorized":
                raise EscalationIntegrityError("report clinical-use status changed")
            if report["model"]["architecture"] != artifact_contract["architecture"]:
                raise EscalationIntegrityError("report architecture mismatch")
            if (
                report["model"]["backbone_sha256"]
                != promotion["bindings"]["backbone"]["sha256"]
            ):
                raise EscalationIntegrityError("report backbone mismatch")
            policy = report["policy"]
            for key in (
                "routine_if_score_strictly_less_than",
                "priority_if_score_strictly_greater_than",
            ):
                if policy[key] != policy_contract[key]:
                    raise EscalationIntegrityError("report threshold mismatch")
        except (KeyError, TypeError) as error:
            raise EscalationIntegrityError("invalid escalation report contract") from error

    @staticmethod
    def _validate_artifact(
        artifact: dict[str, Any], promotion: dict[str, Any], torch: Any
    ) -> None:
        try:
            contract = promotion["artifact_contract"]
            policy_contract = promotion["policy_contract"]
            for key in (
                "schema_version",
                "experimental_only",
                "runtime_integration_authorized",
                "diagnostic_use_authorized",
                "architecture",
                "input_dim",
                "hidden_dim",
            ):
                if artifact[key] != contract[key]:
                    raise EscalationIntegrityError(f"artifact contract mismatch: {key}")
            members = artifact["members"]
            if (
                not isinstance(members, list)
                or len(members) != contract["ensemble_members"]
            ):
                raise EscalationIntegrityError("artifact ensemble mismatch")
            feature_mean = artifact["feature_mean"]
            feature_std = artifact["feature_std"]
            expected_shape = (1, contract["input_dim"])
            if tuple(feature_mean.shape) != expected_shape:
                raise EscalationIntegrityError("artifact feature mean shape mismatch")
            if tuple(feature_std.shape) != expected_shape:
                raise EscalationIntegrityError("artifact feature std shape mismatch")
            if not bool(torch.isfinite(feature_mean).all()):
                raise EscalationIntegrityError("artifact feature mean is non-finite")
            if not bool(torch.isfinite(feature_std).all()) or not bool(
                (feature_std > 0).all()
            ):
                raise EscalationIntegrityError("artifact feature std is invalid")
            policy = artifact["policy"]
            if policy["routine_if_score_strictly_less_than"] != policy_contract[
                "routine_if_score_strictly_less_than"
            ]:
                raise EscalationIntegrityError("artifact ROUTINE threshold mismatch")
            if policy["priority_if_score_strictly_greater_than"] != policy_contract[
                "priority_if_score_strictly_greater_than"
            ]:
                raise EscalationIntegrityError("artifact PRIORITY threshold mismatch")
        except (KeyError, TypeError, AttributeError) as error:
            raise EscalationIntegrityError("invalid escalation artifact contract") from error

    @staticmethod
    def _migrate_legacy_densenet_keys(state: dict[str, Any]) -> None:
        legacy_pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\."
            r"((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state):
            match = legacy_pattern.match(key)
            if match:
                state[match.group(1) + match.group(2)] = state.pop(key)

    def _load_heads(self, artifact: dict[str, Any]) -> list[Any]:
        torch = self._torch
        input_dim = int(artifact["input_dim"])
        hidden_dim = int(artifact["hidden_dim"])

        def make_head() -> Any:
            if hidden_dim:
                return torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.15),
                    torch.nn.Linear(hidden_dim, 1),
                )
            return torch.nn.Linear(input_dim, 1)

        heads = []
        for state in artifact["members"]:
            head = make_head()
            head.load_state_dict(state)
            heads.append(head.eval().to(self._device))
        return heads

    @staticmethod
    def _crop_black_border(image: Any, np: Any, threshold: int = 15) -> Any:
        """Exact baseline crop: threshold 15 and 20-pixel safety buffer."""

        array = np.asarray(image.convert("RGB"))
        visible = array.mean(axis=-1) > threshold
        ys, xs = np.where(visible)
        if not len(xs) or not len(ys):
            return image.convert("RGB")
        buffer = 20
        return image.convert("RGB").crop(
            (
                max(0, int(xs.min()) - buffer),
                max(0, int(ys.min()) - buffer),
                min(array.shape[1], int(xs.max()) + buffer + 1),
                min(array.shape[0], int(ys.max()) + buffer + 1),
            )
        )

    def _preprocess(self, image: Any) -> Any:
        """Exact baseline square-pad/512/[-1,1] preprocessing contract."""

        import numpy as np
        from torchvision.transforms import functional

        image = self._crop_black_border(image, np)
        width, height = image.size
        if width > height:
            delta = width - height
            padding = [0, delta // 2, 0, delta - delta // 2]
        else:
            delta = height - width
            padding = [delta // 2, 0, delta - delta // 2, 0]
        image = functional.pad(image, padding)
        image = functional.resize(image, [512, 512], antialias=True)
        tensor = functional.to_tensor(image)
        return functional.normalize(tensor, [0.5] * 3, [0.5] * 3)

    def _score(self, image_bytes: bytes) -> float:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = self._preprocess(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError("escalation specialist could not decode image") from error

        torch = self._torch
        with torch.inference_mode():
            features = self._backbone(tensor.unsqueeze(0).to(self._device)).float()
            standardized = (features - self._feature_mean) / self._feature_std
            score = torch.stack(
                [torch.sigmoid(head(standardized)[0]) for head in self._heads]
            ).mean()
        value = float(score.detach().cpu())
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("escalation specialist produced invalid score")
        return value

    def _assessment_from_score(self, score: float) -> EscalationAssessment:
        routine_threshold = float(
            self._policy["routine_if_score_strictly_less_than"]
        )
        priority_threshold = float(
            self._policy["priority_if_score_strictly_greater_than"]
        )
        if score < routine_threshold:
            decision = EscalationDecision.ROUTINE_REVIEW
        elif score > priority_threshold:
            decision = EscalationDecision.PRIORITY_REVIEW
        else:
            return uncertain_escalation(
                reason=EscalationReason.MODEL_ABSTAINED,
                summary="The local review-priority specialist abstained.",
                instruction="Route the image to human prioritization.",
                model=self.model_label,
                executed=True,
                model_available=True,
            )
        return EscalationAssessment(
            decision=decision,
            confidence=None,
            executed=True,
            model_available=True,
            release_allowed=True,
            reason=EscalationReason.COMPLETED,
            summary="Local review-priority research assessment completed.",
            instruction="A clinician makes the final review-order decision.",
            model=self.model_label,
        )

    def _assess_sync(self, image_bytes: bytes) -> EscalationAssessment:
        # Re-hash all bound files for every release attempt. This keeps a model
        # already resident in memory from continuing after on-disk tampering.
        self._validate_opt_in(self._promotion)
        self._verify_bound_files()
        return self._assessment_from_score(self._score(image_bytes))

    async def assess(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        allow_experimental_input: bool = False,
    ) -> EscalationAssessment:
        del filename, allow_experimental_input
        if content_type not in {"image/jpeg", "image/png", "image/webp"}:
            return uncertain_escalation(
                reason=EscalationReason.ADAPTER_ERROR,
                summary="Review priority is uncertain for this input type.",
                instruction="Route the image to human prioritization.",
                model=self.model_label,
            )
        try:
            return await asyncio.to_thread(self._assess_sync, image_bytes)
        except EscalationIntegrityError:
            return uncertain_escalation(
                reason=EscalationReason.ARTIFACT_UNAVAILABLE,
                summary="Review priority is uncertain because artifact verification failed.",
                instruction="Use the normal clinician-review queue.",
                model=self.model_label,
            )
        except Exception:
            return uncertain_escalation(
                reason=EscalationReason.ADAPTER_ERROR,
                summary="Review priority is uncertain because local inference failed safely.",
                instruction="Route the image to human prioritization.",
                model=self.model_label,
                executed=True,
                model_available=True,
            )

    def runtime_status(self) -> dict[str, object]:
        try:
            self._validate_opt_in(self._promotion)
            self._verify_bound_files()
        except Exception:
            return {
                "status": "unavailable",
                "profile": PROMOTION_PROFILE,
                "model_verified": False,
                "report_verified": False,
                "promotion_verified": False,
                "release_enabled": False,
                "scope": PROMOTION_SCOPE,
                "model": self.model_label,
                "network_required": False,
            }
        return {
            "status": "ready",
            "profile": PROMOTION_PROFILE,
            "model_verified": True,
            "report_verified": True,
            "promotion_verified": True,
            "release_enabled": True,
            "scope": PROMOTION_SCOPE,
            "model": self.model_label,
            "network_required": False,
        }
