"""Offline runtime for the compact retinal image-quality specialist."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any


QUALITY_ATTENTION_LABEL = (
    "Model quality attention \u2014 not pathology localization."
)
QUALITY_FACTOR_INDEX = {
    "artifact": 0,
    "clarity": 1,
    "field_definition": 2,
}
QUALITY_FACTOR_LABEL = {
    "artifact": "Artifact quality",
    "clarity": "Clarity",
    "field_definition": "Field definition",
}


@dataclass(frozen=True)
class SpecialistAssessment:
    decision: str
    ready_score: float
    scores: dict[str, int]
    issue_codes: list[str]
    ready_threshold: float
    retake_threshold: float
    quality_attention: dict[str, str] | None = None

    def prompt_context(self) -> str:
        return (
            "A separate local, frozen retinal-quality specialist produced "
            "the following technical acquisition evidence. Treat its decision as "
            "the fixed quality gate, not as a diagnosis. If the image is unsupported "
            "or your visual assessment conflicts with the gate, return LIMITED. "
            f"Gate decision: {self.decision}. "
            f"READY score: {self.ready_score:.4f}; READY threshold: "
            f">{self.ready_threshold:.4f}; RETAKE threshold: "
            f"<{self.retake_threshold:.4f}. "
            f"Artifact quality: {self.scores['artifact']} / 100; "
            f"clarity: {self.scores['clarity']} / 100; field definition: "
            f"{self.scores['field_definition']} / 100. "
            "Use only the allowed capture-issue codes. Never discuss pathology."
        )


class QualitySpecialist:
    """Frozen DenseNet features plus tiny decision and quality-factor heads."""

    def __init__(
        self,
        *,
        backbone_path: Path,
        decision_head_path: Path,
        factor_head_path: Path,
        device: str = "cpu",
    ) -> None:
        import torch
        from torchvision import models

        for path in (backbone_path, decision_head_path, factor_head_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.bundle_verified = self._verify_bundle_manifest(
            backbone_path=backbone_path,
            decision_head_path=decision_head_path,
            factor_head_path=factor_head_path,
        )

        self._torch = torch
        self._device = torch.device(device)
        decision_artifact = torch.load(
            decision_head_path, map_location="cpu", weights_only=True
        )
        factor_artifact = torch.load(
            factor_head_path, map_location="cpu", weights_only=True
        )
        self._validate_artifacts(decision_artifact, factor_artifact)

        backbone = models.densenet121(weights=None)
        backbone_state = torch.load(
            backbone_path, map_location="cpu", weights_only=True
        )
        # Torchvision's official DenseNet checkpoint retains legacy keys such
        # as ``norm.1.weight``. Apply the same migration as its enum loader so
        # the bundled checkpoint remains usable without a network request.
        legacy_pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\."
            r"((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(backbone_state):
            match = legacy_pattern.match(key)
            if match:
                backbone_state[match.group(1) + match.group(2)] = backbone_state.pop(key)
        backbone.load_state_dict(backbone_state)
        backbone.classifier = torch.nn.Identity()
        self._backbone = backbone.eval().to(self._device)

        self._decision_heads = self._load_heads(decision_artifact)
        self._factor_heads = self._load_heads(factor_artifact)
        self._decision_mean = decision_artifact["feature_mean"].to(self._device)
        self._decision_std = decision_artifact["feature_std"].to(self._device)
        self._factor_mean = factor_artifact["feature_mean"].to(self._device)
        self._factor_std = factor_artifact["feature_std"].to(self._device)
        self._policy = decision_artifact["policy"]

        # The specialist is inference-only. Freezing every parameter keeps the
        # optional attention pass from allocating parameter-gradient buffers.
        for module in [
            self._backbone,
            *self._decision_heads,
            *self._factor_heads,
        ]:
            module.requires_grad_(False)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_bundle_manifest(
        cls,
        *,
        backbone_path: Path,
        decision_head_path: Path,
        factor_head_path: Path,
    ) -> bool:
        manifest_path = decision_head_path.parent / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                raise ValueError("unsupported specialist bundle manifest schema")
            entries = (
                ("backbone", backbone_path),
                ("decision_head", decision_head_path),
                ("factor_head", factor_head_path),
            )
            for name, path in entries:
                entry = manifest[name]
                if entry["file"] != path.name or entry["sha256"] != cls._sha256(path):
                    raise ValueError(f"specialist bundle checksum mismatch: {path.name}")
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid specialist bundle manifest") from error
        return True

    @staticmethod
    def _validate_artifacts(
        decision_artifact: dict[str, Any], factor_artifact: dict[str, Any]
    ) -> None:
        required = {
            "schema_version",
            "hidden_dim",
            "input_dim",
            "members",
            "feature_mean",
            "feature_std",
            "policy",
        }
        for name, artifact in (
            ("decision", decision_artifact),
            ("factor", factor_artifact),
        ):
            missing = required - set(artifact)
            if missing:
                raise ValueError(f"{name} specialist artifact missing {sorted(missing)}")
            if artifact["schema_version"] != 1:
                raise ValueError(f"unsupported {name} artifact schema")
            if not artifact["members"]:
                raise ValueError(f"{name} artifact contains no ensemble members")
        if decision_artifact["input_dim"] != factor_artifact["input_dim"]:
            raise ValueError("decision and factor artifacts use different feature widths")

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
                    torch.nn.Linear(hidden_dim, 4),
                )
            return torch.nn.Linear(input_dim, 4)

        heads = []
        for state in artifact["members"]:
            # Training wraps this sequence in ``QualityHead.network``.
            stripped = {
                key.removeprefix("network."): value for key, value in state.items()
            }
            head = make_head()
            head.load_state_dict(stripped)
            heads.append(head.eval().to(self._device))
        return heads

    @staticmethod
    def _crop_black_border(image: Any, np: Any, threshold: int = 15) -> Any:
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

    @staticmethod
    def _render_quality_attention(
        tensor: Any,
        attention_map: Any,
        *,
        factor_key: str,
        method: str,
    ) -> dict[str, str]:
        """Blend a technical-quality saliency map with the preprocessed image."""

        import numpy as np
        from PIL import Image

        source = (
            tensor.detach().float().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        )
        source = np.clip(source * 255.0, 0, 255)
        saliency = np.clip(attention_map.detach().float().cpu().numpy(), 0, 1)

        # A yellow-to-coral wash remains legible on both dark fundus fields and
        # pale optic-disc regions. Alpha is zero outside attended regions.
        heat = np.empty_like(source)
        heat[..., 0] = 255
        heat[..., 1] = 210 - 130 * saliency
        heat[..., 2] = 60
        alpha = (0.58 * np.power(saliency, 0.8))[..., None]
        blended = np.clip(source * (1 - alpha) + heat * alpha, 0, 255).astype(
            np.uint8
        )

        output = BytesIO()
        Image.fromarray(blended).save(output, format="PNG", optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return {
            "label": QUALITY_ATTENTION_LABEL,
            "factor": factor_key,
            "factor_label": QUALITY_FACTOR_LABEL[factor_key],
            "method": method,
            "image_data_url": f"data:image/png;base64,{encoded}",
        }

    def _quality_attention(
        self, tensor: Any, factor_key: str
    ) -> dict[str, str] | None:
        """Create factor-specific Grad-CAM without participating in the decision."""

        torch = self._torch
        factor_index = QUALITY_FACTOR_INDEX[factor_key]
        with torch.enable_grad():
            batch = tensor.unsqueeze(0).to(self._device).requires_grad_(True)
            feature_map = self._backbone.features(batch)
            activated = torch.nn.functional.relu(feature_map, inplace=False)
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                activated, (1, 1)
            ).flatten(1).float()
            factor_features = (pooled - self._factor_mean) / self._factor_std
            quality_score = torch.stack(
                [
                    torch.sigmoid(head(factor_features)[0, factor_index + 1])
                    for head in self._factor_heads
                ]
            ).mean()

            # Target the inverse quality score so the map emphasizes evidence
            # that lowers this acquisition-quality factor.
            gradients = torch.autograd.grad(-quality_score, activated)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            attention = torch.nn.functional.relu(
                (weights * activated).sum(dim=1, keepdim=True)
            )
            method = "factor-grad-cam"

            minimum = attention.amin()
            maximum = attention.amax()
            spread = maximum - minimum
            if not bool(torch.isfinite(attention).all()):
                return None
            if float(spread.detach().cpu()) <= 1e-8:
                # Saturated sigmoid heads can yield an all-zero signed CAM.
                # Gradient x activation still provides factor-specific
                # sensitivity, while retaining the same non-localization caveat.
                attention = (gradients * activated).abs().mean(
                    dim=1, keepdim=True
                )
                method = "factor-gradient-sensitivity"
                minimum = attention.amin()
                maximum = attention.amax()
                spread = maximum - minimum
                if (
                    not bool(torch.isfinite(attention).all())
                    or float(spread.detach().cpu()) <= 1e-8
                ):
                    return None

            attention = ((attention - minimum) / spread).detach()
            attention = torch.nn.functional.interpolate(
                attention,
                size=tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        return self._render_quality_attention(
            tensor,
            attention,
            factor_key=factor_key,
            method=method,
        )

    def _safe_quality_attention(
        self, tensor: Any, factor_key: str
    ) -> dict[str, str] | None:
        """Keep explanation failures non-fatal and outside the quality gate."""

        try:
            return self._quality_attention(tensor, factor_key)
        except Exception:
            # This is presentation-only metadata. It must never change an
            # otherwise valid quality assessment or trigger a false retake.
            return None

    def assess(self, image_bytes: bytes) -> SpecialistAssessment:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = self._preprocess(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("specialist could not decode the image") from exc

        torch = self._torch
        with torch.inference_mode():
            features = self._backbone(tensor.unsqueeze(0).to(self._device)).float()
            decision_features = (features - self._decision_mean) / self._decision_std
            factor_features = (features - self._factor_mean) / self._factor_std
            ready_score = torch.stack(
                [torch.sigmoid(head(decision_features)[0, 0]) for head in self._decision_heads]
            ).mean()
            factors = torch.stack(
                [torch.sigmoid(head(factor_features)[0, 1:]) for head in self._factor_heads]
            ).mean(dim=0)

        ready_score_value = float(ready_score.cpu())
        ready_threshold = float(
            self._policy["ready_threshold_strictly_greater_than"]
        )
        retake_threshold = float(
            self._policy["retake_threshold_strictly_less_than"]
        )
        if ready_score_value > ready_threshold:
            decision = "READY"
        elif ready_score_value < retake_threshold:
            decision = "RETAKE"
        else:
            decision = "LIMITED"

        factor_values = [float(value) for value in factors.cpu()]
        if any(not math.isfinite(value) for value in factor_values):
            raise ValueError("specialist produced a non-finite factor score")
        scores = {
            "artifact": round(100 * factor_values[0]),
            "clarity": round(100 * factor_values[1]),
            "field_definition": round(100 * factor_values[2]),
        }
        issue_codes: list[str] = []
        if decision == "LIMITED":
            issue_codes.append("uncertain")
        elif decision == "RETAKE":
            if scores["artifact"] < 65:
                issue_codes.append("artifact")
            if scores["clarity"] < 65:
                issue_codes.append("blur")
            if scores["field_definition"] < 65:
                issue_codes.append("field_cutoff")

        # A spatial explanation is useful only when the gate has found a
        # concrete recapture action. Showing a "weakest" region on READY would
        # imply a defect, while showing one on LIMITED would over-explain an
        # assessment the system itself considers uncertain. RETAKE-only also
        # keeps the fast abstention path fast.
        quality_attention = None
        if decision == "RETAKE":
            weakest_factor = min(
                QUALITY_FACTOR_INDEX,
                key=lambda key: factor_values[QUALITY_FACTOR_INDEX[key]],
            )
            quality_attention = self._safe_quality_attention(tensor, weakest_factor)

        return SpecialistAssessment(
            decision=decision,
            ready_score=ready_score_value,
            scores=scores,
            issue_codes=issue_codes,
            ready_threshold=ready_threshold,
            retake_threshold=retake_threshold,
            quality_attention=quality_attention,
        )
