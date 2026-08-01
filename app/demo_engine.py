"""Deterministic placeholder inference for the RetinaReady UI.

This module deliberately performs no diagnosis. It provides stable, clearly
labelled presentation responses when the large local Gemma runtime is not
loaded. The real local worker uses the same API boundary.
"""

from __future__ import annotations

from typing import Final


SUPPORTED_STATUSES: Final = ("READY", "LIMITED", "RETAKE", "UNSUPPORTED")

_PROFILES: Final[dict[str, dict[str, object]]] = {
    "READY": {
        "eyebrow": "Capture ready",
        "summary": "This image is technically ready for clinical review.",
        "confidence": 0.94,
        "issues": [],
        "instruction": "No retake needed. Continue to the normal review workflow.",
        "scores": {"Clarity": 94, "Illumination": 91, "Field": 95},
    },
    "LIMITED": {
        "eyebrow": "Usable with limitations",
        "summary": "The main retinal field is visible, but capture quality is uneven.",
        "confidence": 0.82,
        "issues": ["Uneven illumination", "Minor field cutoff"],
        "instruction": "If practical, recenter the eye and use more even illumination.",
        "scores": {"Clarity": 83, "Illumination": 61, "Field": 72},
    },
    "RETAKE": {
        "eyebrow": "Retake recommended",
        "summary": "Technical quality is too low for a dependable review.",
        "confidence": 0.91,
        "issues": ["Motion blur", "Field not centered", "Low contrast"],
        "instruction": "Stabilize the camera, refocus, and center the retinal field before retaking.",
        "scores": {"Clarity": 31, "Illumination": 54, "Field": 38},
    },
    "UNSUPPORTED": {
        "eyebrow": "Unable to assess",
        "summary": "This does not appear to be a supported color fundus photograph.",
        "confidence": 0.97,
        "issues": ["Unsupported image type"],
        "instruction": "Upload a color fundus photograph. OCT and angiography images are not supported.",
        "scores": None,
    },
}

def _choose_status(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    scenario: str | None,
) -> str:
    requested = (scenario or "").strip().upper()
    if requested in SUPPORTED_STATUSES:
        return requested

    if not content_type.lower().startswith("image/"):
        return "UNSUPPORTED"

    # Demo mode is only allowed to show predetermined results for explicit,
    # labelled sample scenarios. An arbitrary upload has not been assessed and
    # therefore fails closed instead of fabricating a READY/RETAKE decision.
    return "LIMITED"


def analyze_demo(
    image_bytes: bytes,
    *,
    filename: str,
    content_type: str,
    scenario: str | None = None,
) -> dict[str, object]:
    """Return a stable technical-quality result for the supplied image."""

    status = _choose_status(image_bytes, filename, content_type, scenario)
    requested = (scenario or "").strip().upper()
    if status == "LIMITED" and requested not in SUPPORTED_STATUSES:
        profile: dict[str, object] = {
            "eyebrow": "Local model not loaded",
            "summary": "This upload was not assessed by the deterministic demo.",
            "confidence": None,
            "issues": ["Assessment unavailable"],
            "instruction": "Start the local Gemma model or choose a labelled demo sample.",
            "scores": None,
        }
    else:
        profile = dict(_PROFILES[status])
    profile.update(
        {
            "status": status,
            "mode": "demo",
            "disclaimer": "Technical image-quality assessment only. No diagnosis is performed.",
        }
    )
    return profile
