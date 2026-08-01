"""Fail-closed product orchestration for quality and review prioritization.

The existing quality analyzer remains the source of truth for READY, RETAKE,
and LIMITED.  This module only controls which stage is allowed to run next.
Review-priority output is deliberately non-diagnostic and cannot be released
when its adapter is unavailable, malformed, or raises an exception.
"""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


WORKFLOW_DISCLAIMER = (
    "Workflow support only; review priority is not a diagnosis or treatment "
    "recommendation."
)


class ProductMode(str, Enum):
    """The three independently selectable product presentations."""

    QUALITY_ONLY = "QUALITY_ONLY"
    ESCALATION_ONLY = "ESCALATION_ONLY"
    COMBINED = "COMBINED"


class EscalationDecision(str, Enum):
    """Typed, non-diagnostic review-priority decisions."""

    ROUTINE_REVIEW = "ROUTINE_REVIEW"
    PRIORITY_REVIEW = "PRIORITY_REVIEW"
    UNCERTAIN = "UNCERTAIN"


class EscalationReason(str, Enum):
    COMPLETED = "completed"
    MODEL_ABSTAINED = "model_abstained"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    QUALITY_GATE_BLOCKED = "quality_gate_blocked"
    ADAPTER_ERROR = "adapter_error"
    INVALID_OUTPUT = "invalid_output"


class StageState(str, Enum):
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    ABSTAINED = "ABSTAINED"
    RELEASED = "RELEASED"


class EscalationAssessment(BaseModel):
    """Stable schema for a review-priority stage result.

    ``release_allowed`` is a policy decision, not merely a model prediction.
    The orchestrator forces it to false for UNCERTAIN or unavailable results.
    """

    decision: EscalationDecision
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    executed: bool
    model_available: bool
    release_allowed: bool
    reason: EscalationReason
    summary: str
    instruction: str
    model: str
    disclaimer: str = WORKFLOW_DISCLAIMER


class WorkflowStage(BaseModel):
    stage: str
    state: StageState
    detail: str


class WorkflowResponse(BaseModel):
    product_mode: ProductMode
    display: dict[str, object]
    quality_assessment: dict[str, object] | None
    escalation_assessment: EscalationAssessment | None
    workflow_trace: list[WorkflowStage]
    disclaimer: str = WORKFLOW_DISCLAIMER


class QualityAnalyzer(Protocol):
    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> dict[str, object]: ...


class EscalationAdapter(Protocol):
    model_label: str

    def runtime_status(self) -> dict[str, object]: ...

    async def assess(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        allow_experimental_input: bool = False,
    ) -> EscalationAssessment: ...


def uncertain_escalation(
    *,
    reason: EscalationReason,
    summary: str,
    instruction: str,
    model: str,
    executed: bool = False,
    model_available: bool = False,
) -> EscalationAssessment:
    """Construct the only safe result for an unavailable priority stage."""

    return EscalationAssessment(
        decision=EscalationDecision.UNCERTAIN,
        confidence=None,
        executed=executed,
        model_available=model_available,
        release_allowed=False,
        reason=reason,
        summary=summary,
        instruction=instruction,
        model=model,
    )


def _safe_decisive_assessment(
    result: EscalationAssessment,
) -> EscalationAssessment:
    """Replace adapter-authored prose with policy-authored queue language."""

    if result.decision is EscalationDecision.PRIORITY_REVIEW:
        summary = "This usable image was flagged for earlier clinician review."
        instruction = (
            "Place it in the priority-review queue; a clinician makes the final "
            "interpretation."
        )
    else:
        summary = "No priority flag was released for this usable image."
        instruction = (
            "Keep it in the routine clinician-review queue. This is not a finding "
            "of normality."
        )
    return EscalationAssessment(
        decision=result.decision,
        confidence=result.confidence,
        executed=True,
        model_available=True,
        release_allowed=True,
        reason=EscalationReason.COMPLETED,
        summary=summary,
        instruction=instruction,
        model=result.model,
    )


class UnavailableEscalationAdapter:
    """Explicit placeholder until a validated local artifact is connected.

    It never invents ROUTINE_REVIEW or PRIORITY_REVIEW and therefore cannot
    release a review-priority decision.
    """

    def __init__(
        self,
        *,
        model_label: str = "RetinaPriority unavailable · no validated artifact loaded",
        status_detail: str = "No allowlisted local review-priority model is active.",
    ) -> None:
        self.model_label = model_label
        self.status_detail = status_detail

    def runtime_status(self) -> dict[str, object]:
        return {
            "status": "unavailable",
            "profile": "fail-closed-stub",
            "model_verified": False,
            "release_enabled": False,
            "model": self.model_label,
            "detail": self.status_detail,
            "network_required": False,
        }

    async def assess(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        allow_experimental_input: bool = False,
    ) -> EscalationAssessment:
        del image_bytes, filename, content_type, allow_experimental_input
        return uncertain_escalation(
            reason=EscalationReason.ARTIFACT_UNAVAILABLE,
            summary="Review priority is uncertain because no validated local priority model is loaded.",
            instruction="Use the normal clinician-review queue until the priority model is validated.",
            model=self.model_label,
        )


def _quality_unavailable() -> dict[str, object]:
    return {
        "status": "LIMITED",
        "eyebrow": "Quality gate unavailable",
        "summary": "Capture quality could not be assessed reliably.",
        "confidence": None,
        "issues": ["Assessment uncertain"],
        "instruction": "Have a trained operator review or recapture the image before continuing.",
        "scores": None,
        "disclaimer": "Technical image-quality assessment only; not a diagnosis.",
        "mode": "workflow-fail-closed",
    }


def _escalation_display(result: EscalationAssessment) -> dict[str, object]:
    if result.decision is EscalationDecision.PRIORITY_REVIEW:
        eyebrow = "Priority review"
        issues = ["Earlier clinician review suggested"]
        summary = "This usable image was flagged for earlier clinician review."
        instruction = (
            "Place it in the priority-review queue; a clinician makes the final "
            "interpretation."
        )
    elif result.decision is EscalationDecision.ROUTINE_REVIEW:
        eyebrow = "Routine review"
        issues = []
        summary = "No priority flag was released for this usable image."
        instruction = (
            "Keep it in the routine clinician-review queue. This is not a finding "
            "of normality."
        )
    else:
        eyebrow = "Review priority uncertain"
        issues = ["Human prioritization required"]
        summary = "Review priority could not be assigned reliably."
        instruction = (
            "Route the image to human prioritization; do not delay review based on "
            "this result."
        )
    return {
        "status": result.decision.value,
        "eyebrow": eyebrow,
        "summary": summary,
        "confidence": result.confidence,
        "issues": issues,
        "instruction": instruction,
        "scores": None,
        "disclaimer": result.disclaimer,
        "mode": "review-priority",
    }


class WorkflowOrchestrator:
    """Run one of the three modes while enforcing the quality-first invariant."""

    def __init__(
        self,
        *,
        quality: QualityAnalyzer,
        escalation: EscalationAdapter,
    ) -> None:
        self.quality = quality
        self.escalation = escalation

    async def _quality(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None,
        allow_experimental_input: bool,
    ) -> dict[str, object]:
        try:
            kwargs: dict[str, object] = {
                "filename": filename,
                "content_type": content_type,
                "scenario": scenario,
            }
            if allow_experimental_input:
                kwargs["allow_experimental_input"] = True
            result = await self.quality.analyze(image_bytes, **kwargs)
        except Exception:
            return _quality_unavailable()
        if not isinstance(result, dict) or result.get("status") not in {
            "READY",
            "LIMITED",
            "RETAKE",
            "UNSUPPORTED",
        }:
            return _quality_unavailable()
        return result

    async def _escalation(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        allow_experimental_input: bool,
    ) -> EscalationAssessment:
        try:
            kwargs: dict[str, object] = {
                "filename": filename,
                "content_type": content_type,
            }
            if allow_experimental_input:
                kwargs["allow_experimental_input"] = True
            result = await self.escalation.assess(image_bytes, **kwargs)
        except Exception:
            return uncertain_escalation(
                reason=EscalationReason.ADAPTER_ERROR,
                summary="Review priority is uncertain because the local priority stage failed safely.",
                instruction="Use the normal clinician-review queue.",
                model=getattr(self.escalation, "model_label", "Unknown local adapter"),
            )
        if not isinstance(result, EscalationAssessment):
            return uncertain_escalation(
                reason=EscalationReason.INVALID_OUTPUT,
                summary="Review priority is uncertain because the local priority output was invalid.",
                instruction="Use the normal clinician-review queue.",
                model=getattr(self.escalation, "model_label", "Unknown local adapter"),
            )
        if result.decision is EscalationDecision.UNCERTAIN:
            return uncertain_escalation(
                reason=result.reason,
                summary="Review priority could not be assigned reliably.",
                instruction=(
                    "Route the image to human prioritization; do not delay review "
                    "based on this result."
                ),
                model=result.model,
                executed=result.executed,
                model_available=result.model_available,
            )
        if not (result.executed and result.model_available and result.release_allowed):
            return uncertain_escalation(
                reason=EscalationReason.INVALID_OUTPUT,
                summary="Review priority is uncertain because release checks were not satisfied.",
                instruction="Use the normal clinician-review queue.",
                model=result.model,
                executed=result.executed,
                model_available=result.model_available,
            )
        return _safe_decisive_assessment(result)

    async def run(
        self,
        mode: ProductMode,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> WorkflowResponse:
        if mode is ProductMode.QUALITY_ONLY:
            quality = await self._quality(
                image_bytes,
                filename=filename,
                content_type=content_type,
                scenario=scenario,
                allow_experimental_input=allow_experimental_input,
            )
            return WorkflowResponse(
                product_mode=mode,
                display=quality,
                quality_assessment=quality,
                escalation_assessment=None,
                workflow_trace=[
                    WorkflowStage(
                        stage="Quality gate",
                        state=StageState.COMPLETED,
                        detail=str(quality["status"]),
                    ),
                    WorkflowStage(
                        stage="Review priority",
                        state=StageState.SKIPPED,
                        detail="Quality-only mode",
                    ),
                    WorkflowStage(
                        stage="Safety policy",
                        state=StageState.COMPLETED,
                        detail="Quality result only",
                    ),
                ],
            )

        if mode is ProductMode.ESCALATION_ONLY:
            escalation = await self._escalation(
                image_bytes,
                filename=filename,
                content_type=content_type,
                allow_experimental_input=allow_experimental_input,
            )
            released = escalation.release_allowed
            return WorkflowResponse(
                product_mode=mode,
                display=_escalation_display(escalation),
                quality_assessment=None,
                escalation_assessment=escalation,
                workflow_trace=[
                    WorkflowStage(
                        stage="Quality gate",
                        state=StageState.SKIPPED,
                        detail="Escalation-only mode",
                    ),
                    WorkflowStage(
                        stage="Review priority",
                        state=(
                            StageState.COMPLETED
                            if escalation.executed
                            else StageState.UNAVAILABLE
                        ),
                        detail=escalation.decision.value,
                    ),
                    WorkflowStage(
                        stage="Safety policy",
                        state=StageState.RELEASED if released else StageState.ABSTAINED,
                        detail=(
                            "Review route released"
                            if released
                            else "No review route released"
                        ),
                    ),
                ],
            )

        quality = await self._quality(
            image_bytes,
            filename=filename,
            content_type=content_type,
            scenario=scenario,
            allow_experimental_input=allow_experimental_input,
        )
        if quality["status"] != "READY":
            escalation = uncertain_escalation(
                reason=EscalationReason.QUALITY_GATE_BLOCKED,
                summary="Review priority was not assessed because the image did not pass the quality gate.",
                instruction="Follow the quality-gate guidance before prioritization.",
                model=getattr(self.escalation, "model_label", "Local priority adapter"),
            )
            return WorkflowResponse(
                product_mode=mode,
                display=quality,
                quality_assessment=quality,
                escalation_assessment=escalation,
                workflow_trace=[
                    WorkflowStage(
                        stage="Quality gate",
                        state=StageState.COMPLETED,
                        detail=str(quality["status"]),
                    ),
                    WorkflowStage(
                        stage="Review priority",
                        state=StageState.BLOCKED,
                        detail="Only READY images may continue",
                    ),
                    WorkflowStage(
                        stage="Safety policy",
                        state=StageState.ABSTAINED,
                        detail="No review priority released",
                    ),
                ],
            )

        escalation = await self._escalation(
            image_bytes,
            filename=filename,
            content_type=content_type,
            allow_experimental_input=allow_experimental_input,
        )
        released = escalation.release_allowed
        return WorkflowResponse(
            product_mode=mode,
            display=_escalation_display(escalation),
            quality_assessment=quality,
            escalation_assessment=escalation,
            workflow_trace=[
                WorkflowStage(
                    stage="Quality gate",
                    state=StageState.COMPLETED,
                    detail="READY",
                ),
                WorkflowStage(
                    stage="Review priority",
                    state=(
                        StageState.COMPLETED
                        if escalation.executed
                        else StageState.UNAVAILABLE
                    ),
                    detail=escalation.decision.value,
                ),
                WorkflowStage(
                    stage="Safety policy",
                    state=StageState.RELEASED if released else StageState.ABSTAINED,
                    detail=(
                        "Review route released"
                        if released
                        else "No review route released"
                    ),
                ),
            ],
        )


def build_escalation_adapter() -> EscalationAdapter:
    """Build the selected exact-identity local research-demo adapter.

    ``specialist`` remains the default. ``gemma`` is an optional, explicitly
    uncalibrated llama.cpp free-generation path. Both require the same
    nonclinical research-demo opt-in; every construction failure falls back to
    the existing UNCERTAIN-only adapter.
    """

    opt_in_name = "RETINA_ENABLE_ESCALATION_RESEARCH_DEMO"
    if os.getenv(opt_in_name) != "1":
        return UnavailableEscalationAdapter(
            model_label="RetinaPriority research demo disabled",
            status_detail=f"Set {opt_in_name}=1 to opt into the local nonclinical demo.",
        )
    engine = os.getenv("RETINA_ESCALATION_ENGINE", "specialist").strip().lower()
    if engine == "gemma":
        try:
            from gemma_escalation import LocalGemmaEscalationAdapter

            lora_path_value = os.getenv("RETINA_ESCALATION_GEMMA_LORA_PATH", "")
            lora_sha256 = os.getenv("RETINA_ESCALATION_GEMMA_LORA_SHA256", "")
            if not lora_path_value or not lora_sha256:
                raise ValueError("Gemma escalation LoRA path and SHA-256 are required")
            return LocalGemmaEscalationAdapter(
                api_url=os.getenv(
                    "RETINA_ESCALATION_GEMMA_API_URL",
                    "http://127.0.0.1:8082",
                ),
                model_id=os.getenv(
                    "RETINA_ESCALATION_GEMMA_MODEL_ID",
                    "retinapriority-gemma4-26b",
                ),
                lora_path=Path(lora_path_value),
                lora_sha256=lora_sha256,
                timeout_seconds=float(
                    os.getenv("RETINA_ESCALATION_GEMMA_TIMEOUT_SECONDS", "90")
                ),
            )
        except Exception:
            return UnavailableEscalationAdapter(
                model_label=(
                    "RetinaPriority Gemma LoRA unavailable · uncalibrated experimental"
                ),
                status_detail=(
                    "Loopback URL, exact LoRA identity, or Gemma escalation "
                    "configuration verification failed; output is forced to UNCERTAIN."
                ),
            )
    if engine != "specialist":
        return UnavailableEscalationAdapter(
            model_label="RetinaPriority research demo unavailable",
            status_detail=(
                "RETINA_ESCALATION_ENGINE must be specialist or gemma; output is "
                "forced to UNCERTAIN."
            ),
        )
    try:
        from escalation_specialist import LocalEscalationSpecialistAdapter

        project_root = Path(__file__).resolve().parents[1]
        return LocalEscalationSpecialistAdapter(
            project_root=project_root,
            promotion_manifest_path=(
                project_root
                / "models/retinaready-escalation-demo/promotion-manifest.json"
            ),
            device=os.getenv("RETINA_ESCALATION_DEVICE", "cpu"),
        )
    except Exception:
        return UnavailableEscalationAdapter(
            model_label="RetinaPriority research demo unavailable",
            status_detail="Promotion or artifact verification failed; output is forced to UNCERTAIN.",
        )
