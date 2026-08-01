import asyncio
import unittest

from workflow import (
    EscalationAssessment,
    EscalationDecision,
    EscalationReason,
    ProductMode,
    StageState,
    UnavailableEscalationAdapter,
    WorkflowOrchestrator,
)


def quality_result(status: str) -> dict[str, object]:
    return {
        "status": status,
        "eyebrow": status.title(),
        "summary": f"Quality returned {status}.",
        "confidence": None,
        "issues": [],
        "instruction": "Follow quality guidance.",
        "scores": None,
        "disclaimer": "Technical image-quality assessment only; not a diagnosis.",
        "mode": "test",
    }


class FakeQuality:
    def __init__(self, status: str = "READY", *, fail: bool = False) -> None:
        self.status = status
        self.fail = fail
        self.calls = 0

    async def analyze(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("quality unavailable")
        return quality_result(self.status)


class FakeEscalation:
    model_label = "RetinaPriority test adapter"

    def __init__(
        self,
        decision: EscalationDecision = EscalationDecision.PRIORITY_REVIEW,
        *,
        release_allowed: bool = True,
        fail: bool = False,
        malformed: bool = False,
    ) -> None:
        self.decision = decision
        self.release_allowed = release_allowed
        self.fail = fail
        self.malformed = malformed
        self.calls = 0

    def runtime_status(self) -> dict[str, object]:
        return {"status": "ready", "release_enabled": True}

    async def assess(self, *args: object, **kwargs: object) -> EscalationAssessment:
        self.calls += 1
        if self.fail:
            raise RuntimeError("priority unavailable")
        if self.malformed:
            return {"decision": "PRIORITY_REVIEW"}  # type: ignore[return-value]
        return EscalationAssessment(
            decision=self.decision,
            confidence=(0.93 if self.decision is not EscalationDecision.UNCERTAIN else None),
            executed=True,
            model_available=True,
            release_allowed=self.release_allowed,
            reason=EscalationReason.COMPLETED,
            summary="Local review-priority suggestion complete.",
            instruction="Keep the clinician in the review loop.",
            model=self.model_label,
        )


def run_workflow(
    mode: ProductMode,
    *,
    quality: FakeQuality,
    escalation: object,
):
    return asyncio.run(
        WorkflowOrchestrator(quality=quality, escalation=escalation).run(
            mode,
            b"image",
            filename="fundus.jpg",
            content_type="image/jpeg",
        )
    )


class WorkflowTests(unittest.TestCase):
    def test_decision_schema_has_only_three_non_diagnostic_states(self) -> None:
        self.assertEqual(
            {decision.value for decision in EscalationDecision},
            {"ROUTINE_REVIEW", "PRIORITY_REVIEW", "UNCERTAIN"},
        )

    def test_quality_only_runs_quality_and_skips_escalation(self) -> None:
        quality = FakeQuality("READY")
        escalation = FakeEscalation()
        result = run_workflow(
            ProductMode.QUALITY_ONLY,
            quality=quality,
            escalation=escalation,
        )
        self.assertEqual(quality.calls, 1)
        self.assertEqual(escalation.calls, 0)
        self.assertEqual(result.display["status"], "READY")
        self.assertIsNone(result.escalation_assessment)
        self.assertEqual(result.workflow_trace[1].state, StageState.SKIPPED)

    def test_escalation_only_skips_quality_and_can_release_routine_review(self) -> None:
        quality = FakeQuality("RETAKE")
        escalation = FakeEscalation(EscalationDecision.ROUTINE_REVIEW)
        result = run_workflow(
            ProductMode.ESCALATION_ONLY,
            quality=quality,
            escalation=escalation,
        )
        self.assertEqual(quality.calls, 0)
        self.assertEqual(escalation.calls, 1)
        self.assertEqual(result.display["status"], "ROUTINE_REVIEW")
        self.assertTrue(result.escalation_assessment.release_allowed)
        self.assertEqual(result.workflow_trace[2].state, StageState.RELEASED)

    def test_combined_ready_runs_priority_and_can_release_priority_review(self) -> None:
        quality = FakeQuality("READY")
        escalation = FakeEscalation(EscalationDecision.PRIORITY_REVIEW)
        result = run_workflow(
            ProductMode.COMBINED,
            quality=quality,
            escalation=escalation,
        )
        self.assertEqual(quality.calls, 1)
        self.assertEqual(escalation.calls, 1)
        self.assertEqual(result.display["status"], "PRIORITY_REVIEW")
        self.assertEqual(result.quality_assessment["status"], "READY")
        self.assertEqual(result.workflow_trace[2].state, StageState.RELEASED)

    def test_routine_review_uses_non_diagnostic_policy_authored_wording(self) -> None:
        quality = FakeQuality("READY")
        escalation = FakeEscalation(EscalationDecision.ROUTINE_REVIEW)
        result = run_workflow(
            ProductMode.COMBINED,
            quality=quality,
            escalation=escalation,
        )
        combined_text = " ".join(
            [
                str(result.display["summary"]),
                str(result.display["instruction"]),
                result.escalation_assessment.summary,
                result.escalation_assessment.instruction,
            ]
        ).lower()
        self.assertIn("no priority flag", combined_text)
        self.assertIn("not a finding of normality", combined_text)
        self.assertNotIn("healthy", combined_text)
        self.assertNotIn("appears fine", combined_text)

    def test_combined_blocks_priority_for_every_non_ready_quality_state(self) -> None:
        for status in ("LIMITED", "RETAKE", "UNSUPPORTED"):
            with self.subTest(status=status):
                quality = FakeQuality(status)
                escalation = FakeEscalation()
                result = run_workflow(
                    ProductMode.COMBINED,
                    quality=quality,
                    escalation=escalation,
                )
                self.assertEqual(escalation.calls, 0)
                self.assertEqual(result.display["status"], status)
                self.assertEqual(
                    result.escalation_assessment.reason,
                    EscalationReason.QUALITY_GATE_BLOCKED,
                )
                self.assertFalse(result.escalation_assessment.release_allowed)
                self.assertEqual(result.workflow_trace[1].state, StageState.BLOCKED)

    def test_unavailable_adapter_never_releases_a_priority(self) -> None:
        quality = FakeQuality("READY")
        result = run_workflow(
            ProductMode.COMBINED,
            quality=quality,
            escalation=UnavailableEscalationAdapter(),
        )
        self.assertEqual(
            result.escalation_assessment.decision,
            EscalationDecision.UNCERTAIN,
        )
        self.assertFalse(result.escalation_assessment.release_allowed)
        self.assertEqual(result.display["status"], "UNCERTAIN")
        self.assertEqual(result.workflow_trace[2].state, StageState.ABSTAINED)

    def test_adapter_exception_fails_closed_to_uncertain(self) -> None:
        result = run_workflow(
            ProductMode.ESCALATION_ONLY,
            quality=FakeQuality(),
            escalation=FakeEscalation(fail=True),
        )
        self.assertEqual(result.display["status"], "UNCERTAIN")
        self.assertEqual(
            result.escalation_assessment.reason,
            EscalationReason.ADAPTER_ERROR,
        )
        self.assertFalse(result.escalation_assessment.release_allowed)

    def test_malformed_adapter_output_fails_closed_to_uncertain(self) -> None:
        result = run_workflow(
            ProductMode.ESCALATION_ONLY,
            quality=FakeQuality(),
            escalation=FakeEscalation(malformed=True),
        )
        self.assertEqual(
            result.escalation_assessment.reason,
            EscalationReason.INVALID_OUTPUT,
        )
        self.assertFalse(result.escalation_assessment.release_allowed)

    def test_non_releaseable_decisive_output_is_downgraded_to_uncertain(self) -> None:
        result = run_workflow(
            ProductMode.ESCALATION_ONLY,
            quality=FakeQuality(),
            escalation=FakeEscalation(release_allowed=False),
        )
        self.assertEqual(
            result.escalation_assessment.decision,
            EscalationDecision.UNCERTAIN,
        )
        self.assertEqual(
            result.escalation_assessment.reason,
            EscalationReason.INVALID_OUTPUT,
        )

    def test_quality_failure_in_combined_mode_blocks_priority(self) -> None:
        quality = FakeQuality(fail=True)
        escalation = FakeEscalation()
        result = run_workflow(
            ProductMode.COMBINED,
            quality=quality,
            escalation=escalation,
        )
        self.assertEqual(result.quality_assessment["status"], "LIMITED")
        self.assertEqual(escalation.calls, 0)
        self.assertEqual(result.workflow_trace[1].state, StageState.BLOCKED)


if __name__ == "__main__":
    unittest.main()
