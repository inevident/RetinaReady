import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from workflow import (
    EscalationAssessment,
    EscalationDecision,
    EscalationReason,
)


class RecordingQuality:
    mode = "specialist-local"
    model_label = "Recording quality specialist"

    def __init__(self, status: str = "READY") -> None:
        self.status = status
        self.calls: list[dict[str, object]] = []

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
        allow_experimental_input: bool = False,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "image_bytes": image_bytes,
                "filename": filename,
                "content_type": content_type,
                "scenario": scenario,
                "allow_experimental_input": allow_experimental_input,
            }
        )
        return {
            "status": self.status,
            "eyebrow": "Capture ready",
            "summary": "Quality passed.",
            "confidence": None,
            "issues": [],
            "instruction": "Continue to clinician review.",
            "scores": None,
            "disclaimer": "Technical quality only; not a diagnosis.",
            "mode": "test",
        }


class RecordingEscalation:
    model_label = "Recording review-priority adapter"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def assess(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        allow_experimental_input: bool = False,
    ) -> EscalationAssessment:
        self.calls.append(
            {
                "image_bytes": image_bytes,
                "filename": filename,
                "content_type": content_type,
                "allow_experimental_input": allow_experimental_input,
            }
        )
        return EscalationAssessment(
            decision=EscalationDecision.PRIORITY_REVIEW,
            confidence=None,
            executed=True,
            model_available=True,
            release_allowed=True,
            reason=EscalationReason.COMPLETED,
            summary="Priority stage completed.",
            instruction="A clinician makes the final decision.",
            model=self.model_label,
        )


class VideoCandidateWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app)

    def test_health_reports_request_gate_opt_in(self) -> None:
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "0"},
            clear=False,
        ):
            disabled = self.client.get("/api/health")
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ):
            enabled = self.client.get("/api/health")

        self.assertEqual(disabled.status_code, 200)
        self.assertFalse(disabled.json()["video_candidate_workflow_enabled"])
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.json()["video_candidate_workflow_enabled"])

    def test_exact_origin_is_rejected_when_opt_in_is_disabled(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "0"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"candidate-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "COMBINED",
                    "x-input-origin": "video-candidate",
                },
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(quality.calls, [])
        self.assertEqual(escalation.calls, [])

    def test_invalid_origin_is_rejected_even_when_opted_in(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"candidate-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "COMBINED",
                    "x-input-origin": "Video-Candidate",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(quality.calls, [])
        self.assertEqual(escalation.calls, [])

    def test_opted_in_exact_origin_reaches_both_stages_with_request_flag(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"candidate-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "COMBINED",
                    "x-input-origin": "video-candidate",
                    "x-filename": "camera-candidate.jpg",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["display"]["status"], "PRIORITY_REVIEW")
        self.assertEqual(len(quality.calls), 1)
        self.assertEqual(len(escalation.calls), 1)
        self.assertIs(quality.calls[0]["allow_experimental_input"], True)
        self.assertIs(escalation.calls[0]["allow_experimental_input"], True)
        self.assertEqual(quality.calls[0]["content_type"], "image/jpeg")
        self.assertEqual(escalation.calls[0]["content_type"], "image/jpeg")

    def test_video_candidate_cannot_bypass_quality_with_escalation_only_mode(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"candidate-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "ESCALATION_ONLY",
                    "x-input-origin": "video-candidate",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("quality gate", response.json()["detail"])
        self.assertEqual(quality.calls, [])
        self.assertEqual(escalation.calls, [])

    def test_quality_only_video_candidate_runs_quality_without_priority(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"candidate-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "QUALITY_ONLY",
                    "x-input-origin": "video-candidate",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["quality_assessment"]["status"], "READY")
        self.assertIsNone(response.json()["escalation_assessment"])
        self.assertEqual(len(quality.calls), 1)
        self.assertIs(quality.calls[0]["allow_experimental_input"], True)
        self.assertEqual(escalation.calls, [])

    def test_non_ready_video_candidate_never_reaches_priority(self) -> None:
        quality = RecordingQuality(status="RETAKE")
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"candidate-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "COMBINED",
                    "x-input-origin": "video-candidate",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["quality_assessment"]["status"], "RETAKE")
        self.assertEqual(response.json()["workflow_trace"][1]["state"], "BLOCKED")
        self.assertEqual(len(quality.calls), 1)
        self.assertEqual(escalation.calls, [])

    def test_video_candidate_cannot_force_a_demo_quality_scenario(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"candidate-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "COMBINED",
                    "x-input-origin": "video-candidate",
                    "x-demo-scenario": "READY",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(quality.calls, [])
        self.assertEqual(escalation.calls, [])

    def test_opt_in_does_not_relax_normal_upload_without_origin_header(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"ordinary-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-product-mode": "COMBINED",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIs(quality.calls[0]["allow_experimental_input"], False)
        self.assertIs(escalation.calls[0]["allow_experimental_input"], False)

    def test_quality_only_api_never_honors_the_video_candidate_header(self) -> None:
        quality = RecordingQuality()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality):
            response = self.client.post(
                "/api/analyze",
                content=b"ordinary-jpeg",
                headers={
                    "content-type": "image/jpeg",
                    "x-input-origin": "video-candidate",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(quality.calls), 1)
        self.assertIs(quality.calls[0]["allow_experimental_input"], False)

    def test_video_content_type_remains_rejected_for_candidate_origin(self) -> None:
        quality = RecordingQuality()
        escalation = RecordingEscalation()
        with patch.dict(
            os.environ,
            {main.VIDEO_CANDIDATE_WORKFLOW_OPT_IN: "1"},
            clear=False,
        ), patch.object(main, "analysis_engine", quality), patch.object(
            main, "escalation_engine", escalation
        ):
            response = self.client.post(
                "/api/workflow",
                content=b"raw-video",
                headers={
                    "content-type": "video/mp4",
                    "x-product-mode": "COMBINED",
                    "x-input-origin": "video-candidate",
                },
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(quality.calls, [])
        self.assertEqual(escalation.calls, [])


if __name__ == "__main__":
    unittest.main()
