import hashlib
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class RecordingAnalyzer:
    mode = "local-model"
    model_label = "recording-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def analyze(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        scenario: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "image_bytes": image_bytes,
                "filename": filename,
                "content_type": content_type,
                "scenario": scenario,
            }
        )
        return {
            "status": "LIMITED",
            "eyebrow": "Unable to assess",
            "summary": "Test result",
            "confidence": None,
            "issues": ["Assessment uncertain"],
            "instruction": "Review the image.",
            "scores": None,
            "disclaimer": "Technical image-quality assessment only; not a diagnosis.",
            "mode": "local-model",
        }


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app)

    def test_health_reports_local_demo(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["privacy"], "local-only")
        self.assertFalse(payload["network_required"])
        self.assertIn("dataset_samples_available", payload)
        self.assertIn("input_scope", payload)
        self.assertEqual(
            payload["product_modes"],
            ["QUALITY_ONLY", "ESCALATION_ONLY", "COMBINED"],
        )
        self.assertFalse(payload["escalation"]["release_enabled"])

    def test_frontend_labels_attention_as_non_pathology(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "Model quality attention \u2014 not pathology localization.",
            response.text,
        )

    def test_frontend_labels_guided_capture_and_video_as_prototypes(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Live capture prototype", response.text)
        self.assertIn("Conventional fundus-camera workflow", response.text)
        self.assertIn("Simulated acquisition using retrospective", response.text)
        self.assertIn("not a clinical or", response.text)
        self.assertIn('id="guidanceCanvas"', response.text)
        self.assertIn('id="guidanceVideo"', response.text)
        self.assertIn('id="loadVideoButton"', response.text)
        self.assertIn('id="videoInput"', response.text)
        self.assertIn("Open camera recording", response.text)
        guidance = self.client.get("/assets/capture-guidance.js")
        self.assertEqual(guidance.status_code, 200)
        self.assertIn("createController", guidance.text)
        self.assertIn("technical-field mask", guidance.text)
        self.assertIn("No color fundus field detected", guidance.text)
        app_javascript = self.client.get("/assets/app.js")
        self.assertEqual(app_javascript.status_code, 200)
        self.assertIn("Retrospective DeepDRiD replay", app_javascript.text)
        self.assertIn("startVideoGuidance", app_javascript.text)
        self.assertIn(
            "never sent to the still-image quality or review-priority models",
            app_javascript.text,
        )

    def test_frontend_does_not_claim_an_unsupported_modality_demo(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('data-scenario="UNSUPPORTED"', response.text)

    def test_frontend_orders_routine_priority_limited_and_retake_samples(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        buttons = (
            ('data-scenario="ROUTINE">Routine', "ROUTINE"),
            ('data-scenario="READY">Priority', "READY"),
            ('data-scenario="LIMITED">Limited', "LIMITED"),
            ('data-scenario="RETAKE">Retake', "RETAKE"),
        )
        positions = []
        for markup, scenario in buttons:
            with self.subTest(scenario=scenario):
                self.assertIn(markup, response.text)
                positions.append(response.text.index(markup))
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('data-scenario="READY">Ready', response.text)

    def test_frontend_states_scope_and_local_decision_trace(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("conventional central-field color fundus", response.text)
        self.assertIn("Zero cloud uploads", response.text)
        self.assertIn("Local decision path", response.text)
        self.assertIn("Gemma 4 verifier", response.text)
        self.assertIn('id="traceGemmaLabel"', response.text)
        self.assertIn("Local models", response.text)
        self.assertIn('id="dropTitle"', response.text)
        self.assertIn("182 held-out usable images", response.text)
        self.assertIn("95.7% priority AUC", response.text)
        self.assertIn("65/65 confident routes correct", response.text)
        self.assertIn("live generation uncalibrated", response.text)
        self.assertIn("not clinical validation", response.text)
        self.assertIn("Weakest predicted factor", response.text)
        self.assertIn("Quality only", response.text)
        self.assertIn("Escalation only", response.text)
        self.assertIn("Combined", response.text)
        self.assertIn("fails closed to UNCERTAIN", response.text)
        app_javascript = self.client.get("/assets/app.js")
        self.assertEqual(app_javascript.status_code, 200)
        self.assertIn(
            "Experimental uncalibrated Gemma priority",
            app_javascript.text,
        )
        self.assertIn("Gemma review-priority LoRA", app_javascript.text)
        self.assertIn(
            "Quality specialist + Gemma LoRA · Local",
            app_javascript.text,
        )

    def test_curated_dataset_sample_endpoint_is_fixed_and_non_arbitrary(self) -> None:
        self.assertEqual(
            tuple(main.DATASET_DEMO_SAMPLES),
            ("ROUTINE", "READY", "LIMITED", "RETAKE"),
        )
        observed_allowlist = frozenset(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in main.DATASET_DEMO_SAMPLES.values()
        )
        self.assertEqual(observed_allowlist, main.SPECIALIST_DEMO_IMAGE_SHA256)

        expected_hashes = {
            "ROUTINE": "21ef6838c18ccfe8697a1e2f4a31d2cce2cb11eb2627995a977d5aaaa9aeeda7",
            # READY remains the backwards-compatible API key for Priority.
            "READY": "b154932d70e281d2b7e2998c52c4c6a4631095f90f48900660c32535b020efd9",
        }
        for scenario, expected_hash in expected_hashes.items():
            with self.subTest(scenario=scenario):
                response = self.client.get(f"/api/demo-samples/{scenario}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "image/jpeg")
                self.assertGreater(len(response.content), 1000)
                self.assertEqual(
                    hashlib.sha256(response.content).hexdigest(), expected_hash
                )
        missing = self.client.get("/api/demo-samples/../../etc/passwd")
        self.assertIn(missing.status_code, {404, 405})

    def test_analyze_accepts_raw_image(self) -> None:
        response = self.client.post(
            "/api/analyze",
            content=b"demo-image-bytes",
            headers={
                "content-type": "image/jpeg",
                "x-filename": "retake-demo.jpg",
                "x-demo-scenario": "RETAKE",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "RETAKE")
        self.assertFalse(payload["meta"]["retained"])

    def test_workflow_quality_only_preserves_quality_result(self) -> None:
        response = self.client.post(
            "/api/workflow",
            content=b"demo-image-bytes",
            headers={
                "content-type": "image/jpeg",
                "x-product-mode": "QUALITY_ONLY",
                "x-demo-scenario": "READY",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["product_mode"], "QUALITY_ONLY")
        self.assertEqual(payload["display"]["status"], "READY")
        self.assertEqual(payload["quality_assessment"]["status"], "READY")
        self.assertIsNone(payload["escalation_assessment"])

    def test_workflow_escalation_only_stub_is_uncertain_and_non_releaseable(self) -> None:
        response = self.client.post(
            "/api/workflow",
            content=b"demo-image-bytes",
            headers={
                "content-type": "image/jpeg",
                "x-product-mode": "ESCALATION_ONLY",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["display"]["status"], "UNCERTAIN")
        self.assertEqual(
            payload["escalation_assessment"]["reason"],
            "artifact_unavailable",
        )
        self.assertFalse(payload["escalation_assessment"]["release_allowed"])
        self.assertIsNone(payload["quality_assessment"])

    def test_workflow_combined_blocks_escalation_after_retake(self) -> None:
        response = self.client.post(
            "/api/workflow",
            content=b"demo-image-bytes",
            headers={
                "content-type": "image/jpeg",
                "x-product-mode": "COMBINED",
                "x-demo-scenario": "RETAKE",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["display"]["status"], "RETAKE")
        self.assertEqual(
            payload["escalation_assessment"]["reason"],
            "quality_gate_blocked",
        )
        self.assertEqual(payload["workflow_trace"][1]["state"], "BLOCKED")

    def test_workflow_combined_ready_still_abstains_without_priority_artifact(self) -> None:
        response = self.client.post(
            "/api/workflow",
            content=b"demo-image-bytes",
            headers={
                "content-type": "image/jpeg",
                "x-product-mode": "COMBINED",
                "x-demo-scenario": "READY",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["quality_assessment"]["status"], "READY")
        self.assertEqual(payload["display"]["status"], "UNCERTAIN")
        self.assertFalse(payload["escalation_assessment"]["release_allowed"])
        self.assertEqual(payload["workflow_trace"][2]["state"], "ABSTAINED")
        self.assertIn(
            main.escalation_engine.model_label,
            payload["display"]["meta"]["model"],
        )

    def test_workflow_rejects_unknown_product_mode(self) -> None:
        response = self.client.post(
            "/api/workflow",
            content=b"demo-image-bytes",
            headers={
                "content-type": "image/jpeg",
                "x-product-mode": "DIAGNOSE",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_raw_jpeg_reaches_analyzer_without_multipart_envelope(self) -> None:
        analyzer = RecordingAnalyzer()
        with patch.object(main, "analysis_engine", analyzer):
            response = self.client.post(
                "/api/analyze",
                content=b"exact-jpeg-bytes",
                headers={
                    "content-type": "image/jpeg; charset=binary",
                    "x-filename": "fundus%20capture.jpg",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            analyzer.calls,
            [
                {
                    "image_bytes": b"exact-jpeg-bytes",
                    "filename": "fundus capture.jpg",
                    "content_type": "image/jpeg",
                    "scenario": None,
                }
            ],
        )

    def test_specialist_only_response_is_labelled_as_local_specialist(self) -> None:
        analyzer = RecordingAnalyzer()
        analyzer.mode = "specialist-local"
        analyzer.model_label = "RetinaReady frozen quality specialist · Local"
        with patch.object(main, "analysis_engine", analyzer):
            response = self.client.post(
                "/api/analyze",
                content=b"image-bytes",
                headers={"content-type": "image/jpeg"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["meta"]["latency_label"], "Local frozen quality specialist"
        )
        self.assertEqual(payload["meta"]["model"], analyzer.model_label)

    def test_supported_raw_image_content_types_reach_analyzer(self) -> None:
        for content_type in (
            "image/jpeg",
            "image/png",
            "image/webp",
        ):
            with self.subTest(content_type=content_type):
                analyzer = RecordingAnalyzer()
                with patch.object(main, "analysis_engine", analyzer):
                    response = self.client.post(
                        "/api/analyze",
                        content=b"image-bytes",
                        headers={"content-type": content_type},
                    )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(analyzer.calls[0]["content_type"], content_type)

    def test_heic_is_rejected_when_no_decoder_is_bundled(self) -> None:
        analyzer = RecordingAnalyzer()
        with patch.object(main, "analysis_engine", analyzer):
            response = self.client.post(
                "/api/analyze",
                content=b"heic-bytes",
                headers={"content-type": "image/heic"},
            )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(analyzer.calls, [])

    def test_multipart_upload_is_rejected_before_analyzer(self) -> None:
        analyzer = RecordingAnalyzer()
        with patch.object(main, "analysis_engine", analyzer):
            response = self.client.post(
                "/api/analyze",
                files={"image": ("fundus.jpg", b"jpeg-bytes", "image/jpeg")},
            )

        self.assertEqual(response.status_code, 415)
        self.assertIn("raw request body", response.json()["detail"])
        self.assertEqual(analyzer.calls, [])

    def test_unsupported_content_type_is_rejected_before_analyzer(self) -> None:
        analyzer = RecordingAnalyzer()
        with patch.object(main, "analysis_engine", analyzer):
            response = self.client.post(
                "/api/analyze",
                content=b"not-an-image",
                headers={"content-type": "application/octet-stream"},
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(analyzer.calls, [])

    def test_workflow_rejects_video_before_analyzer(self) -> None:
        analyzer = RecordingAnalyzer()
        with patch.object(main, "analysis_engine", analyzer):
            response = self.client.post(
                "/api/workflow",
                content=b"mp4-video-bytes",
                headers={"content-type": "video/mp4"},
            )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(analyzer.calls, [])

    def test_analyze_rejects_empty_request(self) -> None:
        response = self.client.post(
            "/api/analyze",
            content=b"",
            headers={"content-type": "image/jpeg"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
