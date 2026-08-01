import json
import hashlib
import os
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

from analyzer import (
    AnalyzerError,
    DemoAnalyzer,
    HybridLocalAnalyzer,
    LocalOpenAIAnalyzer,
    QUALITY_ATTENTION_LABEL,
    SPECIALIST_DEMO_IMAGE_SHA256,
    SpecialistLocalAnalyzer,
    _chat_completions_url,
    _extract_json_object,
    build_analyzer,
    normalize_model_result,
)


class FakeAssessment:
    decision = "READY"
    ready_score = 0.96
    ready_threshold = 0.95
    retake_threshold = 0.01
    scores = {"artifact": 88, "clarity": 92, "field_definition": 90}
    issue_codes: list[str] = []
    quality_attention = None

    def prompt_context(self) -> str:
        return "calibrated specialist evidence"


class FakeSpecialist:
    bundle_verified = True

    def assess(self, image_bytes: bytes) -> FakeAssessment:
        assert image_bytes == b"image"
        return FakeAssessment()


class FakeGemma:
    model_label = "Gemma test"

    def __init__(self, status: str) -> None:
        self.status = status
        self.context: str | None = None

    def runtime_status(self) -> dict[str, object]:
        return {"status": "ready", "profile": "tuned-lora", "model_verified": True}

    def _request(
        self, image_bytes: bytes, content_type: str, quality_context: str | None = None
    ) -> dict[str, object]:
        self.context = quality_context
        return {"status": self.status, "issues": []}


class FailingGemma(FakeGemma):
    def _request(
        self, image_bytes: bytes, content_type: str, quality_context: str | None = None
    ) -> dict[str, object]:
        raise AnalyzerError("local endpoint unavailable")


def model_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "decision": "READY",
        "confidence": 0.93,
        "issues": [],
        "scores": {
            "artifact": 90,
            "clarity": 91,
            "field_definition": 92,
        },
        "retake_instruction": None,
        "disclaimer": "Technical image-quality assessment only; not a diagnosis.",
    }
    payload.update(overrides)
    return payload


class AnalyzerTests(unittest.TestCase):
    def test_specialist_only_releases_valid_ready_with_local_contract(self) -> None:
        analyzer = SpecialistLocalAnalyzer(specialist=FakeSpecialist())

        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["mode"], "specialist-local")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["scores"]["Clarity"], 92)
        self.assertEqual(
            result["decision_trace"],
            {
                "specialist": "READY decision",
                "gemma": "Not used",
                "policy": "READY",
            },
        )
        self.assertNotIn("diagnos", result["summary"].lower())

    def test_specialist_only_runtime_status_is_exact_hash_and_local_only(self) -> None:
        analyzer = SpecialistLocalAnalyzer(specialist=FakeSpecialist())

        status = analyzer.runtime_status()

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["profile"], "quality-specialist")
        self.assertTrue(status["model_verified"])
        self.assertTrue(status["specialist_verified"])
        self.assertFalse(status["lora_verified"])
        self.assertEqual(status["privacy"], "local-only")
        self.assertFalse(status["network_required"])
        self.assertEqual(status["input_scope"], "caller-managed")

    def test_specialist_only_rejects_non_allowlisted_input_before_inference(self) -> None:
        class MustNotRun(FakeSpecialist):
            def assess(self, image_bytes: bytes) -> FakeAssessment:
                raise AssertionError("out-of-scope bytes reached the quality model")

        analyzer = SpecialistLocalAnalyzer(
            specialist=MustNotRun(),
            input_allowlist=frozenset({hashlib.sha256(b"fixed-sample").hexdigest()}),
        )
        result = asyncio.run(
            analyzer.analyze(
                b"arbitrary-non-fundus",
                filename="emoji.png",
                content_type="image/png",
            )
        )

        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["eyebrow"], "Outside demo dataset")
        self.assertEqual(result["issues"], ["Outside validated demo set"])
        self.assertEqual(
            result["decision_trace"]["specialist"], "Outside dataset scope"
        )

    def test_specialist_video_candidate_bypasses_only_the_input_hash_pin(self) -> None:
        class RecordingSpecialist(FakeSpecialist):
            def __init__(self) -> None:
                self.seen: list[bytes] = []

            def assess(self, image_bytes: bytes) -> FakeAssessment:
                self.seen.append(image_bytes)
                return FakeAssessment()

        specialist = RecordingSpecialist()
        analyzer = SpecialistLocalAnalyzer(
            specialist=specialist,
            input_allowlist=frozenset({hashlib.sha256(b"fixed-sample").hexdigest()}),
        )

        normal = asyncio.run(
            analyzer.analyze(
                b"video-derived-jpeg",
                filename="candidate.jpg",
                content_type="image/jpeg",
            )
        )
        experimental = asyncio.run(
            analyzer.analyze(
                b"video-derived-jpeg",
                filename="candidate.jpg",
                content_type="image/jpeg",
                allow_experimental_input=True,
            )
        )

        self.assertEqual(normal["status"], "LIMITED")
        self.assertEqual(normal["decision_trace"]["specialist"], "Outside dataset scope")
        self.assertEqual(experimental["status"], "READY")
        self.assertEqual(specialist.seen, [b"video-derived-jpeg"])

    def test_specialist_video_candidate_still_requires_verified_bundle_and_type(self) -> None:
        class MustNotRun(FakeSpecialist):
            bundle_verified = False

            def assess(self, image_bytes: bytes) -> FakeAssessment:
                raise AssertionError("unverified specialist must not run")

        analyzer = SpecialistLocalAnalyzer(
            specialist=MustNotRun(),
            input_allowlist=frozenset({hashlib.sha256(b"fixed-sample").hexdigest()}),
        )
        unavailable = asyncio.run(
            analyzer.analyze(
                b"video-derived-jpeg",
                filename="candidate.jpg",
                content_type="image/jpeg",
                allow_experimental_input=True,
            )
        )
        unsupported = asyncio.run(
            SpecialistLocalAnalyzer(specialist=FakeSpecialist()).analyze(
                b"image",
                filename="candidate.dcm",
                content_type="application/dicom",
                allow_experimental_input=True,
            )
        )

        self.assertEqual(unavailable["status"], "LIMITED")
        self.assertEqual(unavailable["decision_trace"]["specialist"], "Unavailable")
        self.assertEqual(unsupported["status"], "LIMITED")
        self.assertEqual(unsupported["eyebrow"], "Unsupported image")

    def test_specialist_only_abstention_preserves_limited_contract(self) -> None:
        class LimitedAssessment(FakeAssessment):
            decision = "LIMITED"
            ready_score = 0.5
            issue_codes = ["uncertain"]

        class LimitedSpecialist(FakeSpecialist):
            def assess(self, image_bytes: bytes) -> LimitedAssessment:
                return LimitedAssessment()

        result = asyncio.run(
            SpecialistLocalAnalyzer(specialist=LimitedSpecialist()).analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "LIMITED")
        self.assertIsNone(result["scores"])
        self.assertIn("abstained", result["summary"])
        self.assertEqual(result["decision_trace"]["gemma"], "Not used")

    def test_specialist_only_decode_failure_fails_closed_as_limited(self) -> None:
        class DecodeFailure(FakeSpecialist):
            def assess(self, image_bytes: bytes) -> FakeAssessment:
                raise ValueError("specialist could not decode the image")

        result = asyncio.run(
            SpecialistLocalAnalyzer(specialist=DecodeFailure()).analyze(
                b"broken", filename="broken.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["eyebrow"], "Unsupported image")
        self.assertIsNone(result["scores"])

    def test_specialist_only_inference_failure_fails_closed_as_limited(self) -> None:
        class InferenceFailure(FakeSpecialist):
            def assess(self, image_bytes: bytes) -> FakeAssessment:
                raise RuntimeError("device error")

        result = asyncio.run(
            SpecialistLocalAnalyzer(specialist=InferenceFailure()).analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["issues"], ["Assessment uncertain"])
        self.assertEqual(result["decision_trace"]["specialist"], "Inference failed")

    def test_specialist_only_malformed_decision_fails_closed(self) -> None:
        class MalformedAssessment(FakeAssessment):
            decision = "READY"
            ready_score = 0.5

        class MalformedSpecialist(FakeSpecialist):
            def assess(self, image_bytes: bytes) -> MalformedAssessment:
                return MalformedAssessment()

        result = asyncio.run(
            SpecialistLocalAnalyzer(specialist=MalformedSpecialist()).analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["decision_trace"]["specialist"], "Invalid output")

    def test_specialist_only_unhashable_issue_payload_fails_closed(self) -> None:
        class MalformedAssessment(FakeAssessment):
            issue_codes = [{"unsafe": "shape"}]

        class MalformedSpecialist(FakeSpecialist):
            def assess(self, image_bytes: bytes) -> MalformedAssessment:
                return MalformedAssessment()

        result = asyncio.run(
            SpecialistLocalAnalyzer(specialist=MalformedSpecialist()).analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["decision_trace"]["specialist"], "Invalid output")

    def test_specialist_only_rechecks_bundle_before_each_decision(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        bundle_root = project_root / "models/retinaready-quality-specialist"
        paths = (
            bundle_root / "densenet121-a639ec97.pth",
            bundle_root / "decision-head.pt",
            bundle_root / "factor-head.pt",
        )
        specialist = FakeSpecialist()
        analyzer = SpecialistLocalAnalyzer(
            specialist=specialist,
            bundle_paths=paths,
        )
        self.assertEqual(analyzer.runtime_status()["status"], "ready")

        with patch.object(
            SpecialistLocalAnalyzer, "_sha256", return_value="0" * 64
        ):
            result = asyncio.run(
                analyzer.analyze(
                    b"image", filename="fundus.jpg", content_type="image/jpeg"
                )
            )

        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["eyebrow"], "Quality gate unavailable")
        with patch.object(
            SpecialistLocalAnalyzer, "_sha256", return_value="0" * 64
        ):
            self.assertEqual(analyzer.runtime_status()["status"], "unavailable")

    def test_hybrid_exposes_optional_quality_attention_without_changing_gate(self) -> None:
        class AttentionAssessment(FakeAssessment):
            decision = "RETAKE"
            scores = {"artifact": 20, "clarity": 30, "field_definition": 80}
            issue_codes = ["artifact", "blur"]
            quality_attention = {
                "label": QUALITY_ATTENTION_LABEL,
                "factor": "clarity",
                "factor_label": "Clarity",
                "method": "factor-grad-cam",
                "image_data_url": "data:image/png;base64,cG5n",
            }

        class AttentionSpecialist:
            bundle_verified = True

            def assess(self, image_bytes: bytes) -> AttentionAssessment:
                return AttentionAssessment()

        analyzer = HybridLocalAnalyzer(
            gemma=FakeGemma("RETAKE"), specialist=AttentionSpecialist()
        )
        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "RETAKE")
        self.assertEqual(result["quality_attention"]["factor"], "clarity")

    def test_hybrid_drops_attention_if_gemma_vetoes_retake(self) -> None:
        class AttentionAssessment(FakeAssessment):
            decision = "RETAKE"
            scores = {"artifact": 20, "clarity": 30, "field_definition": 80}
            issue_codes = ["artifact", "blur"]
            quality_attention = {
                "label": QUALITY_ATTENTION_LABEL,
                "factor": "artifact",
                "factor_label": "Artifact quality",
                "method": "factor-grad-cam",
                "image_data_url": "data:image/png;base64,cG5n",
            }

        class AttentionSpecialist:
            bundle_verified = True

            def assess(self, image_bytes: bytes) -> AttentionAssessment:
                return AttentionAssessment()

        analyzer = HybridLocalAnalyzer(
            gemma=FakeGemma("LIMITED"), specialist=AttentionSpecialist()
        )
        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )

        self.assertEqual(result["status"], "LIMITED")
        self.assertNotIn("quality_attention", result)

    def test_hybrid_drops_malformed_attention_without_affecting_gate(self) -> None:
        class UnsafeAttentionAssessment(FakeAssessment):
            quality_attention = {
                "label": QUALITY_ATTENTION_LABEL,
                "factor": "clarity",
                "factor_label": "Clarity",
                "method": "factor-grad-cam",
                "image_data_url": "https://example.test/attention.png",
            }

        result = HybridLocalAnalyzer._specialist_result(
            UnsafeAttentionAssessment()
        )
        self.assertEqual(result["status"], "READY")
        self.assertNotIn("quality_attention", result)

    def test_hybrid_preserves_matching_calibrated_decision(self) -> None:
        gemma = FakeGemma("READY")
        analyzer = HybridLocalAnalyzer(gemma=gemma, specialist=FakeSpecialist())
        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["scores"]["Clarity"], 92)
        self.assertEqual(gemma.context, "calibrated specialist evidence")
        self.assertEqual(
            result["decision_trace"],
            {
                "specialist": "READY candidate",
                "gemma": "Confirmed",
                "policy": "READY",
            },
        )

    def test_hybrid_abstains_when_gemma_conflicts_with_specialist(self) -> None:
        analyzer = HybridLocalAnalyzer(
            gemma=FakeGemma("RETAKE"), specialist=FakeSpecialist()
        )
        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertIsNone(result["scores"])
        self.assertIn("disagree", result["summary"])
        self.assertEqual(result["decision_trace"]["gemma"], "No confirmation")

    def test_hybrid_abstains_when_gemma_is_unavailable(self) -> None:
        analyzer = HybridLocalAnalyzer(
            gemma=FailingGemma("READY"), specialist=FakeSpecialist()
        )
        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertIsNone(result["scores"])
        self.assertIn("unavailable", result["summary"])

    def test_hybrid_skips_gemma_for_calibrated_limited_case(self) -> None:
        class LimitedAssessment(FakeAssessment):
            decision = "LIMITED"
            issue_codes = ["uncertain"]

        class LimitedSpecialist:
            def assess(self, image_bytes: bytes) -> LimitedAssessment:
                return LimitedAssessment()

        gemma = FakeGemma("READY")
        analyzer = HybridLocalAnalyzer(gemma=gemma, specialist=LimitedSpecialist())
        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertIsNone(result["scores"])
        self.assertIsNone(gemma.context)
        self.assertEqual(
            result["decision_trace"],
            {
                "specialist": "Abstained",
                "gemma": "Skipped",
                "policy": "LIMITED",
            },
        )

    def test_hybrid_invalid_image_fails_to_unsupported_limited(self) -> None:
        class InvalidImageSpecialist:
            def assess(self, image_bytes: bytes) -> FakeAssessment:
                raise ValueError("cannot decode")

        gemma = FakeGemma("READY")
        analyzer = HybridLocalAnalyzer(
            gemma=gemma, specialist=InvalidImageSpecialist()
        )
        result = asyncio.run(
            analyzer.analyze(
                b"not-an-image", filename="broken.jpg", content_type="image/jpeg"
            )
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["issues"], ["Unsupported image type"])
        self.assertIsNone(result["scores"])
        self.assertIsNone(gemma.context)

    def test_normalizes_training_contract(self) -> None:
        result = normalize_model_result(
            model_payload(
                decision="RETAKE",
                issues=["blur", "field_cutoff"],
                scores={
                    "artifact": 0.7,
                    "clarity": 28,
                    "field_definition": 42,
                },
                retake_instruction="Refocus and recenter.",
            )
        )
        self.assertEqual(result["status"], "RETAKE")
        self.assertEqual(result["scores"]["Artifact quality"], 1)
        self.assertIn("Motion or focus blur", result["issues"])
        self.assertEqual(result["confidence"], 0.93)

    def test_extracts_json_after_gemma_reasoning_channel(self) -> None:
        payload = model_payload()
        content = (
            "<|channel>thought\nThe schema should be followed.<channel|>\n"
            + json.dumps(payload)
        )
        self.assertEqual(_extract_json_object(content), payload)

    def test_extracts_json_from_code_fence_and_content_blocks(self) -> None:
        payload = model_payload(decision="RETAKE")
        content = [
            {"type": "text", "text": "```json\n"},
            {"type": "text", "text": json.dumps(payload)},
            {"type": "text", "text": "\n```"},
        ]
        self.assertEqual(_extract_json_object(content), payload)

    def test_extractor_skips_non_contract_json_in_prefix(self) -> None:
        payload = model_payload()
        content = '{"scratch": true}\n```json\n' + json.dumps(payload) + "\n```"
        self.assertEqual(_extract_json_object(content), payload)

    def test_unsupported_input_is_always_limited_without_scores(self) -> None:
        result = normalize_model_result(
            model_payload(
                decision="READY",
                confidence=None,
                issues=["unsupported_modality"],
                scores={
                    "artifact": None,
                    "clarity": None,
                    "field_definition": None,
                },
            )
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["eyebrow"], "Unsupported image")
        self.assertEqual(result["scores"], None)
        self.assertIn("Unsupported image type", result["issues"])

    def test_uncertain_input_is_always_limited(self) -> None:
        result = normalize_model_result(
            model_payload(decision="RETAKE", issues=["uncertain"])
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["eyebrow"], "Unable to assess")
        self.assertEqual(result["scores"], None)

    def test_limited_without_reason_gains_uncertain_issue(self) -> None:
        result = normalize_model_result(model_payload(decision="LIMITED"))
        self.assertEqual(result["status"], "LIMITED")
        self.assertIn("Assessment uncertain", result["issues"])

    def test_partial_scores_omit_absent_values(self) -> None:
        result = normalize_model_result(
            model_payload(
                decision="RETAKE",
                scores={
                    "artifact": None,
                    "clarity": 61,
                    "field_definition": None,
                }
            )
        )
        self.assertEqual(result["scores"], {"Clarity": 61})

    def test_contradictory_low_confidence_ready_fails_closed(self) -> None:
        result = normalize_model_result(
            model_payload(
                confidence=0.01,
                issues=["blur"],
                scores={
                    "artifact": 10,
                    "clarity": 10,
                    "field_definition": 10,
                },
            )
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["scores"], None)
        self.assertIn("Assessment uncertain", result["issues"])

    def test_model_authored_instruction_is_not_displayed(self) -> None:
        result = normalize_model_result(
            model_payload(
                decision="RETAKE",
                issues=["blur"],
                retake_instruction="Take a drug and diagnose the patient.",
            )
        )
        self.assertEqual(
            result["instruction"],
            "Stabilize the camera and refocus, then retake the image.",
        )
        self.assertNotIn("drug", result["instruction"])

    def test_out_of_range_confidence_is_rejected(self) -> None:
        with self.assertRaises(AnalyzerError):
            normalize_model_result(model_payload(confidence=1.01))

    def test_out_of_range_score_is_rejected(self) -> None:
        with self.assertRaises(AnalyzerError):
            normalize_model_result(
                model_payload(
                    scores={
                        "artifact": 101,
                        "clarity": 50,
                        "field_definition": 50,
                    }
                )
            )

    def test_unknown_issue_code_is_rejected(self) -> None:
        with self.assertRaises(AnalyzerError):
            normalize_model_result(model_payload(issues=["diagnosis"]))

    def test_malformed_model_answer_becomes_uncertain_limited_result(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                envelope = {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"decision":"READY","confidence":4,'
                                    '"issues":[],"scores":{}}'
                                )
                            }
                        }
                    ]
                }
                return json.dumps(envelope).encode()

        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: float) -> FakeResponse:
            captured["body"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
            captured["timeout"] = timeout
            return FakeResponse()

        analyzer = LocalOpenAIAnalyzer(
            api_url="http://127.0.0.1:8080",
            model_id="gemma-test",
            timeout_seconds=12,
        )
        with patch("analyzer.urlopen", side_effect=fake_urlopen):
            result = analyzer._request(b"image", "image/jpeg")

        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["scores"], None)
        self.assertEqual(captured["timeout"], 12)
        request_body = captured["body"]
        self.assertEqual(  # type: ignore[index]
            request_body["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_endpoint_url_accepts_base_or_v1(self) -> None:
        expected = "http://127.0.0.1:8080/v1/chat/completions"
        self.assertEqual(_chat_completions_url("http://127.0.0.1:8080"), expected)
        self.assertEqual(_chat_completions_url("http://127.0.0.1:8080/v1"), expected)
        self.assertEqual(_chat_completions_url(expected), expected)

    def test_auto_mode_defaults_to_demo(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(build_analyzer(), DemoAnalyzer)

    def test_auto_mode_uses_local_url(self) -> None:
        with patch.dict(
            os.environ,
            {"GEMMA_API_URL": "http://127.0.0.1:8080", "MODEL_ID": "local-model"},
            clear=True,
        ):
            analyzer = build_analyzer()
            self.assertIsInstance(analyzer, LocalOpenAIAnalyzer)
            self.assertEqual(analyzer.model_label, "local-model")

    def test_explicit_specialist_mode_does_not_require_gemma_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RETINA_ANALYZER": "specialist",
                "RETINA_MODEL_LABEL": "Compact quality gate",
            },
            clear=True,
        ), patch(
            "quality_specialist.QualitySpecialist", return_value=FakeSpecialist()
        ) as specialist_type:
            analyzer = build_analyzer()

        self.assertIsInstance(analyzer, SpecialistLocalAnalyzer)
        self.assertEqual(analyzer.mode, "specialist-local")
        self.assertEqual(analyzer.model_label, "Compact quality gate")
        self.assertEqual(analyzer.input_allowlist, SPECIALIST_DEMO_IMAGE_SHA256)
        specialist_type.assert_called_once()

    def test_specialist_bundle_load_error_builds_unavailable_fail_closed_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"RETINA_ANALYZER": "specialist"},
            clear=True,
        ), patch(
            "quality_specialist.QualitySpecialist",
            side_effect=ValueError("checksum mismatch"),
        ):
            analyzer = build_analyzer()

        self.assertIsInstance(analyzer, SpecialistLocalAnalyzer)
        self.assertEqual(analyzer.runtime_status()["status"], "unavailable")
        result = asyncio.run(
            analyzer.analyze(
                b"image", filename="fundus.jpg", content_type="image/jpeg"
            )
        )
        self.assertEqual(result["status"], "LIMITED")
        self.assertEqual(result["eyebrow"], "Quality gate unavailable")

    def test_remote_model_url_is_rejected_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                LocalOpenAIAnalyzer(
                    api_url="https://example.com/v1",
                    model_id="remote-model",
                )

    def test_runtime_status_verifies_health_and_model_alias(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode()

        responses = [
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [{"id": "expected-model"}]}),
            FakeResponse([{"id": 0, "path": "/models/adapter.gguf", "scale": 1.0}]),
        ]
        analyzer = LocalOpenAIAnalyzer(
            api_url="http://127.0.0.1:8080",
            model_id="expected-model",
            model_profile="tuned-lora",
        )
        with patch("analyzer.urlopen", side_effect=responses):
            status = analyzer.runtime_status()
        self.assertEqual(status["status"], "ready")
        self.assertTrue(status["model_verified"])
        self.assertTrue(status["lora_verified"])
        self.assertEqual(status["profile"], "tuned-lora")

    def test_runtime_status_fails_closed_when_alias_is_missing(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode()

        responses = [
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [{"id": "different-model"}]}),
        ]
        analyzer = LocalOpenAIAnalyzer(
            api_url="http://127.0.0.1:8080",
            model_id="expected-model",
        )
        with patch("analyzer.urlopen", side_effect=responses):
            status = analyzer.runtime_status()
        self.assertEqual(status["status"], "unavailable")
        self.assertFalse(status["model_verified"])


if __name__ == "__main__":
    unittest.main()
