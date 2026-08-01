import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gemma_escalation import (
    ESCALATION_DISCLAIMER,
    ESCALATION_RESPONSE_SCHEMA,
    ESCALATION_SYSTEM_PROMPT,
    ESCALATION_USER_PROMPT,
    GEMMA_ESCALATION_DEMO_IMAGE_SHA256,
    GEMMA_ESCALATION_PROFILE,
    LocalGemmaEscalationAdapter,
)
from workflow import (
    EscalationDecision,
    EscalationReason,
    UnavailableEscalationAdapter,
    ProductMode,
    WorkflowOrchestrator,
    build_escalation_adapter,
)


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None


class GemmaEscalationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.opt_in = patch.dict(
            "os.environ",
            {"RETINA_ENABLE_ESCALATION_RESEARCH_DEMO": "1"},
            clear=False,
        )
        self.opt_in.start()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lora_path = Path(self.temp_dir.name) / "retinapriority-lora.gguf"
        self.lora_path.write_bytes(b"fixed-test-lora")
        self.lora_sha256 = hashlib.sha256(self.lora_path.read_bytes()).hexdigest()
        self.image = b"fixed-quality-passing-deepdrid-image"
        self.image_hash = hashlib.sha256(self.image).hexdigest()
        self.model_id = "retinapriority-gemma4-26b-test"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self.opt_in.stop()

    def adapter(self, **overrides: object) -> LocalGemmaEscalationAdapter:
        values = {
            "api_url": "http://127.0.0.1:8082",
            "model_id": self.model_id,
            "lora_path": self.lora_path,
            "lora_sha256": self.lora_sha256,
            "timeout_seconds": 5,
            "input_allowlist": frozenset({self.image_hash}),
        }
        values.update(overrides)
        return LocalGemmaEscalationAdapter(**values)

    @staticmethod
    def target(decision: str) -> dict[str, object]:
        return {
            "confidence": None,
            "decision": decision,
            "disclaimer": ESCALATION_DISCLAIMER,
            "next_step": (
                "Route for priority clinician review."
                if decision == "PRIORITY"
                else "Keep in the routine clinician review queue."
            ),
        }

    def server(self, target: object):
        def responder(request, timeout: float):
            del timeout
            url = request.full_url
            if url.endswith("/health"):
                return FakeResponse({"status": "ok"})
            if url.endswith("/v1/models"):
                return FakeResponse({"data": [{"id": self.model_id}]})
            if url.endswith("/lora-adapters"):
                return FakeResponse(
                    [{"id": 0, "path": str(self.lora_path.resolve()), "scale": 1.0}]
                )
            if url.endswith("/v1/chat/completions"):
                body = json.loads(request.data.decode("utf-8"))
                self.assertEqual(body["model"], self.model_id)
                self.assertEqual(
                    body["response_format"],
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "retinapriority_assessment",
                            "strict": True,
                            "schema": ESCALATION_RESPONSE_SCHEMA,
                        },
                    },
                )
                self.assertEqual(body["messages"][0]["content"], ESCALATION_SYSTEM_PROMPT)
                self.assertEqual(
                    body["messages"][1]["content"][0]["text"],
                    ESCALATION_USER_PROMPT,
                )
                content = target if isinstance(target, str) else json.dumps(target)
                return FakeResponse(
                    {"choices": [{"message": {"content": content}}]}
                )
            raise AssertionError(f"unexpected URL: {url}")

        return responder

    def assess(self, adapter: LocalGemmaEscalationAdapter):
        return asyncio.run(
            adapter.assess(
                self.image,
                filename="296_l2.jpg",
                content_type="image/jpeg",
            )
        )

    def test_rejects_every_non_loopback_or_credentialed_url(self) -> None:
        invalid_urls = (
            "https://127.0.0.1:8082",
            "http://localhost:8082",
            "http://example.com:8082",
            "http://user:secret@127.0.0.1:8082",
            "http://127.0.0.1:8082/v1?target=remote",
        )
        for url in invalid_urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.adapter(api_url=url)

    def test_prompt_constants_match_training_source_exactly(self) -> None:
        training_path = Path(__file__).resolve().parents[2] / "ml/train_qlora.py"
        spec = importlib.util.spec_from_file_location(
            "retinapriority_training_contract", training_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(ESCALATION_SYSTEM_PROMPT, module.ESCALATION_SYSTEM_PROMPT)
        self.assertEqual(ESCALATION_USER_PROMPT, module.ESCALATION_USER_PROMPT)
        self.assertEqual(ESCALATION_DISCLAIMER, module.ESCALATION_DISCLAIMER)
        target = module.target_for(
            {"escalation_label": "PRIORITY"}, task="escalation"
        )
        self.assertEqual(target, self.target("PRIORITY"))

    def test_default_input_pins_are_exact_quality_passing_demo_images(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        image_paths = (
            project_root
            / "data/raw/deepdrid-v1.1/regular_fundus_images/"
            "regular-fundus-training/Images/146/146_l2.jpg",
            project_root
            / "data/raw/deepdrid-v1.1/regular_fundus_images/"
            "regular-fundus-validation/Images/296/296_l2.jpg",
        )
        observed = frozenset(
            hashlib.sha256(image_path.read_bytes()).hexdigest()
            for image_path in image_paths
        )
        self.assertIn(
            "21ef6838c18ccfe8697a1e2f4a31d2cce2cb11eb2627995a977d5aaaa9aeeda7",
            observed,
        )
        self.assertEqual(GEMMA_ESCALATION_DEMO_IMAGE_SHA256, observed)

    def test_health_verifies_exact_alias_lora_path_scale_and_hash(self) -> None:
        adapter = self.adapter()
        with patch(
            "gemma_escalation._open_loopback",
            side_effect=self.server(self.target("PRIORITY")),
        ):
            status = adapter.runtime_status()
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["profile"], GEMMA_ESCALATION_PROFILE)
        self.assertTrue(status["model_verified"])
        self.assertTrue(status["lora_verified"])
        self.assertTrue(status["adapter_hash_verified"])
        self.assertTrue(status["release_enabled"])
        self.assertEqual(
            status["input_scope"],
            "fixed-deepdrid-quality-pass-demo-samples",
        )
        self.assertEqual(status["calibration"], "uncalibrated-free-generation-experimental")
        self.assertFalse(status["clinical_use"])
        self.assertFalse(status["network_required"])
        self.assertTrue(status["loopback_http_required"])

    def test_maps_only_priority_and_routine_to_review_labels(self) -> None:
        for internal, expected in (
            ("PRIORITY", EscalationDecision.PRIORITY_REVIEW),
            ("ROUTINE", EscalationDecision.ROUTINE_REVIEW),
        ):
            with self.subTest(internal=internal):
                adapter = self.adapter()
                with patch(
                    "gemma_escalation._open_loopback",
                    side_effect=self.server(self.target(internal)),
                ):
                    result = self.assess(adapter)
                self.assertEqual(result.decision, expected)
                self.assertTrue(result.executed)
                self.assertTrue(result.model_available)
                self.assertTrue(result.release_allowed)
                self.assertIsNone(result.confidence)
                self.assertIn("uncalibrated", result.summary.lower())

    def test_fixed_input_hash_and_content_type_fail_before_network(self) -> None:
        adapter = self.adapter()
        with patch("gemma_escalation._open_loopback") as mocked:
            wrong_image = asyncio.run(
                adapter.assess(
                    b"another-image",
                    filename="random.jpg",
                    content_type="image/jpeg",
                )
            )
            wrong_type = asyncio.run(
                adapter.assess(
                    self.image,
                    filename="296_l2.dcm",
                    content_type="application/dicom",
                )
            )
            wrong_declared_image_type = asyncio.run(
                adapter.assess(
                    self.image,
                    filename="296_l2.png",
                    content_type="image/png",
                )
            )
        mocked.assert_not_called()
        for result in (wrong_image, wrong_type, wrong_declared_image_type):
            self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
            self.assertFalse(result.release_allowed)
            self.assertFalse(result.executed)

    def test_combined_non_ready_quality_gate_never_reaches_gemma_server(self) -> None:
        class RetakeQuality:
            async def analyze(self, *args: object, **kwargs: object):
                return {
                    "status": "RETAKE",
                    "eyebrow": "Retake",
                    "summary": "Technically insufficient.",
                    "confidence": None,
                    "issues": ["Blur"],
                    "instruction": "Retake the image.",
                    "scores": None,
                    "disclaimer": "Technical quality only.",
                    "mode": "test",
                }

        adapter = self.adapter()
        with patch("gemma_escalation._open_loopback") as mocked:
            workflow = asyncio.run(
                WorkflowOrchestrator(
                    quality=RetakeQuality(), escalation=adapter
                ).run(
                    ProductMode.COMBINED,
                    self.image,
                    filename="431_l2.jpg",
                    content_type="image/jpeg",
                )
            )
        mocked.assert_not_called()
        self.assertEqual(workflow.display["status"], "RETAKE")
        self.assertEqual(
            workflow.escalation_assessment.decision,
            EscalationDecision.UNCERTAIN,
        )
        self.assertFalse(workflow.escalation_assessment.release_allowed)

    def test_model_alias_or_adapter_identity_failure_never_calls_generation(self) -> None:
        def wrong_alias(request, timeout: float):
            del timeout
            if request.full_url.endswith("/health"):
                return FakeResponse({"status": "ok"})
            if request.full_url.endswith("/v1/models"):
                return FakeResponse({"data": [{"id": "wrong-model"}]})
            raise AssertionError("identity failure must stop before generation")

        adapter = self.adapter()
        with patch("gemma_escalation._open_loopback", side_effect=wrong_alias):
            result = self.assess(adapter)
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ARTIFACT_UNAVAILABLE)
        self.assertFalse(result.release_allowed)

        def wrong_adapter(request, timeout: float):
            del timeout
            if request.full_url.endswith("/health"):
                return FakeResponse({"status": "ok"})
            if request.full_url.endswith("/v1/models"):
                return FakeResponse({"data": [{"id": self.model_id}]})
            if request.full_url.endswith("/lora-adapters"):
                return FakeResponse(
                    [{"id": 0, "path": "/tmp/wrong.gguf", "scale": 1.0}]
                )
            raise AssertionError("identity failure must stop before generation")

        with patch("gemma_escalation._open_loopback", side_effect=wrong_adapter):
            result = self.assess(adapter)
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ARTIFACT_UNAVAILABLE)
        self.assertFalse(result.release_allowed)

    def test_post_start_lora_tamper_fails_before_network(self) -> None:
        adapter = self.adapter()
        self.lora_path.write_bytes(b"tampered")
        with patch("gemma_escalation._open_loopback") as mocked:
            result = self.assess(adapter)
        mocked.assert_not_called()
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ARTIFACT_UNAVAILABLE)
        self.assertFalse(result.release_allowed)

    def test_revoked_research_opt_in_fails_before_network(self) -> None:
        adapter = self.adapter()
        with patch.dict("os.environ", {}, clear=True), patch(
            "gemma_escalation._open_loopback"
        ) as mocked:
            result = self.assess(adapter)
        mocked.assert_not_called()
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ARTIFACT_UNAVAILABLE)
        self.assertFalse(result.release_allowed)

    def test_network_failure_and_bad_health_fail_closed(self) -> None:
        adapter = self.adapter()
        with patch(
            "gemma_escalation._open_loopback", side_effect=OSError("offline")
        ):
            result = self.assess(adapter)
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ADAPTER_ERROR)
        self.assertFalse(result.release_allowed)

        with patch(
            "gemma_escalation._open_loopback",
            return_value=FakeResponse({"status": "loading"}),
        ):
            result = self.assess(adapter)
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ARTIFACT_UNAVAILABLE)
        self.assertFalse(result.release_allowed)

    def test_strict_schema_rejects_prose_fences_unknown_decisions_and_extra_keys(self) -> None:
        invalid_outputs = (
            "```json\n" + json.dumps(self.target("PRIORITY")) + "\n```",
            {**self.target("PRIORITY"), "extra": "unsafe"},
            {**self.target("PRIORITY"), "decision": "UNCERTAIN"},
            {**self.target("PRIORITY"), "confidence": 0.99},
            {**self.target("ROUTINE"), "next_step": "The eye is healthy."},
        )
        for output in invalid_outputs:
            with self.subTest(output=output):
                adapter = self.adapter()
                with patch(
                    "gemma_escalation._open_loopback", side_effect=self.server(output)
                ):
                    result = self.assess(adapter)
                self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
                self.assertEqual(result.reason, EscalationReason.INVALID_OUTPUT)
                self.assertTrue(result.executed)
                self.assertTrue(result.model_available)
                self.assertFalse(result.release_allowed)

    def test_build_selection_requires_opt_in_exact_lora_pin_and_known_engine(self) -> None:
        base_environment = {
            "RETINA_ENABLE_ESCALATION_RESEARCH_DEMO": "1",
            "RETINA_ESCALATION_ENGINE": "gemma",
            "RETINA_ESCALATION_GEMMA_API_URL": "http://127.0.0.1:8082",
            "RETINA_ESCALATION_GEMMA_MODEL_ID": self.model_id,
            "RETINA_ESCALATION_GEMMA_LORA_PATH": str(self.lora_path),
            "RETINA_ESCALATION_GEMMA_LORA_SHA256": self.lora_sha256,
        }
        with patch.dict("os.environ", base_environment, clear=True):
            built = build_escalation_adapter()
        self.assertIsInstance(built, LocalGemmaEscalationAdapter)

        with patch.dict(
            "os.environ",
            {"RETINA_ENABLE_ESCALATION_RESEARCH_DEMO": "1", "RETINA_ESCALATION_ENGINE": "other"},
            clear=True,
        ):
            built = build_escalation_adapter()
        self.assertIsInstance(built, UnavailableEscalationAdapter)
        self.assertFalse(built.runtime_status()["release_enabled"])


if __name__ == "__main__":
    unittest.main()
