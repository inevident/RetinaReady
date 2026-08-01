import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from escalation_specialist import (
    EscalationIntegrityError,
    LocalEscalationSpecialistAdapter,
)
from quality_specialist import QualitySpecialist
from workflow import (
    EscalationDecision,
    EscalationReason,
    ProductMode,
    UnavailableEscalationAdapter,
    WorkflowOrchestrator,
    build_escalation_adapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMOTION_MANIFEST = (
    PROJECT_ROOT
    / "models/retinaready-escalation-demo/promotion-manifest.json"
)
OPT_IN = {"RETINA_ENABLE_ESCALATION_RESEARCH_DEMO": "1"}


class EscalationPromotionTests(unittest.TestCase):
    def test_opt_in_is_required_by_builder(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            adapter = build_escalation_adapter()
        self.assertIsInstance(adapter, UnavailableEscalationAdapter)
        status = adapter.runtime_status()
        self.assertFalse(status["release_enabled"])
        self.assertFalse(status["network_required"])

    def test_opted_in_builder_returns_verified_local_adapter(self) -> None:
        with patch.dict(os.environ, OPT_IN, clear=True):
            adapter = build_escalation_adapter()
            status = adapter.runtime_status()
        self.assertIsInstance(adapter, LocalEscalationSpecialistAdapter)
        self.assertTrue(status["release_enabled"])
        self.assertFalse(status["network_required"])

    def test_promotion_manifest_binds_exact_artifact_report_and_backbone(self) -> None:
        promotion = json.loads(PROMOTION_MANIFEST.read_text())
        self.assertEqual(
            promotion["scope"], "nonclinical-hackathon-research-demo-only"
        )
        self.assertFalse(promotion["network_required"])
        self.assertEqual(promotion["fail_closed_decision"], "UNCERTAIN")
        for binding in promotion["bindings"].values():
            path = PROJECT_ROOT / binding["file"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), binding["sha256"]
            )

    def test_source_artifact_remains_experimental_and_unauthorized(self) -> None:
        promotion = json.loads(PROMOTION_MANIFEST.read_text())
        artifact_path = PROJECT_ROOT / promotion["bindings"]["artifact"]["file"]
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        self.assertTrue(artifact["experimental_only"])
        self.assertFalse(artifact["runtime_integration_authorized"])
        self.assertFalse(artifact["diagnostic_use_authorized"])

    def test_checksum_mismatch_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ("artifact", "report", "backbone"):
                path = root / f"{name}.bin"
                path.write_bytes(name.encode())
                files[name] = path
            manifest = json.loads(PROMOTION_MANIFEST.read_text())
            manifest["bindings"] = {
                name: {
                    "file": path.name,
                    "sha256": (
                        "0" * 64
                        if name == "artifact"
                        else hashlib.sha256(path.read_bytes()).hexdigest()
                    ),
                }
                for name, path in files.items()
            }
            manifest_path = root / "promotion.json"
            manifest_path.write_text(json.dumps(manifest))
            with patch.dict(os.environ, OPT_IN, clear=True):
                with self.assertRaisesRegex(
                    EscalationIntegrityError, "artifact checksum mismatch"
                ):
                    LocalEscalationSpecialistAdapter(
                        project_root=root,
                        promotion_manifest_path=manifest_path,
                    )


class EscalationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = patch.dict(os.environ, OPT_IN, clear=True)
        cls.environment.start()
        cls.adapter = LocalEscalationSpecialistAdapter(
            project_root=PROJECT_ROOT,
            promotion_manifest_path=PROMOTION_MANIFEST,
            device="cpu",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.environment.stop()

    def test_runtime_status_is_local_verified_and_research_scoped(self) -> None:
        status = self.adapter.runtime_status()
        self.assertEqual(status["status"], "ready")
        self.assertTrue(status["model_verified"])
        self.assertTrue(status["report_verified"])
        self.assertTrue(status["promotion_verified"])
        self.assertTrue(status["release_enabled"])
        self.assertFalse(status["network_required"])
        self.assertEqual(
            status["scope"], "nonclinical-hackathon-research-demo-only"
        )

    def test_preprocessing_exactly_matches_baseline_quality_preprocessing(self) -> None:
        image = Image.new("RGB", (80, 60), (0, 0, 0))
        for x in range(15, 65):
            for y in range(10, 50):
                image.putpixel((x, y), (100 + x, 40 + y, 25))
        baseline = QualitySpecialist.__new__(QualitySpecialist)
        expected = baseline._preprocess(image)
        actual = self.adapter._preprocess(image)
        self.assertTrue(torch.equal(actual, expected))

    def test_strict_thresholds_map_internal_labels_to_review_labels(self) -> None:
        lower = self.adapter._policy["routine_if_score_strictly_less_than"]
        upper = self.adapter._policy["priority_if_score_strictly_greater_than"]
        self.assertEqual(
            self.adapter._assessment_from_score(lower - 1e-8).decision,
            EscalationDecision.ROUTINE_REVIEW,
        )
        self.assertEqual(
            self.adapter._assessment_from_score(upper + 1e-8).decision,
            EscalationDecision.PRIORITY_REVIEW,
        )
        for score in (lower, (lower + upper) / 2, upper):
            result = self.adapter._assessment_from_score(score)
            self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
            self.assertFalse(result.release_allowed)
            self.assertEqual(result.reason, EscalationReason.MODEL_ABSTAINED)

    def test_real_images_reproduce_frozen_report_scores_and_decisions(self) -> None:
        report = json.loads(
            (PROJECT_ROOT / "outputs/escalation-baseline/report.json").read_text()
        )
        expected = {
            row["image_id"]: row
            for row in report["official_validation_results"]
            if row["image_id"] in {"298_l2", "265_l2", "265_l1"}
        }
        mapping = {
            "ROUTINE": EscalationDecision.ROUTINE_REVIEW,
            "PRIORITY": EscalationDecision.PRIORITY_REVIEW,
            "UNCERTAIN": EscalationDecision.UNCERTAIN,
        }
        for image_id, row in expected.items():
            with self.subTest(image_id=image_id):
                image_bytes = (PROJECT_ROOT / row["image_path"]).read_bytes()
                score = self.adapter._score(image_bytes)
                # CPU convolution kernels can differ by a few float32 ULPs;
                # require practical identity while separately asserting the
                # exact preprocessing tensor and the released decision.
                self.assertAlmostEqual(
                    score, row["review_priority_score"], delta=2e-6
                )
                result = asyncio.run(
                    self.adapter.assess(
                        image_bytes,
                        filename=f"{image_id}.jpg",
                        content_type="image/jpeg",
                    )
                )
                self.assertEqual(result.decision, mapping[row["decision"]])
                self.assertEqual(
                    result.release_allowed,
                    row["decision"] in {"ROUTINE", "PRIORITY"},
                )

    def test_real_priority_image_flows_through_combined_quality_first_policy(self) -> None:
        class ReadyQuality:
            async def analyze(self, *args: object, **kwargs: object) -> dict[str, object]:
                return {
                    "status": "READY",
                    "eyebrow": "Ready",
                    "summary": "Technical quality passed.",
                    "confidence": None,
                    "issues": [],
                    "instruction": "Continue.",
                    "scores": None,
                    "disclaimer": "Technical quality only.",
                    "mode": "test",
                }

        image_path = (
            PROJECT_ROOT
            / "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-validation/Images/265/265_l2.jpg"
        )
        result = asyncio.run(
            WorkflowOrchestrator(
                quality=ReadyQuality(), escalation=self.adapter
            ).run(
                ProductMode.COMBINED,
                image_path.read_bytes(),
                filename=image_path.name,
                content_type="image/jpeg",
            )
        )
        self.assertEqual(result.quality_assessment["status"], "READY")
        self.assertEqual(result.display["status"], "PRIORITY_REVIEW")
        self.assertTrue(result.escalation_assessment.release_allowed)
        self.assertIn("clinician", result.display["summary"].lower())

    def test_revoked_opt_in_fails_closed_even_after_model_loaded(self) -> None:
        image_path = (
            PROJECT_ROOT
            / "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-validation/Images/298/298_l2.jpg"
        )
        with patch.dict(os.environ, {}, clear=True):
            result = asyncio.run(
                self.adapter.assess(
                    image_path.read_bytes(),
                    filename=image_path.name,
                    content_type="image/jpeg",
                )
            )
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ARTIFACT_UNAVAILABLE)
        self.assertFalse(result.release_allowed)

    def test_post_load_integrity_failure_fails_closed(self) -> None:
        with patch.object(
            self.adapter,
            "_verify_bound_files",
            side_effect=EscalationIntegrityError("tampered"),
        ):
            result = asyncio.run(
                self.adapter.assess(
                    b"not inspected because integrity fails first",
                    filename="capture.jpg",
                    content_type="image/jpeg",
                )
            )
        self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
        self.assertEqual(result.reason, EscalationReason.ARTIFACT_UNAVAILABLE)
        self.assertFalse(result.release_allowed)

    def test_decode_and_content_type_errors_fail_closed(self) -> None:
        invalid = asyncio.run(
            self.adapter.assess(
                b"not an image",
                filename="bad.jpg",
                content_type="image/jpeg",
            )
        )
        unsupported = asyncio.run(
            self.adapter.assess(
                b"anything",
                filename="bad.gif",
                content_type="image/gif",
            )
        )
        for result in (invalid, unsupported):
            self.assertEqual(result.decision, EscalationDecision.UNCERTAIN)
            self.assertFalse(result.release_allowed)


if __name__ == "__main__":
    unittest.main()
