import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from calibrate_escalation_adapter import (  # noqa: E402
    CalibrationInputError,
    build_report,
    verify_integrity,
)
from experiment_escalation_baseline import assign  # noqa: E402


FIELDS = [
    "split",
    "patient_id",
    "image_id",
    "image_path",
    "dr_grade",
    "escalation_label",
    "overall_quality",
    "source_split",
    "grade_source_field",
    "filename_side_matches_grade_field",
]


class GemmaEscalationCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calibration_manifest, self.calibration_rows = self.make_manifest(
            "calibration", "c"
        )
        self.evaluation_manifest, self.evaluation_rows = self.make_manifest(
            "eval", "e"
        )
        self.calibration_report = self.make_decision_report(
            "calibration", self.calibration_manifest, self.calibration_rows
        )
        self.evaluation_report = self.make_decision_report(
            "eval", self.evaluation_manifest, self.evaluation_rows
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_manifest(
        self, split: str, prefix: str
    ) -> tuple[Path, list[dict[str, str]]]:
        rows: list[dict[str, str]] = []
        for label, grade, score_group in (
            ("ROUTINE", "0", "routine"),
            ("PRIORITY", "2", "priority"),
        ):
            patient_id = f"{prefix}-{score_group}"
            for view in range(2):
                image_id = f"{patient_id}-{view}"
                image = self.root / f"{image_id}.jpg"
                image.write_bytes(b"not-opened-by-this-test")
                rows.append(
                    {
                        "split": split,
                        "patient_id": patient_id,
                        "image_id": image_id,
                        "image_path": str(image),
                        "dr_grade": grade,
                        "escalation_label": label,
                        "overall_quality": "1",
                        "source_split": (
                            "regular-fundus-training"
                            if split == "calibration"
                            else "regular-fundus-validation"
                        ),
                        "grade_source_field": "left_eye_DR_Level",
                        "filename_side_matches_grade_field": "true",
                    }
                )
        path = self.root / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path, rows

    @staticmethod
    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_decision_report(
        self,
        split: str,
        manifest: Path,
        rows: list[dict[str, str]],
        *,
        include_split: bool = True,
    ) -> Path:
        results = []
        for index, row in enumerate(rows):
            if row["escalation_label"] == "PRIORITY":
                score = 0.82 + index / 1000
            else:
                score = 0.18 + index / 1000
            margin = math.log(score / (1.0 - score))
            results.append(
                {
                    "image_id": row["image_id"],
                    "patient_id": row["patient_id"],
                    "image_path": row["image_path"],
                    "truth": row["escalation_label"],
                    "prediction": (
                        "PRIORITY" if score >= 0.5 else "ROUTINE"
                    ),
                    "positive_label": "PRIORITY",
                    "negative_label": "ROUTINE",
                    "positive_logit": margin,
                    "negative_logit": 0.0,
                    "positive_minus_negative_logit": margin,
                    "positive_probability": score,
                    "negative_probability": 1.0 - score,
                }
            )
        run = {
            "mode": "decision-token-logits",
            "task": "escalation",
            "model_id": "google/gemma-4-26B-A4B-it",
            "model_revision": "model-revision",
            "processor_id": "google/gemma-4-E2B-it",
            "processor_revision": "processor-revision",
            "adapter": {"weights_sha256": "a" * 64},
            "training_contract": {
                "task": "escalation",
                "loss_scope": "decision_token",
                "provenance_sha256": "b" * 64,
            },
            "roc_auc_positive_class": "PRIORITY",
            "manifest": str(manifest),
            "manifest_sha256": self.file_sha(manifest),
            "available_rows": len(rows),
            "selected_rows": len(rows),
        }
        if include_split:
            run["expected_split"] = split
        payload = {
            "run": run,
            "summary": {"samples": len(results)},
            "results": results,
        }
        path = self.root / f"{split}-report.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def build(self, **overrides: object) -> dict:
        arguments = {
            "calibration_report": self.calibration_report,
            "evaluation_report": self.evaluation_report,
            "calibration_manifest": self.calibration_manifest,
            "evaluation_manifest": self.evaluation_manifest,
            "false_routine_risk": 0.5,
            "false_priority_risk": 0.5,
            "delta": 0.49,
        }
        arguments.update(overrides)
        return build_report(**arguments)

    def test_builds_hash_bound_strict_policy_from_disjoint_full_reports(self) -> None:
        report = self.build()
        self.assertTrue(verify_integrity(report))
        policy = report["policy"]
        lower = policy["routine_if_score_strictly_less_than"]
        upper = policy["priority_if_score_strictly_greater_than"]
        self.assertEqual(assign(lower, policy), "UNCERTAIN")
        self.assertEqual(assign(upper, policy), "UNCERTAIN")
        self.assertEqual(
            report["separation_and_freeze_audit"][
                "evaluation_used_for_threshold_selection"
            ],
            False,
        )
        self.assertEqual(
            report["inputs"]["calibration_decision_report"]["sha256"],
            self.file_sha(self.calibration_report),
        )
        self.assertIn(
            "not a finding of a healthy retina",
            report["semantics"]["ROUTINE"],
        )
        tampered = json.loads(json.dumps(report))
        tampered["policy"]["otherwise"] = "ROUTINE"
        self.assertFalse(verify_integrity(tampered))

    def test_evaluation_scores_cannot_change_frozen_policy(self) -> None:
        first = self.build()
        payload = json.loads(self.evaluation_report.read_text())
        for result in payload["results"]:
            score = 1.0 - result["positive_probability"]
            margin = math.log(score / (1.0 - score))
            result["positive_logit"] = margin
            result["negative_logit"] = 0.0
            result["positive_minus_negative_logit"] = margin
            result["positive_probability"] = score
            result["negative_probability"] = 1.0 - score
        changed = self.root / "changed-evaluation-report.json"
        changed.write_text(json.dumps(payload), encoding="utf-8")
        second = self.build(evaluation_report=changed)
        self.assertEqual(first["policy"], second["policy"])
        self.assertNotEqual(
            first["evaluation"]["metrics"]["ranking"],
            second["evaluation"]["metrics"]["ranking"],
        )

    def test_rejects_partial_report_even_when_selected_count_is_forged(self) -> None:
        payload = json.loads(self.calibration_report.read_text())
        payload["results"].pop()
        payload["run"]["selected_rows"] -= 1
        payload["summary"]["samples"] -= 1
        partial = self.root / "partial.json"
        partial.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CalibrationInputError, "full manifest"):
            self.build(calibration_report=partial)

    def test_rejects_wrong_task_and_patient_overlap(self) -> None:
        payload = json.loads(self.calibration_report.read_text())
        payload["run"]["task"] = "quality"
        wrong_task = self.root / "wrong-task.json"
        wrong_task.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CalibrationInputError, "task must be"):
            self.build(calibration_report=wrong_task)

        evaluation_rows = [dict(row) for row in self.evaluation_rows]
        evaluation_rows[0]["patient_id"] = self.calibration_rows[0]["patient_id"]
        # Keep every image-to-patient relationship internally consistent by
        # rebuilding both the manifest and score report.
        overlap_manifest = self.root / "overlap.csv"
        with overlap_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(evaluation_rows)
        overlap_report = self.make_decision_report(
            "eval", overlap_manifest, evaluation_rows
        )
        with self.assertRaisesRegex(CalibrationInputError, "patient leakage"):
            self.build(
                evaluation_manifest=overlap_manifest,
                evaluation_report=overlap_report,
            )

    def test_legacy_report_without_expected_split_is_hash_verified(self) -> None:
        legacy = self.make_decision_report(
            "calibration",
            self.calibration_manifest,
            self.calibration_rows,
            include_split=False,
        )
        report = self.build(calibration_report=legacy)
        self.assertEqual(
            report["inputs"]["calibration_manifest"]["split_provenance"],
            "legacy_report_inferred_and_verified_from_full_hash_bound_manifest",
        )

    def test_rejects_manifest_sha_and_truth_tampering(self) -> None:
        payload = json.loads(self.calibration_report.read_text())
        payload["run"]["manifest_sha256"] = "0" * 64
        bad_sha = self.root / "bad-sha.json"
        bad_sha.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CalibrationInputError, "SHA mismatch"):
            self.build(calibration_report=bad_sha)

        payload = json.loads(self.calibration_report.read_text())
        payload["results"][0]["truth"] = "PRIORITY"
        bad_truth = self.root / "bad-truth.json"
        bad_truth.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CalibrationInputError, "truth does not match"):
            self.build(calibration_report=bad_truth)


if __name__ == "__main__":
    unittest.main()
