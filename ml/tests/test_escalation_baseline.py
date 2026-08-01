from pathlib import Path
import sys
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from experiment_escalation_baseline import (  # noqa: E402
    EscalationExample,
    assert_patient_disjoint,
    assign,
    calibrate_thresholds,
    selective_metrics,
    validate_sources as validate_experiment_sources,
)
from prepare_escalation_manifests import (  # noqa: E402
    EscalationRow,
    grade_from_populated_eye_field,
    patient_stratified_partition,
    validate_sources as validate_preparation_sources,
)


class EscalationManifestTests(unittest.TestCase):
    def test_grade_uses_populated_field_even_when_filename_side_disagrees(self) -> None:
        row = {
            "image_id": "56_r3",
            "left_eye_DR_Level": "2",
            "right_eye_DR_Level": "",
        }
        grade, field, matches = grade_from_populated_eye_field(row)
        self.assertEqual(grade, 2)
        self.assertEqual(field, "left_eye_DR_Level")
        self.assertFalse(matches)

    def test_patient_partition_is_disjoint_and_stratified(self) -> None:
        rows = []
        for grade in range(5):
            for patient in range(10):
                patient_id = str(grade * 100 + patient)
                for image in range(4):
                    rows.append(
                        EscalationRow(
                            split="official-train",
                            patient_id=patient_id,
                            image_id=f"{patient_id}_{image}",
                            image_path="unused.jpg",
                            dr_grade=grade,
                            escalation_label="PRIORITY" if grade >= 2 else "ROUTINE",
                            overall_quality=1,
                            source_split="regular-fundus-training",
                            grade_source_field="left_eye_DR_Level",
                            filename_side_matches_grade_field=True,
                        )
                    )
        assignment = patient_stratified_partition(
            rows, validation_fraction=0.2, calibration_fraction=0.4, seed=42
        )
        self.assertEqual(sum(value == "train" for value in assignment.values()), 20)
        self.assertEqual(sum(value == "val" for value in assignment.values()), 10)
        self.assertEqual(
            sum(value == "calibration" for value in assignment.values()), 20
        )

    def test_forbidden_sources_are_refused(self) -> None:
        validate_preparation_sources([Path("data/manifests/train.csv")])
        validate_experiment_sources([Path("data/escalation-manifests/eval.csv")])
        for path in (
            Path("data/manifests/test.csv"),
            Path("data/mshf/eval.csv"),
            Path("data/ultra-widefield-validation.csv"),
        ):
            with self.assertRaises(ValueError):
                validate_preparation_sources([path])
            with self.assertRaises(ValueError):
                validate_experiment_sources([path])


class EscalationPolicyTests(unittest.TestCase):
    def test_calibration_is_patient_grouped_strict_and_abstaining(self) -> None:
        rows = []
        for index in range(60):
            rows.append(
                {
                    "patient_id": f"p-{index}",
                    "truth_review_priority": "PRIORITY",
                    "review_priority_score": 0.8 + index / 10000,
                }
            )
            rows.append(
                {
                    "patient_id": f"r-{index}",
                    "truth_review_priority": "ROUTINE",
                    "review_priority_score": 0.2 - index / 10000,
                }
            )
        policy = calibrate_thresholds(
            rows,
            false_routine_risk=0.05,
            false_priority_risk=0.05,
            delta=0.05,
        )
        lower = policy["routine_if_score_strictly_less_than"]
        upper = policy["priority_if_score_strictly_greater_than"]
        self.assertLess(lower, upper)
        self.assertEqual(assign(lower, policy), "UNCERTAIN")
        self.assertEqual(assign(upper, policy), "UNCERTAIN")
        self.assertEqual(assign(lower - 0.01, policy), "ROUTINE")
        self.assertEqual(assign(upper + 0.01, policy), "PRIORITY")
        self.assertEqual(
            policy["false_routine"]["observed_strict_threshold_errors"], 0
        )
        self.assertEqual(
            policy["false_priority"]["observed_strict_threshold_errors"], 0
        )

    def test_uncertain_decisions_remain_in_metric_denominator(self) -> None:
        policy = {
            "routine_if_score_strictly_less_than": 0.25,
            "priority_if_score_strictly_greater_than": 0.75,
        }
        rows = [
            {
                "truth_review_priority": "ROUTINE",
                "review_priority_score": 0.1,
            },
            {
                "truth_review_priority": "ROUTINE",
                "review_priority_score": 0.5,
            },
            {
                "truth_review_priority": "PRIORITY",
                "review_priority_score": 0.5,
            },
            {
                "truth_review_priority": "PRIORITY",
                "review_priority_score": 0.9,
            },
        ]
        metrics = selective_metrics(rows, policy, delta=0.5)
        self.assertEqual(metrics["images"], 4)
        self.assertEqual(metrics["decision_counts"]["UNCERTAIN"], 2)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["priority_recall"], 0.5)
        self.assertEqual(metrics["routine_recall"], 0.5)

    def test_patient_overlap_is_rejected(self) -> None:
        def example(split: str, patient_id: str) -> EscalationExample:
            return EscalationExample(
                split=split,
                patient_id=patient_id,
                image_id=f"{patient_id}-{split}",
                image_path="unused.jpg",
                dr_grade=0,
                escalation_label="ROUTINE",
                overall_quality=1,
                source_split="regular-fundus-training",
                grade_source_field="left_eye_DR_Level",
                filename_side_matches_grade_field=True,
            )

        with self.assertRaises(ValueError):
            assert_patient_disjoint(
                {"train": [example("train", "1")], "val": [example("val", "1")]}
            )


if __name__ == "__main__":
    unittest.main()
