import math
from pathlib import Path
import sys
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from calibrate_selective_policy import (  # noqa: E402
    assign,
    calibration_scores,
    calibrate_thresholds,
    evaluate_patient_events,
    exact_upper_bound,
    maximum_certified_errors,
)
from train_quality_specialist import (  # noqa: E402
    Example,
    auc,
    patient_grouped_split,
    patient_grouped_three_way_split,
)


class SelectivePolicyTests(unittest.TestCase):
    def test_exact_bound_reproduces_current_calibration_contract(self) -> None:
        self.assertEqual(maximum_certified_errors(218, 0.05, 0.05), 5)
        self.assertLessEqual(exact_upper_bound(5, 218, 0.05), 0.05)
        self.assertGreater(exact_upper_bound(6, 218, 0.05), 0.05)

    def test_policy_uses_strict_boundaries_and_abstains_on_equality(self) -> None:
        rows = [
            {"image_id": f"bad-{index}", "truth": "RETAKE", "ready_probability": score}
            for index, score in enumerate((0.9, 0.8, 0.7, 0.6, 0.5, 0.4))
        ] + [
            {"image_id": f"good-{index}", "truth": "READY", "ready_probability": score}
            for index, score in enumerate((0.1, 0.2, 0.3, 0.4, 0.8, 0.9))
        ]
        policy = calibrate_thresholds(
            rows,
            false_ready_risk=0.5,
            false_retake_risk=0.5,
            delta=0.5,
        )
        high = policy["ready_threshold_strictly_greater_than"]
        low = policy["retake_threshold_strictly_less_than"]
        self.assertEqual(assign(high, policy), "LIMITED")
        self.assertEqual(assign(low, policy), "LIMITED")
        self.assertEqual(assign(math.nextafter(high, math.inf), policy), "READY")
        self.assertEqual(assign(math.nextafter(low, -math.inf), policy), "RETAKE")

    def test_patient_calibration_uses_one_worst_case_score_per_patient(self) -> None:
        rows = [
            {"patient_id": "a", "image_id": "a1", "truth": "RETAKE", "decision_score": 0.8},
            {"patient_id": "a", "image_id": "a2", "truth": "RETAKE", "decision_score": 0.3},
            {"patient_id": "b", "image_id": "b1", "truth": "RETAKE", "decision_score": 0.6},
            {"patient_id": "c", "image_id": "c1", "truth": "READY", "decision_score": 0.2},
            {"patient_id": "c", "image_id": "c2", "truth": "READY", "decision_score": 0.7},
            {"patient_id": "d", "image_id": "d1", "truth": "READY", "decision_score": 0.4},
        ]
        retake_scores, ready_scores = calibration_scores(rows, unit="patient")
        self.assertEqual(retake_scores, [0.8, 0.6])
        self.assertEqual(ready_scores, [0.2, 0.4])

    def test_patient_event_evaluation_counts_each_patient_once(self) -> None:
        policy = {
            "ready_threshold_strictly_greater_than": 0.7,
            "retake_threshold_strictly_less_than": 0.3,
        }
        rows = [
            {"patient_id": "a", "image_id": "a1", "truth": "RETAKE", "decision_score": 0.9},
            {"patient_id": "a", "image_id": "a2", "truth": "RETAKE", "decision_score": 0.8},
            {"patient_id": "b", "image_id": "b1", "truth": "READY", "decision_score": 0.1},
            {"patient_id": "b", "image_id": "b2", "truth": "READY", "decision_score": 0.2},
        ]
        metrics = evaluate_patient_events(rows, policy)
        self.assertEqual(metrics["false_ready_patient_count"], 1)
        self.assertEqual(metrics["false_retake_patient_count"], 1)


class SpecialistUtilityTests(unittest.TestCase):
    @staticmethod
    def example(patient: str, image: str, label: str) -> Example:
        return Example(patient, image, "unused.jpg", label, 0, 10, 10)

    def test_auc_is_tie_aware(self) -> None:
        self.assertEqual(auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(auc([0.5, 0.5], [1, 0]), 0.5)

    def test_patient_split_never_leaks_images_from_one_patient(self) -> None:
        examples = []
        for patient in range(30):
            labels = (
                ("RETAKE",) * 4
                if patient % 3 == 0
                else ("READY",) * 4
                if patient % 3 == 1
                else ("READY", "READY", "RETAKE", "RETAKE")
            )
            for image, label in enumerate(labels):
                examples.append(self.example(str(patient), f"{patient}-{image}", label))
        fit, calibration = patient_grouped_split(
            examples, calibration_fraction=0.2, seed=42
        )
        fit_patients = {examples[index].patient_id for index in fit}
        calibration_patients = {examples[index].patient_id for index in calibration}
        self.assertFalse(fit_patients & calibration_patients)
        self.assertEqual(len(fit) + len(calibration), len(examples))
        self.assertEqual(len(calibration_patients), 6)

    def test_three_way_patient_split_has_no_overlap(self) -> None:
        examples = [
            self.example(str(patient), f"{patient}-{image}", "READY" if patient % 2 else "RETAKE")
            for patient in range(30)
            for image in range(4)
        ]
        fit, tuning, calibration = patient_grouped_three_way_split(
            examples, tuning_fraction=0.2, calibration_fraction=0.3, seed=7
        )
        patient_sets = [
            {examples[index].patient_id for index in indices}
            for indices in (fit, tuning, calibration)
        ]
        self.assertFalse(patient_sets[0] & patient_sets[1])
        self.assertFalse(patient_sets[0] & patient_sets[2])
        self.assertFalse(patient_sets[1] & patient_sets[2])
        self.assertEqual(len(fit) + len(tuning) + len(calibration), len(examples))


if __name__ == "__main__":
    unittest.main()
