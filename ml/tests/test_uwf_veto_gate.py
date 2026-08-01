from argparse import Namespace
from pathlib import Path
import sys
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from experiment_uwf_veto_gate import (  # noqa: E402
    ModalityExample,
    calibrate_veto_threshold,
    patient_group_three_way_split,
    recommendation,
    threshold_metrics,
    validate_sources,
)


class UwfVetoGateTests(unittest.TestCase):
    def test_patient_split_keeps_modalities_from_one_patient_together(self) -> None:
        examples = []
        for patient in range(12):
            examples.append(
                ModalityExample(
                    str(patient), f"c-{patient}", "unused.jpg", "CONVENTIONAL_CFP", "train"
                )
            )
            if patient < 6:
                examples.append(
                    ModalityExample(
                        str(patient), f"u-{patient}", "unused.jpg", "UWF", "train"
                    )
                )
        fit, tuning, calibration = patient_group_three_way_split(
            examples, tuning_fraction=0.2, calibration_fraction=0.2, seed=42
        )
        patient_sets = [
            {examples[index].patient_id for index in indices}
            for indices in (fit, tuning, calibration)
        ]
        self.assertFalse(patient_sets[0] & patient_sets[1])
        self.assertFalse(patient_sets[0] & patient_sets[2])
        self.assertFalse(patient_sets[1] & patient_sets[2])

    def test_threshold_is_strict_and_veto_only(self) -> None:
        rows = [
            {
                "patient_id": str(index),
                "truth_modality": "CONVENTIONAL_CFP",
                "uwf_score": score,
            }
            for index, score in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        ] + [
            {"patient_id": "u", "truth_modality": "UWF", "uwf_score": 0.9}
        ]
        policy = calibrate_veto_threshold(
            rows, false_uwf_patient_risk=0.5, delta=0.5
        )
        self.assertEqual(policy["action_if_score_strictly_greater_than_threshold"], "FORCE_LIMITED")
        self.assertFalse(policy["can_promote"])
        equal_row = [
            {
                "patient_id": "c",
                "truth_modality": "CONVENTIONAL_CFP",
                "uwf_score": policy["threshold"],
            },
            {"patient_id": "u", "truth_modality": "UWF", "uwf_score": 1.0},
        ]
        metrics = threshold_metrics(equal_row, policy["threshold"], delta=0.5)
        self.assertEqual(metrics["false_uwf_count_on_conventional_images"], 0)

    def test_metrics_and_recommendation_use_false_uwf_and_recall(self) -> None:
        rows = [
            {"patient_id": "c", "truth_modality": "CONVENTIONAL_CFP", "uwf_score": 0.1},
            {"patient_id": "u", "truth_modality": "UWF", "uwf_score": 0.9},
        ]
        metrics = threshold_metrics(rows, 0.5, delta=0.5)
        self.assertEqual(metrics["false_uwf_rate_on_conventional_images"], 0.0)
        self.assertEqual(metrics["uwf_recall"], 1.0)
        decision = recommendation(
            metrics,
            {"clean_modality_separation_rate": 1.0},
        )
        self.assertEqual(decision["runtime_integration"], "do-not-integrate-yet")

    def test_test_and_mshf_sources_are_refused(self) -> None:
        base = Namespace(
            regular_train_manifest=Path("data/manifests/train.csv"),
            regular_val_manifest=Path("data/manifests/val.csv"),
            uwf_train_csv=Path(
                "data/raw/deepdrid-v1.1/ultra-widefield_images/ultra-widefield-training/ultra-widefield-training.csv"
            ),
            uwf_val_csv=Path(
                "data/raw/deepdrid-v1.1/ultra-widefield_images/ultra-widefield-validation/ultra-widefield-validation.csv"
            ),
        )
        validate_sources(base)
        base.regular_val_manifest = Path("data/manifests/test.csv")
        with self.assertRaises(ValueError):
            validate_sources(base)
        base.regular_val_manifest = Path("data/manifests/val.csv")
        base.uwf_val_csv = Path("data/mshf/val.csv")
        with self.assertRaises(ValueError):
            validate_sources(base)


if __name__ == "__main__":
    unittest.main()
