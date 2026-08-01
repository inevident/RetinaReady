from pathlib import Path
import sys
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from experiment_spatial_color_challenger import (  # noqa: E402
    build_feature_variants,
    color_space_statistics,
    make_recommendation,
    validate_experiment_manifests,
)


class SpatialColorChallengerTests(unittest.TestCase):
    def test_color_statistics_are_finite_and_have_declared_width(self) -> None:
        import numpy as np

        image = np.zeros((8, 8, 3), dtype="float32")
        image[2:6, 2:6, 0] = 0.8
        image[2:6, 2:6, 1] = 0.3
        image[2:6, 2:6, 2] = 0.1
        features = color_space_statistics(image, np)
        self.assertEqual(features.shape, (45,))
        self.assertTrue(np.isfinite(features).all())

    def test_feature_variants_have_expected_dimensions(self) -> None:
        import numpy as np

        bundle = {
            "global_features": np.zeros((3, 4), dtype="float32"),
            "spatial_2x2": np.ones((3, 16), dtype="float32"),
            "color_stats": np.full((3, 45), 2.0, dtype="float32"),
        }
        variants = build_feature_variants(bundle, np)
        self.assertEqual(variants["global-baseline"].shape, (3, 4))
        self.assertEqual(variants["global-spatial-2x2"].shape, (3, 20))
        self.assertEqual(
            variants["global-spatial-2x2-color-stats"].shape, (3, 65)
        )

    def test_only_internal_train_and_validation_manifests_are_allowed(self) -> None:
        validate_experiment_manifests(
            Path("data/manifests/train.csv"), Path("data/manifests/val.csv")
        )
        with self.assertRaises(ValueError):
            validate_experiment_manifests(
                Path("data/manifests/train.csv"), Path("data/manifests/test.csv")
            )
        with self.assertRaises(ValueError):
            validate_experiment_manifests(
                Path("data/manifests/train.csv"), Path("data/mshf/val.csv")
            )

    def test_promotion_rule_rejects_factor_only_tradeoff(self) -> None:
        recommendation = make_recommendation(
            {
                "global-baseline": {},
                "global-spatial-2x2": {
                    "roc_auc_delta_vs_global": -0.01,
                    "balanced_accuracy_delta_vs_global": -0.01,
                    "mean_factor_mae_delta_vs_global": -0.002,
                    "selective_coverage_delta_vs_global": -0.02,
                },
            }
        )
        self.assertEqual(
            recommendation["decision"],
            "retain-global-baseline; do-not-promote-challenger",
        )


if __name__ == "__main__":
    unittest.main()
