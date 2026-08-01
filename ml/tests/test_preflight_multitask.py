import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from preflight_a100 import (  # noqa: E402
    PROJECT_ROOT,
    Report,
    check_c_compiler,
    check_dataset,
    manifest_paths_for_config,
    read_manifest,
)


class PreflightMultitaskTests(unittest.TestCase):
    def write_manifest(self, row: dict[str, str]) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", suffix=".csv", delete=False
        )
        with temporary:
            writer = csv.DictWriter(temporary, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def write_escalation_partition(
        self,
        directory: Path,
        split: str,
        index: int,
    ) -> Path:
        image = directory / f"{split}.jpg"
        image.write_bytes(f"image-{split}".encode())
        path = directory / f"{split}.csv"
        row = {
            "split": split,
            "patient_id": f"patient-{index}",
            "image_id": f"image-{index}",
            "image_path": str(image),
            "dr_grade": "2" if index % 2 else "1",
            "escalation_label": "PRIORITY" if index % 2 else "ROUTINE",
            "overall_quality": "1",
            "source_split": "fixture",
        }
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_escalation_manifest_contract_passes(self) -> None:
        path = self.write_manifest(
            {
                "split": "train",
                "patient_id": "1",
                "image_id": "1_l1",
                "image_path": "does-not-need-to-exist-for-schema-check.jpg",
                "dr_grade": "2",
                "escalation_label": "PRIORITY",
                "overall_quality": "1",
                "source_split": "regular-fundus-training",
            }
        )
        report = Report()
        rows = read_manifest(report, path, "train", task="escalation")
        self.assertEqual(len(rows), 1)
        self.assertEqual(report.failures, 0)

    def test_escalation_grade_label_mismatch_fails_closed(self) -> None:
        path = self.write_manifest(
            {
                "split": "val",
                "patient_id": "2",
                "image_id": "2_r1",
                "image_path": "unused.jpg",
                "dr_grade": "3",
                "escalation_label": "ROUTINE",
                "overall_quality": "1",
                "source_split": "regular-fundus-training",
            }
        )
        report = Report()
        self.assertEqual(
            read_manifest(report, path, "val", task="escalation"), []
        )
        self.assertEqual(report.failures, 1)

    def test_quality_manifest_remains_backward_compatible(self) -> None:
        path = self.write_manifest(
            {
                "split": "train",
                "patient_id": "3",
                "image_id": "3_l1",
                "image_path": "unused.jpg",
                "overall_quality": "0",
                "quality_label": "RETAKE",
                "artifact": "5",
                "clarity": "4",
                "field_definition": "4",
                "source_split": "regular-fundus-training",
            }
        )
        report = Report()
        rows = read_manifest(report, path, "train", task="quality")
        self.assertEqual(len(rows), 1)
        self.assertEqual(report.failures, 0)

    def test_escalation_manifest_paths_accept_quality_pass_overrides(self) -> None:
        config = {
            "task": "escalation",
            "train_manifest": "/tmp/custom-train.csv",
            "val_manifest": "/tmp/custom-val.csv",
            "calibration_manifest": "/tmp/custom-calibration.csv",
            "eval_manifest": "/tmp/custom-eval.csv",
        }
        self.assertEqual(
            manifest_paths_for_config(config),
            {
                "train": Path("/tmp/custom-train.csv"),
                "val": Path("/tmp/custom-val.csv"),
                "calibration": Path("/tmp/custom-calibration.csv"),
                "eval": Path("/tmp/custom-eval.csv"),
            },
        )

    def test_escalation_manifest_paths_keep_legacy_defaults(self) -> None:
        paths = manifest_paths_for_config(
            {
                "task": "escalation",
                "train_manifest": "train.csv",
                "val_manifest": "val.csv",
            }
        )
        self.assertEqual(
            paths["calibration"],
            PROJECT_ROOT / "data" / "escalation-manifests" / "calibration.csv",
        )
        self.assertEqual(
            paths["eval"],
            PROJECT_ROOT / "data" / "escalation-manifests" / "eval.csv",
        )

    def test_dataset_check_uses_custom_escalation_calibration_and_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = {
                split: self.write_escalation_partition(directory, split, index)
                for index, split in enumerate(
                    ("train", "val", "calibration", "eval"), start=1
                )
            }
            report = Report()
            check_dataset(
                report,
                {
                    "task": "escalation",
                    "train_manifest": str(paths["train"]),
                    "val_manifest": str(paths["val"]),
                    "calibration_manifest": str(paths["calibration"]),
                    "eval_manifest": str(paths["eval"]),
                },
                verify_image_headers=False,
            )
        self.assertEqual(report.failures, 0)
        integrity = next(
            check for check in report.checks if check.name == "dataset.integrity"
        )
        self.assertEqual(integrity.status, "PASS")
        self.assertIn("4 unique images exist", integrity.message)

    @patch("preflight_a100.command_output", return_value="gcc test-version")
    @patch("preflight_a100.shutil.which")
    def test_c_compiler_passes_when_gcc_is_available(
        self, which, _command_output
    ) -> None:
        which.side_effect = lambda candidate: (
            "/usr/bin/gcc" if candidate == "gcc" else None
        )
        report = Report()
        with patch.dict("preflight_a100.os.environ", {}, clear=True):
            check_c_compiler(report)
        self.assertEqual(report.failures, 0)
        self.assertEqual(report.checks[-1].name, "system.c_compiler")
        self.assertEqual(report.checks[-1].status, "PASS")

    @patch("preflight_a100.shutil.which", return_value=None)
    def test_c_compiler_fails_closed_when_missing(self, _which) -> None:
        report = Report()
        with patch.dict("preflight_a100.os.environ", {}, clear=True):
            check_c_compiler(report)
        self.assertEqual(report.failures, 1)
        self.assertEqual(report.checks[-1].name, "system.c_compiler")
        self.assertEqual(report.checks[-1].status, "FAIL")


if __name__ == "__main__":
    unittest.main()
