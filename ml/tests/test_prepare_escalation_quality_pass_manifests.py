import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from prepare_escalation_quality_pass_manifests import (  # noqa: E402
    PARTITIONS,
    build_quality_pass_manifests,
    sha256_file,
    summarize,
)


FIELDNAMES = [
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


class QualityPassManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_dir = self.root / "source"
        self.output_dir = self.root / "derived"
        self.source_dir.mkdir()
        self.rows_by_split: dict[str, list[dict[str, str]]] = {}

        for split_index, split in enumerate(PARTITIONS):
            rows: list[dict[str, str]] = []
            for quality in ("1", "0"):
                grade = "2" if (split_index + int(quality)) % 2 else "1"
                image_id = f"{split}-{quality}"
                image = self.root / f"{image_id}.jpg"
                image.write_bytes(f"image-{image_id}".encode())
                rows.append(
                    {
                        "split": split,
                        "patient_id": f"patient-{split}-{quality}",
                        "image_id": image_id,
                        "image_path": str(image),
                        "dr_grade": grade,
                        "escalation_label": (
                            "PRIORITY" if int(grade) >= 2 else "ROUTINE"
                        ),
                        "overall_quality": quality,
                        "source_split": "regular-fundus-training",
                        "grade_source_field": "left_eye_DR_Level",
                        "filename_side_matches_grade_field": "true",
                    }
                )
            self.rows_by_split[split] = rows
        self.write_sources_and_summary()

    def write_manifest(self, split: str) -> None:
        path = self.source_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FIELDNAMES, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(self.rows_by_split[split])

    def write_sources_and_summary(self) -> None:
        partitions = {}
        for split in PARTITIONS:
            self.write_manifest(split)
            path = self.source_dir / f"{split}.csv"
            partitions[split] = {
                **summarize(self.rows_by_split[split]),
                "manifest": str(path),
                "manifest_sha256": sha256_file(path),
            }
        self.summary_path = self.source_dir / "summary.json"
        self.summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sources": {"upstream.csv": "a" * 64},
                    "license": {"identifier": "CC BY-SA 4.0"},
                    "partitions": partitions,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def build(self) -> dict:
        return build_quality_pass_manifests(
            self.source_dir,
            self.summary_path,
            self.output_dir,
        )

    def test_filter_is_deterministic_and_preserves_rows_and_lineage(self) -> None:
        source_summary_hash = sha256_file(self.summary_path)
        summary = self.build()
        first_bytes = {
            path.name: path.read_bytes()
            for path in sorted(self.output_dir.iterdir())
        }
        second_summary = self.build()
        second_bytes = {
            path.name: path.read_bytes()
            for path in sorted(self.output_dir.iterdir())
        }

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(summary, second_summary)
        self.assertEqual(
            summary["source_summary"]["sha256"], source_summary_hash
        )
        self.assertEqual(summary["global_counts"]["source_images"], 8)
        self.assertEqual(summary["global_counts"]["derived_images"], 4)
        self.assertEqual(summary["global_counts"]["excluded_images"], 4)
        self.assertTrue(
            all(
                value == 0
                for audit in (
                    summary["source_overlap_audit"],
                    summary["derived_overlap_audit"],
                )
                for values in audit.values()
                for value in values.values()
            )
        )

        for split in PARTITIONS:
            output_path = self.output_dir / f"{split}.csv"
            with output_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, FIELDNAMES)
            self.assertEqual(rows, [self.rows_by_split[split][0]])
            self.assertEqual(rows[0]["split"], split)
            self.assertEqual(rows[0]["overall_quality"], "1")
            partition = summary["partitions"][split]
            self.assertEqual(partition["source"]["images"], 2)
            self.assertEqual(partition["derived"]["images"], 1)
            self.assertEqual(partition["excluded_images"], 1)
            self.assertEqual(
                partition["derived"]["manifest_sha256"],
                sha256_file(output_path),
            )

    def test_wrong_escalation_label_is_refused(self) -> None:
        self.rows_by_split["train"][0]["escalation_label"] = "ROUTINE"
        self.write_sources_and_summary()
        with self.assertRaisesRegex(ValueError, "requires escalation_label"):
            self.build()

    def test_invalid_quality_value_is_refused(self) -> None:
        self.rows_by_split["train"][0]["overall_quality"] = "2"
        self.write_sources_and_summary()
        with self.assertRaisesRegex(ValueError, "invalid overall_quality"):
            self.build()

    def test_missing_identifier_is_refused(self) -> None:
        self.rows_by_split["train"][0]["image_id"] = ""
        self.write_sources_and_summary()
        with self.assertRaisesRegex(ValueError, "missing or blank image_id"):
            self.build()

    def test_missing_image_file_is_refused(self) -> None:
        self.rows_by_split["train"][0]["image_path"] = str(
            self.root / "not-present.jpg"
        )
        self.write_sources_and_summary()
        with self.assertRaisesRegex(ValueError, "missing image file"):
            self.build()

    def test_duplicate_image_id_across_partitions_is_refused(self) -> None:
        self.rows_by_split["val"][0]["image_id"] = self.rows_by_split["train"][0][
            "image_id"
        ]
        self.write_sources_and_summary()
        with self.assertRaisesRegex(ValueError, "duplicate image_id"):
            self.build()

    def test_parent_manifest_hash_mismatch_is_refused(self) -> None:
        payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        payload["partitions"]["train"]["manifest_sha256"] = "0" * 64
        self.summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "source summary hash mismatch"):
            self.build()


if __name__ == "__main__":
    unittest.main()
