#!/usr/bin/env python3
"""Build leakage-safe DeepDRiD image-quality manifests.

This script uses only the Python standard library. It preserves the official
DeepDRiD train/validation/evaluation patient boundaries and intentionally
omits the disease-grading labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_DIR / "data" / "raw" / "deepdrid-v1.1"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "manifests"

MANIFEST_FIELDS = [
    "split",
    "patient_id",
    "image_id",
    "image_path",
    "overall_quality",
    "quality_label",
    "artifact",
    "clarity",
    "field_definition",
    "source_split",
]

QUALITY_LABELS = {0: "RETAKE", 1: "READY"}
ALLOWED_ARTIFACT = {0, 1, 4, 6, 8, 10}
ALLOWED_CLARITY = {1, 4, 6, 8, 10}
ALLOWED_FIELD_DEFINITION = {1, 4, 6, 8, 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Extracted DeepDRiD v1.1 root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Manifest output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Build labels without requiring each referenced image to exist.",
    )
    return parser.parse_args()


def normalized_int(value: object, field: str, image_id: str) -> int:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{image_id}: invalid {field} value {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"{image_id}: non-integral {field} value {value!r}")
    return int(number)


def project_path(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        return [{key.strip(): value.strip() for key, value in row.items()} for row in reader]


def excel_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {reference!r}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + (ord(character) - ord("A") + 1)
    return result - 1


def read_first_xlsx_sheet(path: Path) -> list[dict[str, object]]:
    """Read a simple first-sheet XLSX table without a third-party dependency."""
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns = {"m": main_ns}
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared_strings.append("".join(node.text or "" for node in item.findall(".//m:t", ns)))

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        raw_rows: list[list[object]] = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            values: list[object] = []
            for cell in row.findall("m:c", ns):
                index = excel_column_index(cell.attrib["r"])
                while len(values) <= index:
                    values.append("")

                kind = cell.attrib.get("t")
                value_node = cell.find("m:v", ns)
                if kind == "inlineStr":
                    value: object = "".join(
                        node.text or "" for node in cell.findall(".//m:is/m:t", ns)
                    )
                elif value_node is None:
                    value = ""
                elif kind == "s":
                    value = shared_strings[int(value_node.text or "0")]
                else:
                    value = value_node.text or ""
                values[index] = value
            raw_rows.append(values)

    if not raw_rows:
        raise ValueError(f"No rows found in {path}")
    headers = [str(value).strip() for value in raw_rows[0]]
    return [
        {headers[index]: row[index] if index < len(row) else "" for index in range(len(headers))}
        for row in raw_rows[1:]
        if any(str(value).strip() for value in row)
    ]


def make_quality_row(
    *,
    split: str,
    source_split: str,
    patient_id: str,
    image_id: str,
    image_path: Path,
    overall_quality: object,
    artifact: object,
    clarity: object,
    field_definition: object,
) -> dict[str, object]:
    quality = normalized_int(overall_quality, "overall_quality", image_id)
    artifact_value = normalized_int(artifact, "artifact", image_id)
    clarity_value = normalized_int(clarity, "clarity", image_id)
    field_value = normalized_int(field_definition, "field_definition", image_id)

    if quality not in QUALITY_LABELS:
        raise ValueError(f"{image_id}: unexpected overall_quality={quality}")
    if artifact_value not in ALLOWED_ARTIFACT:
        raise ValueError(f"{image_id}: unexpected artifact={artifact_value}")
    if clarity_value not in ALLOWED_CLARITY:
        raise ValueError(f"{image_id}: unexpected clarity={clarity_value}")
    if field_value not in ALLOWED_FIELD_DEFINITION:
        raise ValueError(f"{image_id}: unexpected field_definition={field_value}")

    return {
        "split": split,
        "patient_id": str(patient_id),
        "image_id": image_id,
        "image_path": project_path(image_path),
        "overall_quality": quality,
        "quality_label": QUALITY_LABELS[quality],
        "artifact": artifact_value,
        "clarity": clarity_value,
        "field_definition": field_value,
        "source_split": source_split,
    }


def load_csv_split(dataset_root: Path, split: str, folder_name: str) -> list[dict[str, object]]:
    split_dir = dataset_root / "regular_fundus_images" / folder_name
    labels_path = split_dir / f"{folder_name}.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing label file: {labels_path}")

    output: list[dict[str, object]] = []
    for source in read_csv_rows(labels_path):
        patient_id = source["patient_id"]
        image_id = source["image_id"]
        output.append(
            make_quality_row(
                split=split,
                source_split=folder_name,
                patient_id=patient_id,
                image_id=image_id,
                image_path=split_dir / "Images" / patient_id / f"{image_id}.jpg",
                overall_quality=source["Overall quality"],
                artifact=source["Artifact"],
                clarity=source["Clarity"],
                field_definition=source["Field definition"],
            )
        )
    return output


def load_test_split(dataset_root: Path) -> list[dict[str, object]]:
    folder_name = "Online-Challenge1&2-Evaluation"
    split_dir = dataset_root / "regular_fundus_images" / folder_name
    labels_path = split_dir / "Challenge2_labels.xlsx"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing released test labels: {labels_path}")

    output: list[dict[str, object]] = []
    for source in read_first_xlsx_sheet(labels_path):
        image_id = str(source["image_id"]).strip()
        patient_id = image_id.split("_", 1)[0]
        output.append(
            make_quality_row(
                split="test",
                source_split=folder_name,
                patient_id=patient_id,
                image_id=image_id,
                image_path=split_dir / "Images" / patient_id / f"{image_id}.jpg",
                overall_quality=source["Overall quality"],
                artifact=source["Artifact"],
                clarity=source["Clarity"],
                field_definition=source["Field definition"],
            )
        )
    return output


def validate_splits(
    split_rows: dict[str, list[dict[str, object]]],
    *,
    check_images: bool,
) -> dict[str, object]:
    patient_sets = {
        split: {str(row["patient_id"]) for row in rows}
        for split, rows in split_rows.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            preview = ", ".join(sorted(overlap)[:10])
            raise ValueError(f"Patient leakage between {left} and {right}: {preview}")

    all_image_ids: list[str] = [
        str(row["image_id"]) for rows in split_rows.values() for row in rows
    ]
    duplicates = [name for name, count in Counter(all_image_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate image IDs across manifests: {duplicates[:10]}")

    missing: list[str] = []
    if check_images:
        for rows in split_rows.values():
            for row in rows:
                candidate = Path(str(row["image_path"]))
                if not candidate.is_absolute():
                    candidate = PROJECT_DIR / candidate
                if not candidate.is_file():
                    missing.append(str(row["image_path"]))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest images are missing; first: {missing[0]}"
            )

    return {
        "patient_overlap": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
        "duplicate_image_ids": 0,
        "missing_images": len(missing),
    }


def count_values(rows: Iterable[dict[str, object]], field: str) -> dict[str, int]:
    counts = Counter(str(row[field]) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    split_rows = {
        "train": load_csv_split(dataset_root, "train", "regular-fundus-training"),
        "val": load_csv_split(dataset_root, "val", "regular-fundus-validation"),
        "test": load_test_split(dataset_root),
    }
    checks = validate_splits(split_rows, check_images=not args.skip_image_check)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        write_manifest(output_dir / f"{split}.csv", rows)
    write_manifest(
        output_dir / "all.csv",
        [row for split in ("train", "val", "test") for row in split_rows[split]],
    )

    summary: dict[str, object] = {
        "dataset": {
            "name": "DeepDRiD",
            "version": "v1.1",
            "doi": "10.5281/zenodo.8248825",
            "archive_bytes": 1_373_472_897,
            "archive_md5": "3379e2fd7a2dd398545a67148420a5d3",
            "license": "CC BY-SA 4.0 (upstream repository LICENSE)",
        },
        "path_policy": "image_path is relative to retina-ready when data is stored inside the project",
        "quality_mapping": {"0": "RETAKE", "1": "READY"},
        "splits": {},
        "checks": checks,
    }
    for split, rows in split_rows.items():
        summary["splits"][split] = {
            "images": len(rows),
            "patients": len({str(row["patient_id"]) for row in rows}),
            "overall_quality": count_values(rows, "overall_quality"),
            "artifact": count_values(rows, "artifact"),
            "clarity": count_values(rows, "clarity"),
            "field_definition": count_values(rows, "field_definition"),
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for split in ("train", "val", "test"):
        stats = summary["splits"][split]
        print(
            f"{split:>5}: {stats['images']:4} images, "
            f"{stats['patients']:3} patients, "
            f"quality={stats['overall_quality']}"
        )
    print(f"Wrote manifests to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
