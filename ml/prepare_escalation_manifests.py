#!/usr/bin/env python3
"""Prepare patient-disjoint DeepDRiD manifests for review-priority research.

This derives a deliberately non-diagnostic binary training target from the
regular-fundus DeepDRiD eye grades:

* grades 0-1 -> ROUTINE review priority
* grades 2-4 -> PRIORITY review priority

``UNCERTAIN`` is not invented as a label.  It is reserved for selective-model
abstention at inference time.  Official training patients are split into
train, validation, and calibration cohorts.  Official validation patients are
written to a separately named evaluation manifest and are never used for
model or threshold selection.  The script refuses test, MSHF, and UWF paths.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_REQUIRED_COLUMNS = {
    "patient_id",
    "image_id",
    "Overall quality",
    "left_eye_DR_Level",
    "right_eye_DR_Level",
    "patient_DR_Level",
}
BASE_REQUIRED_COLUMNS = {
    "split",
    "patient_id",
    "image_id",
    "image_path",
    "overall_quality",
    "source_split",
}
OUTPUT_COLUMNS = (
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
)


@dataclass(frozen=True)
class EscalationRow:
    split: str
    patient_id: str
    image_id: str
    image_path: str
    dr_grade: int
    escalation_label: str
    overall_quality: int
    source_split: str
    grade_source_field: str
    filename_side_matches_grade_field: bool


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(paths: list[Path]) -> None:
    combined = " ".join(str(path).lower() for path in paths)
    if any(forbidden in combined for forbidden in ("test", "mshf", "widefield")):
        raise ValueError("escalation preparation refuses test, MSHF, and UWF sources")


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{resolved} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{resolved} contains no rows")
    return rows


def grade_from_populated_eye_field(row: dict[str, str]) -> tuple[int, str, bool]:
    """Return grade from the one authoritative non-empty eye-grade field."""

    populated = [
        (field, row[field].strip())
        for field in ("left_eye_DR_Level", "right_eye_DR_Level")
        if row[field].strip()
    ]
    if len(populated) != 1:
        raise ValueError(
            f"{row.get('image_id')}: expected exactly one populated eye-grade field"
        )
    field, raw_grade = populated[0]
    if raw_grade not in {"0", "1", "2", "3", "4"}:
        raise ValueError(f"{row.get('image_id')}: invalid DR grade {raw_grade!r}")
    suffix = row["image_id"].rsplit("_", 1)[-1]
    if not suffix or suffix[0] not in {"l", "r"}:
        raise ValueError(f"{row['image_id']}: cannot audit filename eye side")
    expected_field = (
        "left_eye_DR_Level" if suffix[0] == "l" else "right_eye_DR_Level"
    )
    return int(raw_grade), field, field == expected_field


def join_source(
    raw_csv: Path,
    base_manifest: Path,
    *,
    expected_base_split: str,
    output_split: str,
) -> list[EscalationRow]:
    raw_rows = _read_csv(raw_csv, RAW_REQUIRED_COLUMNS)
    base_rows = _read_csv(base_manifest, BASE_REQUIRED_COLUMNS)
    raw_by_image = {row["image_id"]: row for row in raw_rows}
    if len(raw_by_image) != len(raw_rows):
        raise ValueError(f"{raw_csv}: duplicate image IDs")
    if {row["image_id"] for row in base_rows} != set(raw_by_image):
        raise ValueError("raw label CSV and base manifest image sets differ")

    output: list[EscalationRow] = []
    for base in base_rows:
        if base["split"] != expected_base_split:
            raise ValueError(
                f"{base_manifest}: expected split={expected_base_split!r}, "
                f"got {base['split']!r}"
            )
        raw = raw_by_image[base["image_id"]]
        if raw["patient_id"] != base["patient_id"]:
            raise ValueError(f"{base['image_id']}: patient ID mismatch")
        if raw["Overall quality"] != base["overall_quality"]:
            raise ValueError(f"{base['image_id']}: quality label mismatch")
        grade, grade_field, side_matches = grade_from_populated_eye_field(raw)
        patient_grade = raw["patient_DR_Level"].strip()
        if patient_grade not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"{base['image_id']}: invalid patient grade")
        image_path = resolve(Path(base["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        output.append(
            EscalationRow(
                split=output_split,
                patient_id=base["patient_id"],
                image_id=base["image_id"],
                image_path=base["image_path"],
                dr_grade=grade,
                escalation_label="PRIORITY" if grade >= 2 else "ROUTINE",
                overall_quality=int(base["overall_quality"]),
                source_split=base["source_split"],
                grade_source_field=grade_field,
                filename_side_matches_grade_field=side_matches,
            )
        )

    patients: dict[str, list[EscalationRow]] = {}
    patient_grade_by_id: dict[str, int] = {}
    for row, raw in zip(output, (raw_by_image[item["image_id"]] for item in base_rows), strict=True):
        patients.setdefault(row.patient_id, []).append(row)
        patient_grade_by_id.setdefault(row.patient_id, int(raw["patient_DR_Level"]))
        if patient_grade_by_id[row.patient_id] != int(raw["patient_DR_Level"]):
            raise ValueError(f"{row.patient_id}: inconsistent patient grade")
    for patient_id, patient_rows in patients.items():
        if len(patient_rows) != 4:
            raise ValueError(f"{patient_id}: expected four images")
        if max(row.dr_grade for row in patient_rows) != patient_grade_by_id[patient_id]:
            raise ValueError(f"{patient_id}: eye grades do not match patient maximum")
    return output


def patient_stratified_partition(
    rows: list[EscalationRow],
    *,
    validation_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> dict[str, str]:
    patients: dict[str, list[EscalationRow]] = {}
    for row in rows:
        patients.setdefault(row.patient_id, []).append(row)
    strata: dict[int, list[str]] = {}
    for patient_id, patient_rows in patients.items():
        strata.setdefault(max(row.dr_grade for row in patient_rows), []).append(patient_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for patient_ids in strata.values():
        patient_ids.sort(key=int)
        rng.shuffle(patient_ids)
        validation_count = round(len(patient_ids) * validation_fraction)
        calibration_count = round(len(patient_ids) * calibration_fraction)
        if min(validation_count, calibration_count) <= 0:
            raise ValueError("each grade stratum needs validation and calibration patients")
        if validation_count + calibration_count >= len(patient_ids):
            raise ValueError("partition fractions leave no training patients")
        for patient_id in patient_ids[:validation_count]:
            assignment[patient_id] = "val"
        for patient_id in patient_ids[
            validation_count : validation_count + calibration_count
        ]:
            assignment[patient_id] = "calibration"
        for patient_id in patient_ids[validation_count + calibration_count :]:
            assignment[patient_id] = "train"
    if set(assignment) != set(patients):
        raise AssertionError("not every training patient was assigned")
    return assignment


def with_split(row: EscalationRow, split: str) -> EscalationRow:
    values = asdict(row)
    values["split"] = split
    return EscalationRow(**values)


def write_manifest(path: Path, rows: list[EscalationRow]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["filename_side_matches_grade_field"] = str(
                row.filename_side_matches_grade_field
            ).lower()
            writer.writerow(values)


def summarize(rows: list[EscalationRow]) -> dict[str, Any]:
    patients = {row.patient_id for row in rows}
    return {
        "images": len(rows),
        "patients": len(patients),
        "escalation_labels": dict(sorted(Counter(row.escalation_label for row in rows).items())),
        "dr_grades": {
            str(key): value for key, value in sorted(Counter(row.dr_grade for row in rows).items())
        },
        "overall_quality": {
            str(key): value
            for key, value in sorted(Counter(row.overall_quality for row in rows).items())
        },
        "filename_side_grade_field_mismatches": sum(
            not row.filename_side_matches_grade_field for row in rows
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-train-csv",
        type=Path,
        default=Path(
            "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-training/"
            "regular-fundus-training.csv"
        ),
    )
    parser.add_argument(
        "--raw-eval-csv",
        type=Path,
        default=Path(
            "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-validation/"
            "regular-fundus-validation.csv"
        ),
    )
    parser.add_argument("--base-train-manifest", type=Path, default=Path("data/manifests/train.csv"))
    parser.add_argument("--base-eval-manifest", type=Path, default=Path("data/manifests/val.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/escalation-manifests"))
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sources = [
        args.raw_train_csv,
        args.raw_eval_csv,
        args.base_train_manifest,
        args.base_eval_manifest,
    ]
    validate_sources(sources)
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be between zero and 0.5")
    if not 0 < args.calibration_fraction < 0.6:
        raise ValueError("calibration fraction must be between zero and 0.6")
    if args.validation_fraction + args.calibration_fraction >= 0.8:
        raise ValueError("fractions leave too few training patients")

    official_train = join_source(
        args.raw_train_csv,
        args.base_train_manifest,
        expected_base_split="train",
        output_split="official-train",
    )
    official_eval = join_source(
        args.raw_eval_csv,
        args.base_eval_manifest,
        expected_base_split="val",
        output_split="eval",
    )
    assignment = patient_stratified_partition(
        official_train,
        validation_fraction=args.validation_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    partitions = {
        split: [with_split(row, split) for row in official_train if assignment[row.patient_id] == split]
        for split in ("train", "val", "calibration")
    }
    partitions["eval"] = official_eval
    patient_sets = {
        split: {row.patient_id for row in rows} for split, rows in partitions.items()
    }
    overlaps = {
        f"{left}_{right}": len(patient_sets[left] & patient_sets[right])
        for index, left in enumerate(partitions)
        for right in list(partitions)[index + 1 :]
    }
    if any(overlaps.values()):
        raise AssertionError(f"patient leakage across escalation partitions: {overlaps}")

    output_dir = resolve(args.output_dir)
    paths: dict[str, Path] = {}
    for split, rows in partitions.items():
        paths[split] = output_dir / f"{split}.csv"
        write_manifest(paths[split], rows)

    license_path = resolve(Path("data/raw/deepdrid-v1.1/LICENSE"))
    upstream_readme = resolve(Path("data/raw/deepdrid-v1.1/README.md"))
    summary = {
        "schema_version": 1,
        "purpose": "non-diagnostic retinal review-priority research",
        "label_semantics": {
            "ROUTINE": "dataset eye grade 0 or 1; lower review priority, not disease exclusion",
            "PRIORITY": "dataset eye grade 2, 3, or 4; higher review priority, not a diagnosis",
            "UNCERTAIN": "reserved for model abstention; never used as a training truth label",
        },
        "grade_derivation": (
            "the one non-empty left_eye_DR_Level/right_eye_DR_Level field; filename side is audit-only"
        ),
        "split_policy": {
            "seed": args.seed,
            "validation_fraction_within_official_training": args.validation_fraction,
            "calibration_fraction_within_official_training": args.calibration_fraction,
            "stratification": "patient maximum eye grade",
            "official_validation_role": "evaluation only; never model or threshold selection",
        },
        "license": {
            "identifier": "CC BY-SA 4.0",
            "license_file": str(license_path.relative_to(PROJECT_ROOT)),
            "license_sha256": sha256_file(license_path),
            "upstream_readme": str(upstream_readme.relative_to(PROJECT_ROOT)),
            "upstream_readme_sha256": sha256_file(upstream_readme),
            "citation_doi": "10.1016/j.patter.2022.100512",
            "obligations_note": "preserve attribution, indicate modifications, and share adapted dataset material alike",
        },
        "sources": {
            str(path): sha256_file(resolve(path)) for path in sources
        },
        "partitions": {
            split: {
                **summarize(rows),
                "manifest": str(paths[split].relative_to(PROJECT_ROOT)),
                "manifest_sha256": sha256_file(paths[split]),
            }
            for split, rows in partitions.items()
        },
        "patient_overlap": overlaps,
        "test_used": False,
        "mshf_used": False,
        "uwf_used": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
