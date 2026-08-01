#!/usr/bin/env python3
"""Derive clinically usable review-priority manifests from DeepDRiD.

The review-priority model is only defined for technically usable conventional
color fundus photographs.  This generator therefore retains exactly the rows
with ``overall_quality == 1`` from the existing patient-disjoint escalation
manifests.  It does not relabel, reshuffle, or repartition any image.

Generation is deterministic and fail-closed.  Before writing output, the
script verifies the parent summary and manifest hashes, schema, split names,
labels, quality values, required identifiers, image files, duplicate IDs, and
patient separation across all four partitions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "val", "calibration", "eval")
REQUIRED_COLUMNS = {
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
}
NONEMPTY_COLUMNS = REQUIRED_COLUMNS


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"source summary {path} must contain one JSON object")
    partitions = summary.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError(f"source summary {path} is missing partitions")
    for split in PARTITIONS:
        if not isinstance(partitions.get(split), dict):
            raise ValueError(f"source summary {path} is missing partition {split!r}")
    return summary


def _required_string(row: dict[str, str], key: str, context: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"{context}: missing or blank {key}")
    return value


def read_and_validate_manifest(
    path: Path,
    expected_split: str,
) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ())
            missing_columns = REQUIRED_COLUMNS - set(fieldnames)
            if missing_columns:
                raise ValueError(f"missing columns: {sorted(missing_columns)}")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: manifest is empty")

    for index, row in enumerate(rows, start=2):
        context = f"{path}:{index}"
        for key in NONEMPTY_COLUMNS:
            _required_string(row, key, context)
        if row["split"] != expected_split:
            raise ValueError(
                f"{context}: expected split={expected_split!r}, got {row['split']!r}"
            )
        if row["overall_quality"] not in {"0", "1"}:
            raise ValueError(
                f"{context}: invalid overall_quality={row['overall_quality']!r}"
            )
        if row["dr_grade"] not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"{context}: invalid dr_grade={row['dr_grade']!r}")
        expected_label = (
            "PRIORITY" if int(row["dr_grade"]) >= 2 else "ROUTINE"
        )
        if row["escalation_label"] != expected_label:
            raise ValueError(
                f"{context}: dr_grade={row['dr_grade']} requires "
                f"escalation_label={expected_label!r}, got "
                f"{row['escalation_label']!r}"
            )
        if row["grade_source_field"] not in {
            "left_eye_DR_Level",
            "right_eye_DR_Level",
        }:
            raise ValueError(
                f"{context}: invalid grade_source_field="
                f"{row['grade_source_field']!r}"
            )
        if row["filename_side_matches_grade_field"].lower() not in {
            "true",
            "false",
        }:
            raise ValueError(
                f"{context}: invalid filename_side_matches_grade_field="
                f"{row['filename_side_matches_grade_field']!r}"
            )
        image_path = resolve(Path(row["image_path"]))
        if not image_path.is_file():
            raise ValueError(f"{context}: missing image file {image_path}")
        if image_path.stat().st_size == 0:
            raise ValueError(f"{context}: empty image file {image_path}")
    return fieldnames, rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "images": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "escalation_labels": dict(
            sorted(Counter(row["escalation_label"] for row in rows).items())
        ),
        "dr_grades": dict(
            sorted(Counter(row["dr_grade"] for row in rows).items())
        ),
        "overall_quality": dict(
            sorted(Counter(row["overall_quality"] for row in rows).items())
        ),
        "filename_side_grade_field_mismatches": sum(
            row["filename_side_matches_grade_field"].lower() == "false"
            for row in rows
        ),
    }


def validate_parent_summary_partition(
    source_summary: dict[str, Any],
    split: str,
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    entry = source_summary["partitions"][split]
    expected_hash = entry.get("manifest_sha256")
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"{path}: source summary hash mismatch; expected {expected_hash!r}, "
            f"got {actual_hash}"
        )

    reported_manifest = entry.get("manifest")
    if not isinstance(reported_manifest, str) or not reported_manifest:
        raise ValueError(f"source summary partition {split!r} has no manifest path")
    reported_path = resolve(Path(reported_manifest)).resolve()
    if reported_path != path.resolve():
        raise ValueError(
            f"source summary partition {split!r} points to {reported_path}, "
            f"not {path.resolve()}"
        )

    actual = summarize(rows)
    for key in (
        "images",
        "patients",
        "escalation_labels",
        "dr_grades",
        "overall_quality",
        "filename_side_grade_field_mismatches",
    ):
        if entry.get(key) != actual[key]:
            raise ValueError(
                f"{path}: source summary {key} mismatch; expected "
                f"{entry.get(key)!r}, got {actual[key]!r}"
            )


def pairwise_overlap(
    rows_by_split: dict[str, list[dict[str, str]]],
    field: str,
) -> dict[str, int]:
    values = {
        split: {row[field] for row in rows}
        for split, rows in rows_by_split.items()
    }
    return {
        f"{left}_{right}": len(values[left] & values[right])
        for index, left in enumerate(PARTITIONS)
        for right in PARTITIONS[index + 1 :]
    }


def write_manifest(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_quality_pass_manifests(
    source_dir: Path,
    source_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_dir = resolve(source_dir).resolve()
    source_summary_path = resolve(source_summary_path).resolve()
    output_dir = resolve(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("output directory must differ from the source directory")

    source_summary = load_source_summary(source_summary_path)
    source_rows: dict[str, list[dict[str, str]]] = {}
    fieldnames: list[str] | None = None
    source_paths: dict[str, Path] = {}
    seen_image_ids: dict[str, str] = {}

    for split in PARTITIONS:
        path = source_dir / f"{split}.csv"
        partition_fieldnames, rows = read_and_validate_manifest(path, split)
        if fieldnames is None:
            fieldnames = partition_fieldnames
        elif partition_fieldnames != fieldnames:
            raise ValueError(
                f"{path}: columns/order differ from the other source manifests"
            )
        validate_parent_summary_partition(source_summary, split, path, rows)
        for row in rows:
            image_id = row["image_id"]
            previous = seen_image_ids.get(image_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate image_id {image_id!r} in {previous} and {split}"
                )
            seen_image_ids[image_id] = split
        source_paths[split] = path
        source_rows[split] = rows

    assert fieldnames is not None
    source_patient_overlap = pairwise_overlap(source_rows, "patient_id")
    source_image_overlap = pairwise_overlap(source_rows, "image_id")
    if any(source_patient_overlap.values()):
        raise ValueError(
            f"patient overlap in source manifests: {source_patient_overlap}"
        )
    if any(source_image_overlap.values()):
        raise ValueError(f"image overlap in source manifests: {source_image_overlap}")

    derived_rows = {
        split: [row for row in source_rows[split] if row["overall_quality"] == "1"]
        for split in PARTITIONS
    }
    for split, rows in derived_rows.items():
        if not rows:
            raise ValueError(f"filter produced an empty {split!r} partition")

    derived_paths = {
        split: output_dir / f"{split}.csv" for split in PARTITIONS
    }
    for split in PARTITIONS:
        write_manifest(derived_paths[split], fieldnames, derived_rows[split])

    derived_patient_overlap = pairwise_overlap(derived_rows, "patient_id")
    derived_image_overlap = pairwise_overlap(derived_rows, "image_id")
    if any(derived_patient_overlap.values()) or any(derived_image_overlap.values()):
        raise AssertionError("filtering introduced an impossible partition overlap")

    summary = {
        "schema_version": 1,
        "purpose": (
            "clinically usable conventional color-fundus review-priority research"
        ),
        "derivation": {
            "operation": "row filter only; no relabeling, reshuffling, or repartitioning",
            "field": "overall_quality",
            "retained_value": "1",
            "clinical_boundary": (
                "review-priority labels are used only after technical quality passes"
            ),
        },
        "generator": {
            "path": display_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "source_summary": {
            "path": display_path(source_summary_path),
            "sha256": sha256_file(source_summary_path),
        },
        "upstream_sources": source_summary.get("sources", {}),
        "license": source_summary.get("license"),
        "partitions": {
            split: {
                "source": {
                    **summarize(source_rows[split]),
                    "manifest": display_path(source_paths[split]),
                    "manifest_sha256": sha256_file(source_paths[split]),
                },
                "derived": {
                    **summarize(derived_rows[split]),
                    "manifest": display_path(derived_paths[split]),
                    "manifest_sha256": sha256_file(derived_paths[split]),
                },
                "excluded_images": len(source_rows[split]) - len(derived_rows[split]),
            }
            for split in PARTITIONS
        },
        "global_counts": {
            "source_images": sum(len(rows) for rows in source_rows.values()),
            "derived_images": sum(len(rows) for rows in derived_rows.values()),
            "excluded_images": sum(
                len(source_rows[split]) - len(derived_rows[split])
                for split in PARTITIONS
            ),
        },
        "source_overlap_audit": {
            "patient_id": source_patient_overlap,
            "image_id": source_image_overlap,
        },
        "derived_overlap_audit": {
            "patient_id": derived_patient_overlap,
            "image_id": derived_image_overlap,
        },
        "test_used": False,
        "mshf_used": False,
        "uwf_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/escalation-manifests"),
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        help="defaults to <source-dir>/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/escalation-quality-pass-manifests"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_summary = args.source_summary or args.source_dir / "summary.json"
    summary = build_quality_pass_manifests(
        args.source_dir,
        source_summary,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
