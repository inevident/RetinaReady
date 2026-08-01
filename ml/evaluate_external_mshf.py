#!/usr/bin/env python3
"""Evaluate the frozen RetinaReady specialist on the untouched MSHF test set.

MSHF supplies 1,302 images from seven sources and an author-provided 1,042/260
train/test split.  This evaluator uses only the 260-image test directory and
reconstructs the published binary labels by majority vote across the three
annotators in ``Individual_scores.xlsx.xlsx``.  It never fits a model or
changes a threshold.

The result is an external device-shift stress test, not a clinical validation
or a new calibration guarantee.  MSHF does not expose a patient identifier in
the release, so every error rate reported here is explicitly image-level.

Dataset: https://doi.org/10.6084/m9.figshare.21507564
Paper: https://doi.org/10.1038/s41597-023-02188-x
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterable
from xml.etree import ElementTree
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SHEET_NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
SOURCES = (
    "UWF-mosaic",
    "DR-XJU",
    "DR-ZJU",
    "Glaucoma",
    "Healthy",
    "Local1",
    "Local2",
)
CAMERA_GROUPS = {
    "UWF-mosaic": "ultrawide-field",
    "DR-XJU": "standard-CFP",
    "DR-ZJU": "standard-CFP",
    "Glaucoma": "standard-CFP",
    "Healthy": "standard-CFP",
    "Local1": "portable-camera",
    "Local2": "portable-camera",
}
ANNOTATOR_COLUMNS = {
    "illumination": ("B", "G", "L"),
    "clarity": ("C", "H", "M"),
    "contrast": ("D", "I", "N"),
    "overall": ("E", "J", "O"),
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_column(reference: str) -> str:
    match = re.match(r"([A-Z]+)", reference)
    if not match:
        raise ValueError(f"invalid spreadsheet cell reference: {reference!r}")
    return match.group(1)


def read_first_xlsx_sheet(path: Path) -> list[dict[str, str]]:
    """Read the first XLSX sheet with only the Python standard library."""

    with ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.iter(f"{SHEET_NAMESPACE}t"))
            for item in shared_root.findall(f"{SHEET_NAMESPACE}si")
        ]
        sheet_root = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet1.xml")
        )

    rows: list[dict[str, str]] = []
    for row in sheet_root.findall(f".//{SHEET_NAMESPACE}row"):
        values: dict[str, str] = {}
        for cell in row.findall(f"{SHEET_NAMESPACE}c"):
            value_node = cell.find(f"{SHEET_NAMESPACE}v")
            if value_node is None or value_node.text is None:
                continue
            value = value_node.text
            if cell.get("t") == "s":
                value = shared_strings[int(value)]
            values[_cell_column(cell.get("r", ""))] = value
        rows.append(values)
    return rows


def load_majority_labels(path: Path) -> dict[str, dict[str, int]]:
    rows = read_first_xlsx_sheet(path)
    labels: dict[str, dict[str, int]] = {}
    for row in rows[2:]:
        image_name = row.get("A")
        if not image_name or not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if image_name in labels:
            raise ValueError(f"duplicate MSHF label row: {image_name}")
        image_labels: dict[str, int] = {}
        for target, columns in ANNOTATOR_COLUMNS.items():
            try:
                votes = [int(row[column]) for column in columns]
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"missing annotator vote for {image_name} / {target}"
                ) from error
            if any(vote not in {0, 1} for vote in votes):
                raise ValueError(f"non-binary vote for {image_name} / {target}")
            image_labels[target] = int(sum(votes) >= 2)
        labels[image_name] = image_labels
    if len(labels) != 1302:
        raise ValueError(f"expected 1,302 MSHF labels, found {len(labels)}")
    return labels


def source_for(image_name: str) -> str:
    for source in SOURCES:
        if image_name.startswith(f"{source}-"):
            return source
    raise ValueError(f"unrecognized MSHF image source: {image_name}")


def roc_auc(rows: list[dict[str, Any]], *, label_key: str, score_key: str) -> float | None:
    positive = [float(row[score_key]) for row in rows if int(row[label_key]) == 1]
    negative = [float(row[score_key]) for row in rows if int(row[label_key]) == 0]
    if not positive or not negative:
        return None
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def wilson_interval(errors: int, samples: int, *, z: float = 1.959963984540054) -> list[float] | None:
    if samples <= 0:
        return None
    rate = errors / samples
    denominator = 1 + z * z / samples
    center = (rate + z * z / (2 * samples)) / denominator
    radius = z * math.sqrt(
        rate * (1 - rate) / samples + z * z / (4 * samples * samples)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in rows if row["truth"] == "READY"]
    retake = [row for row in rows if row["truth"] == "RETAKE"]
    at_half = [
        "READY" if float(row["decision_score"]) >= 0.5 else "RETAKE"
        for row in rows
    ]
    correct_at_half = sum(
        prediction == row["truth"] for prediction, row in zip(at_half, rows)
    )
    ready_recall = (
        sum(prediction == "READY" for prediction, row in zip(at_half, rows) if row["truth"] == "READY")
        / len(ready)
        if ready
        else None
    )
    retake_recall = (
        sum(prediction == "RETAKE" for prediction, row in zip(at_half, rows) if row["truth"] == "RETAKE")
        / len(retake)
        if retake
        else None
    )

    decisions = Counter(str(row["decision"]) for row in rows)
    accepted = [row for row in rows if row["decision"] != "LIMITED"]
    false_ready = sum(
        row["truth"] == "RETAKE" and row["decision"] == "READY" for row in rows
    )
    false_retake = sum(
        row["truth"] == "READY" and row["decision"] == "RETAKE" for row in rows
    )
    return {
        "images": len(rows),
        "truth_counts": {"READY": len(ready), "RETAKE": len(retake)},
        "at_0_5": {
            "roc_auc_ready_positive": roc_auc(
                rows, label_key="overall", score_key="decision_score"
            ),
            "accuracy": correct_at_half / len(rows) if rows else None,
            "balanced_accuracy": (
                (ready_recall + retake_recall) / 2
                if ready_recall is not None and retake_recall is not None
                else None
            ),
            "ready_recall": ready_recall,
            "retake_recall": retake_recall,
            "false_ready_rate": 1 - retake_recall if retake_recall is not None else None,
            "false_retake_rate": 1 - ready_recall if ready_recall is not None else None,
        },
        "frozen_selective_policy": {
            "decision_counts": {
                "READY": decisions["READY"],
                "RETAKE": decisions["RETAKE"],
                "LIMITED": decisions["LIMITED"],
            },
            "coverage": len(accepted) / len(rows) if rows else None,
            "accepted_accuracy": (
                sum(row["decision"] == row["truth"] for row in accepted) / len(accepted)
                if accepted
                else None
            ),
            "false_ready_count": false_ready,
            "false_ready_rate_given_retake": false_ready / len(retake) if retake else None,
            "false_ready_rate_wilson_95": wilson_interval(false_ready, len(retake)),
            "false_retake_count": false_retake,
            "false_retake_rate_given_ready": false_retake / len(ready) if ready else None,
            "false_retake_rate_wilson_95": wilson_interval(false_retake, len(ready)),
        },
        "factor_alignment": {
            "clarity_auc_against_mshf_binary_clarity": roc_auc(
                rows, label_key="clarity", score_key="clarity_score"
            ),
            "note": (
                "Only clarity has a directly named MSHF counterpart. RetinaReady's "
                "artifact-quality and field-definition targets are not equivalent "
                "to MSHF illumination and contrast, so they are not scored here."
            ),
        },
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/external/mshf/MSHF dataset 2.0"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/external/mshf/MSHF-dataset-2.0.zip"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/retinaready-quality-specialist"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mshf-external-test-specialist.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = resolve(args.dataset_root)
    archive_path = resolve(args.archive)
    model_dir = resolve(args.model_dir)
    output_path = resolve(args.output)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    labels_path = dataset_root / "Individual_scores.xlsx.xlsx"
    test_dir = dataset_root / "AI-use" / "test"
    for path in (archive_path, labels_path, test_dir, model_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    expected_archive_md5 = "58203e6c2e064dafc800f5b83887487b"
    actual_archive_md5 = md5_file(archive_path)
    if actual_archive_md5 != expected_archive_md5:
        raise ValueError(
            f"MSHF archive checksum mismatch: {actual_archive_md5}"
        )

    labels = load_majority_labels(labels_path)
    image_paths = sorted(test_dir.glob("*.jpg"))
    if len(image_paths) != 260:
        raise ValueError(f"expected 260 MSHF test images, found {len(image_paths)}")
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    missing_labels = [path.name for path in image_paths if path.name not in labels]
    if missing_labels:
        raise ValueError(f"missing labels for MSHF test images: {missing_labels[:5]}")

    from app.quality_specialist import QualitySpecialist

    device = choose_device(args.device)
    specialist = QualitySpecialist(
        backbone_path=model_dir / "densenet121-a639ec97.pth",
        decision_head_path=model_dir / "decision-head.pt",
        factor_head_path=model_dir / "factor-head.pt",
        device=device,
    )
    if not specialist.bundle_verified:
        raise ValueError("specialist bundle manifest was not verified")

    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    started = time.perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        before = time.perf_counter()
        assessment = specialist.assess(image_path.read_bytes())
        latencies_ms.append((time.perf_counter() - before) * 1000)
        truth = labels[image_path.name]
        source = source_for(image_path.name)
        results.append(
            {
                "image_name": image_path.name,
                "source": source,
                "camera_group": CAMERA_GROUPS[source],
                **truth,
                "truth": "READY" if truth["overall"] else "RETAKE",
                "decision_score": assessment.ready_score,
                "decision": assessment.decision,
                "clarity_score": assessment.scores["clarity"] / 100.0,
            }
        )
        if index % 25 == 0 or index == len(image_paths):
            print(f"Scored {index}/{len(image_paths)} MSHF test images")

    by_source = {
        source: summarize([row for row in results if row["source"] == source])
        for source in SOURCES
        if any(row["source"] == source for row in results)
    }
    by_camera = {
        camera: summarize([row for row in results if row["camera_group"] == camera])
        for camera in sorted(set(CAMERA_GROUPS.values()))
        if any(row["camera_group"] == camera for row in results)
    }
    report = {
        "schema_version": 1,
        "evaluation_type": "untouched external image-level device-shift stress test",
        "dataset": {
            "name": "MSHF dataset 2.0",
            "doi": "10.6084/m9.figshare.21507564",
            "paper_doi": "10.1038/s41597-023-02188-x",
            "license": "CC BY 4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_md5": actual_archive_md5,
            "archive_md5_expected": expected_archive_md5,
            "archive_sha256": sha256_file(archive_path),
            "labels_sha256": sha256_file(labels_path),
            "split": "author-provided AI-use/test",
            "test_images": len(image_paths),
            "model_or_threshold_tuning_on_mshf": False,
        },
        "model": {
            "bundle_manifest_sha256": sha256_file(model_dir / "manifest.json"),
            "bundle_verified": specialist.bundle_verified,
            "device": device,
            "policy": {
                "ready_threshold_strictly_greater_than": results
                and specialist._policy["ready_threshold_strictly_greater_than"],
                "retake_threshold_strictly_less_than": results
                and specialist._policy["retake_threshold_strictly_less_than"],
            },
        },
        "overall": summarize(results),
        "by_camera_group": by_camera,
        "by_source": by_source,
        "runtime": {
            "total_seconds": time.perf_counter() - started,
            "mean_ms_per_image": statistics.fmean(latencies_ms),
            "median_ms_per_image": statistics.median(latencies_ms),
            "p95_ms_per_image": percentile(latencies_ms, 0.95),
        },
        "limitations": [
            "MSHF patient identifiers are not exposed, so metrics and intervals are image-level only.",
            "The frozen DeepDRiD thresholds are not recalibrated on MSHF and carry no risk guarantee here.",
            "MSHF overall quality is a related but not identical labeling protocol to DeepDRiD.",
            "Per-source groups are small and some contain only one overall-quality class.",
            "This is technical capture-quality research, not diagnosis or clinical validation.",
        ],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"overall": report["overall"], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
