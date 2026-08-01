#!/usr/bin/env python3
"""Evaluate the published QuickQual-MEME score on a RetinaReady manifest.

QuickQual-MEME is a deliberately tiny retinal-quality head over frozen
ImageNet DenseNet-121 features.  This script reproduces the public inference
formula from Engelmann et al. (2023) and writes raw, uncalibrated scores so it
can be tested as an external specialist or ensemble input without touching the
sealed policy threshold.

Paper: https://arxiv.org/abs/2307.13646
Code:  https://github.com/justinengelmann/QuickQual
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_INDICES = (71, 109, 121, 53, 55, 123, 29, 133, 84)
MEME_WEIGHTS = (-1411.32, 517.09, 342.41, -707.9, 1442.09, -23.25, -541.64, -8.44, 5.44)
MEME_BIAS = 5.18


def resolve_from_project(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(manifest: Path, limit: int | None) -> list[dict[str, str]]:
    with resolve_from_project(manifest).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("manifest contains no rows")
    for row in rows:
        if row.get("quality_label") not in {"READY", "RETAKE"}:
            raise ValueError(f"invalid quality label: {row.get('quality_label')!r}")
        image_path = resolve_from_project(Path(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    return rows


def crop_black_border(image: Any, np: Any, *, threshold: int = 15) -> Any:
    """Match QuickQual's public black-border crop and square padding."""

    array = np.asarray(image.convert("RGB"))
    visible = array.mean(axis=-1) > threshold
    ys, xs = np.where(visible)
    if not len(xs) or not len(ys):
        return image.convert("RGB")
    buffer = 20
    left = max(0, int(xs.min()) - buffer)
    right = min(array.shape[1], int(xs.max()) + buffer + 1)
    top = max(0, int(ys.min()) - buffer)
    bottom = min(array.shape[0], int(ys.max()) + buffer + 1)
    return image.convert("RGB").crop((left, top, right, bottom))


def preprocess(image: Any, np: Any, functional: Any) -> Any:
    image = crop_black_border(image, np)
    width, height = image.size
    if width > height:
        delta = width - height
        padding = [0, delta // 2, 0, delta - delta // 2]
    else:
        delta = height - width
        padding = [delta // 2, 0, delta - delta // 2, 0]
    image = functional.pad(image, padding)
    image = functional.resize(image, [512, 512], antialias=True)
    tensor = functional.to_tensor(image)
    return functional.normalize(tensor, [0.5] * 3, [0.5] * 3)


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def roc_auc_ready_positive(results: list[dict[str, Any]]) -> float:
    positives = [r["ready_probability"] for r in results if r["truth"] == "READY"]
    negatives = [r["ready_probability"] for r in results if r["truth"] == "RETAKE"]
    if not positives or not negatives:
        raise ValueError("both READY and RETAKE examples are required for ROC-AUC")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def summarize(results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    matrix = {
        truth: {
            prediction: sum(
                r["truth"] == truth and r["prediction_at_0_5"] == prediction
                for r in results
            )
            for prediction in ("READY", "RETAKE")
        }
        for truth in ("READY", "RETAKE")
    }
    ready_total = sum(r["truth"] == "READY" for r in results)
    retake_total = len(results) - ready_total
    correct = matrix["READY"]["READY"] + matrix["RETAKE"]["RETAKE"]
    ready_recall = matrix["READY"]["READY"] / ready_total
    retake_recall = matrix["RETAKE"]["RETAKE"] / retake_total
    return {
        "samples": len(results),
        "truth_counts": {"READY": ready_total, "RETAKE": retake_total},
        "confusion_matrix_at_0_5": matrix,
        "metrics_at_0_5": {
            "accuracy": round(correct / len(results), 6),
            "balanced_accuracy": round((ready_recall + retake_recall) / 2, 6),
            "ready_recall": round(ready_recall, 6),
            "retake_recall": round(retake_recall, 6),
            "false_ready_rate": round(matrix["RETAKE"]["READY"] / retake_total, 6),
        },
        "roc_auc_ready_positive": round(roc_auc_ready_positive(results), 6),
        "wall_seconds": round(wall_seconds, 3),
        "images_per_second": round(len(results) / wall_seconds, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/val.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import models
    from torchvision.transforms import functional as functional

    rows = load_rows(args.manifest, args.limit)
    device = choose_device(torch, args.device)
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    weights = torch.tensor(MEME_WEIGHTS, dtype=torch.float32, device=device)
    bias = torch.tensor(MEME_BIAS, dtype=torch.float32, device=device)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset : offset + args.batch_size]
        tensors = []
        for row in batch_rows:
            with Image.open(resolve_from_project(Path(row["image_path"]))) as image:
                tensors.append(preprocess(image, np, functional))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            features = model(batch)
            selected = features[:, list(FEATURE_INDICES)]
            bad_probabilities = torch.sigmoid(selected @ weights + bias)
        for row, bad_probability in zip(batch_rows, bad_probabilities.tolist(), strict=True):
            if not math.isfinite(bad_probability):
                raise ValueError(f"non-finite score for {row['image_id']}")
            ready_probability = 1.0 - float(bad_probability)
            results.append(
                {
                    "patient_id": row["patient_id"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "truth": row["quality_label"],
                    "ready_probability": ready_probability,
                    "bad_probability": float(bad_probability),
                    "prediction_at_0_5": "READY" if ready_probability > 0.5 else "RETAKE",
                }
            )

    wall_seconds = time.perf_counter() - started
    payload = {
        "run": {
            "method": "QuickQual-MEME published external baseline",
            "paper": "https://arxiv.org/abs/2307.13646",
            "code": "https://github.com/justinengelmann/QuickQual",
            "manifest": str(args.manifest),
            "device": device,
            "batch_size": args.batch_size,
            "calibration": "none; raw external scores",
        },
        "summary": summarize(results, wall_seconds),
        "results": results,
    }
    output = resolve_from_project(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
