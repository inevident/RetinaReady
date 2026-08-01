#!/usr/bin/env python3
"""Train a compact, multi-task retinal image-quality specialist.

The specialist complements Gemma rather than replacing it: a frozen ImageNet
DenseNet-121 supplies inexpensive visual features, a tiny learned head predicts
the DeepDRiD overall quality label plus its artifact, clarity, and field scores,
and Gemma remains responsible for the structured user-facing explanation.

The split is patient-disjoint within one run. Thresholds are set on patients
held out from that run's head training and evaluated on the official DeepDRiD
validation patients. Historical freshness is a project-level property recorded
in the evaluation ledger, not something this script can infer. The official
test manifest is intentionally not accepted by this script.

Related evidence:
* QuickQual: https://arxiv.org/abs/2307.13646
* Refined multi-task CFP IQA: https://pmc.ncbi.nlm.nih.gov/articles/PMC10903193/
* DeepDRiD: https://pmc.ncbi.nlm.nih.gov/articles/PMC9214346/
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

from calibrate_selective_policy import (
    calibrate_thresholds,
    evaluate as evaluate_policy,
    evaluate_patient_events,
)
from evaluate_quickqual_meme import preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = {
    "patient_id",
    "image_id",
    "image_path",
    "quality_label",
    "artifact",
    "clarity",
    "field_definition",
}


@dataclass(frozen=True)
class Example:
    patient_id: str
    image_id: str
    image_path: str
    quality_label: str
    artifact: float
    clarity: float
    field_definition: float

    @property
    def ready_target(self) -> float:
        return float(self.quality_label == "READY")

    @property
    def factor_targets(self) -> tuple[float, float, float]:
        # Normalize every target so one means best technical quality.
        return (
            1.0 - self.artifact / 10.0,
            self.clarity / 10.0,
            self.field_definition / 10.0,
        )


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path, *, expected_split: str) -> list[Example]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = EXPECTED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{resolved} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{resolved} contains no examples")

    examples: list[Example] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("split") != expected_split:
            raise ValueError(
                f"{resolved}: expected split={expected_split!r}, got {row.get('split')!r}"
            )
        if row["image_id"] in seen:
            raise ValueError(f"{resolved}: duplicate image_id {row['image_id']}")
        seen.add(row["image_id"])
        if row["quality_label"] not in {"READY", "RETAKE"}:
            raise ValueError(f"invalid quality label {row['quality_label']!r}")
        image_path = resolve(Path(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        factors = [float(row[key]) for key in ("artifact", "clarity", "field_definition")]
        if any(not math.isfinite(value) or not 0 <= value <= 10 for value in factors):
            raise ValueError(f"invalid factor score for {row['image_id']}")
        examples.append(
            Example(
                patient_id=row["patient_id"],
                image_id=row["image_id"],
                image_path=row["image_path"],
                quality_label=row["quality_label"],
                artifact=factors[0],
                clarity=factors[1],
                field_definition=factors[2],
            )
        )
    return examples


def patient_grouped_three_way_split(
    examples: list[Example],
    *,
    tuning_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Create patient-disjoint fit, tuning, and within-run calibration sets."""

    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.patient_id, []).append(index)

    strata: dict[int, list[str]] = {}
    for patient_id, indices in groups.items():
        ready_count = sum(examples[index].quality_label == "READY" for index in indices)
        strata.setdefault(ready_count, []).append(patient_id)

    rng = random.Random(seed)
    tuning_patients: set[str] = set()
    calibration_patients: set[str] = set()
    for patient_ids in strata.values():
        rng.shuffle(patient_ids)
        calibration_count = max(1, round(len(patient_ids) * calibration_fraction))
        tuning_count = max(1, round(len(patient_ids) * tuning_fraction))
        if calibration_count + tuning_count >= len(patient_ids):
            raise ValueError("split fractions leave no fit patients in a stratum")
        calibration_patients.update(patient_ids[:calibration_count])
        tuning_patients.update(
            patient_ids[calibration_count : calibration_count + tuning_count]
        )

    fit = [
        index
        for index, example in enumerate(examples)
        if example.patient_id not in calibration_patients
        and example.patient_id not in tuning_patients
    ]
    tuning = [
        index
        for index, example in enumerate(examples)
        if example.patient_id in tuning_patients
    ]
    calibration = [
        index
        for index, example in enumerate(examples)
        if example.patient_id in calibration_patients
    ]
    if not fit or not tuning or not calibration:
        raise ValueError("patient split produced an empty partition")
    partitions = [
        {examples[index].patient_id for index in indices}
        for indices in (fit, tuning, calibration)
    ]
    if any(
        partitions[left] & partitions[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("patient leakage across fit/tuning/calibration split")
    return fit, tuning, calibration


def patient_grouped_split(
    examples: list[Example], *, calibration_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Backward-compatible two-way splitter used by existing utility tests."""

    fit, tuning, calibration = patient_grouped_three_way_split(
        examples,
        tuning_fraction=calibration_fraction,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    return [*fit, *tuning], calibration


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cache_key(manifest_path: Path, checkpoint_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(sha256_file(resolve(manifest_path)).encode("ascii"))
    digest.update(checkpoint_name.encode("utf-8"))
    digest.update(b"quickqual-preprocess-512-v1")
    return digest.hexdigest()[:16]


def extract_features(
    examples: list[Example],
    *,
    manifest_path: Path,
    cache_dir: Path,
    batch_size: int,
    device: str,
    torch: Any,
    np: Any,
) -> Any:
    """Return frozen DenseNet features, caching only derived non-image arrays."""

    checkpoint_name = "torchvision-densenet121-imagenet1k-v1"
    cache_path = resolve(cache_dir) / f"{manifest_path.stem}-{cache_key(manifest_path, checkpoint_name)}.npz"
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        image_ids = cached["image_ids"].tolist()
        if image_ids != [example.image_id for example in examples]:
            raise ValueError(f"feature cache image order mismatch: {cache_path}")
        features = cached["features"].astype("float32", copy=False)
        print(f"Loaded {features.shape} features from {cache_path}")
        return features

    from PIL import Image
    from torchvision import models
    from torchvision.transforms import functional

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)

    parts: list[Any] = []
    started = time.perf_counter()
    for offset in range(0, len(examples), batch_size):
        batch_examples = examples[offset : offset + batch_size]
        tensors = []
        for example in batch_examples:
            with Image.open(resolve(Path(example.image_path))) as image:
                tensors.append(preprocess(image, np, functional))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            part = model(batch).detach().float().cpu().numpy()
        parts.append(part)
        completed = min(offset + len(batch_examples), len(examples))
        if completed % 200 == 0 or completed == len(examples):
            elapsed = time.perf_counter() - started
            print(f"Extracted {completed}/{len(examples)} ({completed / elapsed:.1f} images/s)")

    features = np.concatenate(parts).astype("float32", copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        image_ids=np.asarray([example.image_id for example in examples]),
    )
    print(f"Wrote feature cache {cache_path}")
    return features


def auc(scores: Iterable[float], targets: Iterable[float]) -> float:
    """Tie-aware Mann-Whitney ROC-AUC without a scikit-learn dependency."""

    pairs = sorted(zip(scores, targets, strict=True), key=lambda pair: pair[0])
    positives = sum(target == 1 for _, target in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        stop = index + 1
        while stop < len(pairs) and pairs[stop][0] == pairs[index][0]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        rank_sum += average_rank * sum(target == 1 for _, target in pairs[index:stop])
        index = stop
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def classification_summary(
    examples: list[Example], probabilities: Any, *, threshold: float = 0.5
) -> dict[str, Any]:
    targets = [example.ready_target for example in examples]
    predictions = [float(probability) > threshold for probability in probabilities]
    ready_total = sum(targets)
    retake_total = len(targets) - ready_total
    true_ready = sum(prediction and target == 1 for prediction, target in zip(predictions, targets, strict=True))
    false_ready = sum(prediction and target == 0 for prediction, target in zip(predictions, targets, strict=True))
    true_retake = sum(not prediction and target == 0 for prediction, target in zip(predictions, targets, strict=True))
    false_retake = sum(not prediction and target == 1 for prediction, target in zip(predictions, targets, strict=True))
    return {
        "samples": len(examples),
        "threshold": threshold,
        "accuracy": (true_ready + true_retake) / len(examples),
        "balanced_accuracy": ((true_ready / ready_total) + (true_retake / retake_total)) / 2,
        "roc_auc_ready_positive": auc(probabilities.tolist(), targets),
        "ready_recall": true_ready / ready_total,
        "retake_recall": true_retake / retake_total,
        "false_ready_rate": false_ready / retake_total,
        "false_retake_rate": false_retake / ready_total,
        "confusion": {
            "READY": {"READY": true_ready, "RETAKE": false_retake},
            "RETAKE": {"READY": false_ready, "RETAKE": true_retake},
        },
    }


def result_rows(examples: list[Example], probabilities: Any) -> list[dict[str, Any]]:
    return [
        {
            "patient_id": example.patient_id,
            "image_id": example.image_id,
            "image_path": example.image_path,
            "truth": example.quality_label,
            "decision_score": float(probability),
        }
        for example, probability in zip(examples, probabilities, strict=True)
    ]


def train_member(
    *,
    fit_features: Any,
    fit_ready: Any,
    fit_factors: Any,
    tuning_features: Any,
    tuning_ready: Any,
    tuning_factors: Any,
    input_dim: int,
    hidden_dim: int,
    auxiliary_weight: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    seed: int,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    class QualityHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if hidden_dim:
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.15),
                    torch.nn.Linear(hidden_dim, 4),
                )
            else:
                self.network = torch.nn.Linear(input_dim, 4)

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    torch.manual_seed(seed)
    model = QualityHead()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    classification_loss = torch.nn.BCEWithLogitsLoss()
    factor_loss = torch.nn.SmoothL1Loss(beta=0.1)
    generator = torch.Generator().manual_seed(seed)

    best_selection_score = -math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    batch_size = min(128, len(fit_features))
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(fit_features), generator=generator)
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            outputs = model(fit_features[indices])
            loss = classification_loss(outputs[:, 0], fit_ready[indices])
            if auxiliary_weight:
                predicted_factors = torch.sigmoid(outputs[:, 1:])
                loss = loss + auxiliary_weight * sum(
                    factor_loss(predicted_factors[:, index], fit_factors[indices, index])
                    for index in range(3)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            tuning_outputs = model(tuning_features)
            scores = torch.sigmoid(tuning_outputs[:, 0]).numpy()
            tuning_auc = auc(scores.tolist(), tuning_ready.numpy().tolist())
            tuning_factor_mae = float(
                torch.abs(torch.sigmoid(tuning_outputs[:, 1:]) - tuning_factors)
                .mean()
                .item()
            )
        current_selection_score = (
            -tuning_factor_mae if auxiliary_weight else tuning_auc
        )
        if current_selection_score > best_selection_score + 1e-5:
            best_selection_score = current_selection_score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("training failed to produce a checkpoint")
    metadata = {
        "seed": seed,
        "best_epoch": best_epoch + 1,
        "selection_metric": "negative_tuning_factor_mae" if auxiliary_weight else "tuning_auc",
        "selection_score": best_selection_score,
    }
    return best_state, metadata


def refit_member(
    *,
    development_features: Any,
    development_ready: Any,
    development_factors: Any,
    input_dim: int,
    hidden_dim: int,
    auxiliary_weight: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    torch: Any,
) -> dict[str, Any]:
    """Refit a frozen configuration without touching calibration patients."""

    class QualityHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if hidden_dim:
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.15),
                    torch.nn.Linear(hidden_dim, 4),
                )
            else:
                self.network = torch.nn.Linear(input_dim, 4)

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    torch.manual_seed(seed)
    model = QualityHead()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    classification_loss = torch.nn.BCEWithLogitsLoss()
    factor_loss = torch.nn.SmoothL1Loss(beta=0.1)
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(128, len(development_features))
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(development_features), generator=generator)
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            outputs = model(development_features[indices])
            loss = classification_loss(outputs[:, 0], development_ready[indices])
            if auxiliary_weight:
                predicted_factors = torch.sigmoid(outputs[:, 1:])
                loss = loss + auxiliary_weight * sum(
                    factor_loss(
                        predicted_factors[:, index],
                        development_factors[indices, index],
                    )
                    for index in range(3)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def predict_ensemble(
    features: Any,
    *,
    states: list[dict[str, Any]],
    input_dim: int,
    hidden_dim: int,
    torch: Any,
) -> tuple[Any, Any]:
    class QualityHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if hidden_dim:
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.15),
                    torch.nn.Linear(hidden_dim, 4),
                )
            else:
                self.network = torch.nn.Linear(input_dim, 4)

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    predictions = []
    for state in states:
        model = QualityHead()
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            output = model(features)
            predictions.append(
                torch.cat(
                    (torch.sigmoid(output[:, :1]), torch.sigmoid(output[:, 1:])),
                    dim=1,
                ).numpy()
            )
    stacked = sum(predictions) / len(predictions)
    return stacked[:, 0], stacked[:, 1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.csv"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("data/manifests/val.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quality-specialist"))
    parser.add_argument("--cache-dir", type=Path, default=Path("ml/cache/quality-specialist"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tuning-fraction", type=float, default=0.12)
    parser.add_argument("--calibration-fraction", type=float, default=0.36)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--false-ready-risk", type=float, default=0.10)
    parser.add_argument("--false-retake-risk", type=float, default=0.10)
    parser.add_argument(
        "--delta-per-gate",
        "--delta",
        dest="delta_per_gate",
        type=float,
        default=0.025,
        help="one-sided failure probability per gate; 0.025 gives >=95%% joint confidence",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_manifest.name == "test.csv":
        raise ValueError("this trainer intentionally refuses the already-open test manifest")
    if args.batch_size <= 0 or args.hidden_dim < 0:
        raise ValueError("batch size must be positive and hidden dim non-negative")
    if not 0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration fraction must be between zero and 0.5")
    if not 0 < args.tuning_fraction < 0.5:
        raise ValueError("tuning fraction must be between zero and 0.5")
    if args.tuning_fraction + args.calibration_fraction >= 0.8:
        raise ValueError("tuning and calibration fractions leave too little fit data")
    if not 0 <= args.auxiliary_weight <= 1:
        raise ValueError("auxiliary weight must be between zero and one")
    if min(args.epochs, args.patience, args.ensemble_members) <= 0:
        raise ValueError("epochs, patience, and ensemble members must be positive")

    import numpy as np
    import torch

    train_examples = read_manifest(args.train_manifest, expected_split="train")
    eval_examples = read_manifest(args.eval_manifest, expected_split="val")
    train_patients = {example.patient_id for example in train_examples}
    eval_patients = {example.patient_id for example in eval_examples}
    if train_patients & eval_patients:
        raise ValueError("patient overlap between training and evaluation manifests")

    device = choose_device(torch, args.device)
    train_features = extract_features(
        train_examples,
        manifest_path=args.train_manifest,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    eval_features = extract_features(
        eval_examples,
        manifest_path=args.eval_manifest,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )

    fit_indices, tuning_indices, calibration_indices = patient_grouped_three_way_split(
        train_examples,
        tuning_fraction=args.tuning_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    selection_mean = train_features[fit_indices].mean(axis=0, keepdims=True)
    selection_std = train_features[fit_indices].std(axis=0, keepdims=True)
    selection_std[selection_std < 1e-6] = 1.0
    selection_standardized = (train_features - selection_mean) / selection_std

    ready_targets = np.asarray(
        [example.ready_target for example in train_examples], dtype="float32"
    )
    factor_targets = np.asarray(
        [example.factor_targets for example in train_examples], dtype="float32"
    )
    fit_x = torch.from_numpy(selection_standardized[fit_indices]).float()
    fit_ready = torch.from_numpy(ready_targets[fit_indices]).float()
    fit_factors = torch.from_numpy(factor_targets[fit_indices]).float()
    tuning_x = torch.from_numpy(selection_standardized[tuning_indices]).float()
    tuning_ready = torch.from_numpy(ready_targets[tuning_indices]).float()
    tuning_factors = torch.from_numpy(factor_targets[tuning_indices]).float()

    members: list[dict[str, Any]] = []
    for member in range(args.ensemble_members):
        _, metadata = train_member(
            fit_features=fit_x,
            fit_ready=fit_ready,
            fit_factors=fit_factors,
            tuning_features=tuning_x,
            tuning_ready=tuning_ready,
            tuning_factors=tuning_factors,
            input_dim=selection_standardized.shape[1],
            hidden_dim=args.hidden_dim,
            auxiliary_weight=args.auxiliary_weight,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + member,
            torch=torch,
        )
        members.append(metadata)
        print(f"Selected member {member + 1}/{args.ensemble_members}: {metadata}")

    development_indices = [*fit_indices, *tuning_indices]
    feature_mean = train_features[development_indices].mean(axis=0, keepdims=True)
    feature_std = train_features[development_indices].std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    train_standardized = (train_features - feature_mean) / feature_std
    eval_standardized = (eval_features - feature_mean) / feature_std
    development_x = torch.from_numpy(
        train_standardized[development_indices]
    ).float()
    development_ready = torch.from_numpy(ready_targets[development_indices]).float()
    development_factors = torch.from_numpy(factor_targets[development_indices]).float()

    states: list[dict[str, Any]] = []
    for member, metadata in enumerate(members):
        state = refit_member(
            development_features=development_x,
            development_ready=development_ready,
            development_factors=development_factors,
            input_dim=train_standardized.shape[1],
            hidden_dim=args.hidden_dim,
            auxiliary_weight=args.auxiliary_weight,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=metadata["best_epoch"],
            seed=args.seed + member,
            torch=torch,
        )
        states.append(state)
        metadata["refit_images"] = len(development_indices)
        metadata["refit_patients"] = len(
            {train_examples[index].patient_id for index in development_indices}
        )
        print(
            f"Refit member {member + 1}/{args.ensemble_members} for "
            f"{metadata['best_epoch']} frozen epochs"
        )

    calibration_probabilities, calibration_factors = predict_ensemble(
        torch.from_numpy(train_standardized[calibration_indices]).float(),
        states=states,
        input_dim=train_standardized.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    eval_probabilities, eval_factors = predict_ensemble(
        torch.from_numpy(eval_standardized).float(),
        states=states,
        input_dim=train_standardized.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    calibration_examples = [train_examples[index] for index in calibration_indices]
    calibration_results = result_rows(calibration_examples, calibration_probabilities)
    evaluation_results = result_rows(eval_examples, eval_probabilities)
    policy = calibrate_thresholds(
        calibration_results,
        false_ready_risk=args.false_ready_risk,
        false_retake_risk=args.false_retake_risk,
        delta=args.delta_per_gate,
        unit="patient",
    )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "architecture": "frozen DenseNet-121 plus multi-task MLP ensemble",
        "hidden_dim": args.hidden_dim,
        "input_dim": int(train_standardized.shape[1]),
        "members": states,
        "feature_mean": torch.from_numpy(feature_mean.astype("float32")),
        "feature_std": torch.from_numpy(feature_std.astype("float32")),
        "policy": policy,
        "factor_order": ["artifact_quality", "clarity", "field_definition"],
    }
    torch.save(artifact, output_dir / "specialist.pt")

    report = {
        "schema_version": 1,
        "method": "frozen DenseNet-121 plus multi-task quality head",
        "role": "visual specialist feeding the Gemma explanation layer",
        "historical_calibration_status": {
            "freshness_must_be_verified_against_project_history": True,
            "deployment_guarantee": False,
            "ledger": "docs/EVALUATION_LEDGER.md",
        },
        "data": {
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": sha256_file(resolve(args.train_manifest)),
            "evaluation_manifest": str(args.eval_manifest),
            "evaluation_manifest_sha256": sha256_file(resolve(args.eval_manifest)),
            "fit_images": len(fit_indices),
            "fit_patients": len({train_examples[index].patient_id for index in fit_indices}),
            "tuning_images": len(tuning_indices),
            "tuning_patients": len({train_examples[index].patient_id for index in tuning_indices}),
            "final_development_images": len(development_indices),
            "final_development_patients": len({train_examples[index].patient_id for index in development_indices}),
            "calibration_images": len(calibration_indices),
            "calibration_patients": len({train_examples[index].patient_id for index in calibration_indices}),
            "evaluation_images": len(eval_examples),
            "evaluation_patients": len(eval_patients),
            "patient_overlap": 0,
            "official_test_used": False,
        },
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"train_manifest", "eval_manifest", "output_dir", "cache_dir"}
        },
        "members": members,
        "calibration_at_0_5": classification_summary(
            calibration_examples, calibration_probabilities
        ),
        "exploratory_validation_at_0_5": classification_summary(
            eval_examples, eval_probabilities
        ),
        "selective_policy": policy,
        "calibration_selective_metrics": evaluate_policy(calibration_results, policy),
        "calibration_patient_event_metrics": evaluate_patient_events(calibration_results, policy),
        "calibration_results": calibration_results,
        "exploratory_validation_selective_metrics": evaluate_policy(evaluation_results, policy),
        "exploratory_validation_patient_event_metrics": evaluate_patient_events(evaluation_results, policy),
        "factor_mae_on_exploratory_validation": {
            name: float(np.abs(eval_factors[:, index] - np.asarray([example.factor_targets[index] for example in eval_examples])).mean())
            for index, name in enumerate(("artifact_quality", "clarity", "field_definition"))
        },
        "limitations": [
            "This is technical capture-quality research, not diagnosis.",
            "The official test split was not used in this experiment.",
            "Validation results are exploratory because prior architecture comparisons viewed this split.",
            "The model and early stopping checkpoint are frozen before the calibration patients are scored.",
            "Each risk bound treats a patient as one unit and requires exchangeability with future patients.",
            "The two per-gate delta values are combined with a union bound; these are research bounds, not clinical validation.",
            "Finite-sample guarantees additionally require calibration patients that are fresh to the entire development history; consult docs/EVALUATION_LEDGER.md.",
        ],
        "results": evaluation_results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "calibration_at_0_5": report["calibration_at_0_5"],
        "exploratory_validation_at_0_5": report["exploratory_validation_at_0_5"],
        "exploratory_validation_selective_metrics": report["exploratory_validation_selective_metrics"],
        "artifact": str(output_dir / "specialist.pt"),
        "report": str(output_dir / "report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
