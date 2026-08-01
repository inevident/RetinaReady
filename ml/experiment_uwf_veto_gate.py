#!/usr/bin/env python3
"""Train and evaluate an isolated UWF-vs-conventional CFP veto gate.

The experiment uses only DeepDRiD regular-CFP and ultra-widefield (UWF)
training/validation data already present locally.  It shares patient IDs across
modalities, excludes every validation patient from training, and splits the
remaining development patients as whole groups.  DeepDRiD test data and MSHF
are refused.

This gate has one permitted product action: a sufficiently high UWF score may
force ``LIMITED``.  A low score leaves the existing quality decision unchanged;
it can never promote READY or RETAKE.  The script is experimental-only and does
not modify the deployed runtime, model bundle, or thresholds.
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

from calibrate_selective_policy import exact_upper_bound, maximum_certified_errors
from train_quality_specialist import (
    auc,
    choose_device,
    extract_features,
    read_manifest,
    resolve,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("CONVENTIONAL_CFP", "UWF")


@dataclass(frozen=True)
class ModalityExample:
    patient_id: str
    image_id: str
    image_path: str
    modality: str
    source_split: str

    @property
    def target(self) -> float:
        return float(self.modality == "UWF")

    @property
    def record_id(self) -> str:
        return f"{self.modality}:{self.source_split}:{self.image_id}"


def regular_examples(manifest_path: Path, *, expected_split: str) -> list[ModalityExample]:
    return [
        ModalityExample(
            patient_id=example.patient_id,
            image_id=example.image_id,
            image_path=example.image_path,
            modality="CONVENTIONAL_CFP",
            source_split=f"regular-{expected_split}",
        )
        for example in read_manifest(manifest_path, expected_split=expected_split)
    ]


def _resolve_uwf_image(csv_path: Path, patient_id: str, image_id: str) -> Path:
    image_dir = resolve(csv_path).parent / "Images" / patient_id
    candidates = [image_dir / f"{image_id}.jpg"]
    if "_" in image_id:
        candidates.append(image_dir / f"{image_id.split('_', 1)[1]}.jpg")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no UWF image matches patient={patient_id}, image={image_id} in {image_dir}"
    )


def read_uwf_csv(csv_path: Path, *, expected_source: str) -> tuple[list[ModalityExample], int]:
    """Read a UWF manifest, including two locally renamed patient-121 files."""

    resolved = resolve(csv_path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "image_id", "image_path", "DR_level", "position"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{resolved} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{resolved} contains no UWF rows")

    examples: list[ModalityExample] = []
    renamed_paths = 0
    seen: set[str] = set()
    for row in rows:
        image_id = row["image_id"].strip()
        patient_id = row["patient_id"].strip()
        if not image_id or not patient_id or image_id in seen:
            raise ValueError(f"invalid or duplicate UWF row: {row}")
        seen.add(image_id)
        image_path = _resolve_uwf_image(csv_path, patient_id, image_id)
        if image_path.stem != image_id:
            renamed_paths += 1
        examples.append(
            ModalityExample(
                patient_id=patient_id,
                image_id=image_id,
                image_path=str(image_path.relative_to(PROJECT_ROOT)),
                modality="UWF",
                source_split=expected_source,
            )
        )
    return examples, renamed_paths


def patient_group_three_way_split(
    examples: list[ModalityExample],
    *,
    tuning_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split patients while stratifying on whether UWF is available."""

    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.patient_id, []).append(index)
    strata: dict[bool, list[str]] = {False: [], True: []}
    for patient_id, indices in groups.items():
        strata[
            any(examples[index].modality == "UWF" for index in indices)
        ].append(patient_id)

    rng = random.Random(seed)
    tuning_patients: set[str] = set()
    calibration_patients: set[str] = set()
    for patient_ids in strata.values():
        if len(patient_ids) < 3:
            raise ValueError("each patient stratum needs at least three patients")
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
        if example.patient_id not in tuning_patients
        and example.patient_id not in calibration_patients
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
    patient_sets = [
        {examples[index].patient_id for index in indices}
        for indices in (fit, tuning, calibration)
    ]
    if any(
        patient_sets[left] & patient_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise AssertionError("patient leakage across fit/tuning/calibration")
    for indices in (fit, tuning, calibration):
        if {examples[index].modality for index in indices} != set(MODALITIES):
            raise ValueError("each partition must contain both modalities")
    return fit, tuning, calibration


def _make_head(torch: Any, input_dim: int, hidden_dim: int) -> Any:
    if hidden_dim:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_dim, 1),
        )
    return torch.nn.Linear(input_dim, 1)


def train_member(
    *,
    fit_x: Any,
    fit_targets: Any,
    tuning_x: Any,
    tuning_targets: Any,
    input_dim: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    seed: int,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    model = _make_head(torch, input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positives = float(fit_targets.sum().item())
    negatives = float(len(fit_targets) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("fit partition requires both modalities")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives)
    )
    generator = torch.Generator().manual_seed(seed)

    best_auc = -math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    batch_size = min(128, len(fit_x))
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(fit_x), generator=generator)
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            logits = model(fit_x[indices]).squeeze(1)
            loss = loss_fn(logits, fit_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(model(tuning_x).squeeze(1)).numpy()
        tuning_auc = auc(scores.tolist(), tuning_targets.numpy().tolist())
        if tuning_auc > best_auc + 1e-6:
            best_auc = tuning_auc
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
        raise RuntimeError("UWF gate training produced no checkpoint")
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch + 1,
        "tuning_roc_auc": best_auc,
        "selection_metric": "tuning_roc_auc",
    }


def refit_member(
    *,
    development_x: Any,
    development_targets: Any,
    input_dim: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    torch: Any,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = _make_head(torch, input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positives = float(development_targets.sum().item())
    negatives = float(len(development_targets) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives)
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(128, len(development_x))
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(development_x), generator=generator)
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            logits = model(development_x[indices]).squeeze(1)
            loss = loss_fn(logits, development_targets[indices])
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
) -> Any:
    predictions = []
    for state in states:
        model = _make_head(torch, input_dim, hidden_dim)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            predictions.append(torch.sigmoid(model(features).squeeze(1)).numpy())
    return sum(predictions) / len(predictions)


def score_rows(
    examples: list[ModalityExample], scores: Iterable[float]
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": example.record_id,
            "patient_id": example.patient_id,
            "image_id": example.image_id,
            "image_path": example.image_path,
            "source_split": example.source_split,
            "truth_modality": example.modality,
            "uwf_score": float(score),
        }
        for example, score in zip(examples, scores, strict=True)
    ]


def calibrate_veto_threshold(
    calibration_rows: list[dict[str, Any]],
    *,
    false_uwf_patient_risk: float,
    delta: float,
) -> dict[str, Any]:
    """Calibrate a strict UWF threshold from conventional patients only."""

    patient_scores: dict[str, list[float]] = {}
    for row in calibration_rows:
        if row["truth_modality"] == "CONVENTIONAL_CFP":
            patient_scores.setdefault(row["patient_id"], []).append(row["uwf_score"])
    worst_scores = sorted(
        (max(scores) for scores in patient_scores.values()), reverse=True
    )
    if not worst_scores:
        raise ValueError("calibration requires conventional-CFP patients")
    allowed_errors = maximum_certified_errors(
        len(worst_scores), false_uwf_patient_risk, delta
    )
    if allowed_errors < 0:
        threshold = 1.0
        upper_bound = exact_upper_bound(0, len(worst_scores), delta)
        certified = False
    else:
        threshold = worst_scores[min(allowed_errors, len(worst_scores) - 1)]
        upper_bound = exact_upper_bound(allowed_errors, len(worst_scores), delta)
        certified = True
    calibration_flags = sum(score > threshold for score in worst_scores)
    return {
        "action_if_score_strictly_greater_than_threshold": "FORCE_LIMITED",
        "action_otherwise": "NO_CHANGE_TO_EXISTING_GATE",
        "can_promote": False,
        "threshold": float(threshold),
        "calibration_unit": "patient maximum over conventional-CFP images",
        "conventional_calibration_patients": len(worst_scores),
        "target_false_uwf_patient_risk": false_uwf_patient_risk,
        "delta": delta,
        "maximum_certified_errors": allowed_errors,
        "observed_strict_threshold_errors": calibration_flags,
        "upper_bound_at_maximum_errors": upper_bound,
        "nominal_bound_available": certified,
    }


def threshold_metrics(
    rows: list[dict[str, Any]], threshold: float, *, delta: float = 0.05
) -> dict[str, Any]:
    conventional = [row for row in rows if row["truth_modality"] == "CONVENTIONAL_CFP"]
    uwf = [row for row in rows if row["truth_modality"] == "UWF"]
    if not conventional or not uwf:
        raise ValueError("threshold evaluation requires both modalities")
    false_uwf = sum(row["uwf_score"] > threshold for row in conventional)
    detected_uwf = sum(row["uwf_score"] > threshold for row in uwf)
    missed_uwf = len(uwf) - detected_uwf

    conventional_patients: dict[str, list[dict[str, Any]]] = {}
    uwf_patients: dict[str, list[dict[str, Any]]] = {}
    for row in conventional:
        conventional_patients.setdefault(row["patient_id"], []).append(row)
    for row in uwf:
        uwf_patients.setdefault(row["patient_id"], []).append(row)
    false_uwf_patients = sum(
        any(row["uwf_score"] > threshold for row in patient_rows)
        for patient_rows in conventional_patients.values()
    )
    detected_uwf_patients = sum(
        any(row["uwf_score"] > threshold for row in patient_rows)
        for patient_rows in uwf_patients.values()
    )
    missed_uwf_patients = len(uwf_patients) - detected_uwf_patients
    specificity = 1.0 - false_uwf / len(conventional)
    recall = detected_uwf / len(uwf)
    return {
        "images": len(rows),
        "threshold": threshold,
        "conventional_images": len(conventional),
        "uwf_images": len(uwf),
        "false_uwf_count_on_conventional_images": false_uwf,
        "false_uwf_rate_on_conventional_images": false_uwf / len(conventional),
        "false_uwf_image_rate_exact_upper_95": exact_upper_bound(
            false_uwf, len(conventional), delta
        ),
        "uwf_detected_images": detected_uwf,
        "uwf_missed_images": missed_uwf,
        "uwf_recall": recall,
        "uwf_recall_exact_lower_95": 1.0
        - exact_upper_bound(missed_uwf, len(uwf), delta),
        "specificity_on_conventional": specificity,
        "balanced_accuracy": (specificity + recall) / 2.0,
        "conventional_patients": len(conventional_patients),
        "uwf_patients": len(uwf_patients),
        "false_uwf_conventional_patients": false_uwf_patients,
        "false_uwf_rate_on_conventional_patients": (
            false_uwf_patients / len(conventional_patients)
        ),
        "false_uwf_patient_rate_exact_upper_95": exact_upper_bound(
            false_uwf_patients, len(conventional_patients), delta
        ),
        "uwf_detected_patients": detected_uwf_patients,
        "uwf_missed_patients": missed_uwf_patients,
        "uwf_patient_recall": detected_uwf_patients / len(uwf_patients),
        "uwf_patient_recall_exact_lower_95": 1.0
        - exact_upper_bound(missed_uwf_patients, len(uwf_patients), delta),
    }


def ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [float(row["truth_modality"] == "UWF") for row in rows]
    scores = [row["uwf_score"] for row in rows]
    return {
        "images": len(rows),
        "uwf_positive": sum(targets),
        "conventional_negative": len(targets) - sum(targets),
        "roc_auc_uwf_positive": auc(scores, targets),
    }


def paired_patient_metrics(
    paired_conventional_rows: list[dict[str, Any]],
    uwf_rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    conventional: dict[str, list[dict[str, Any]]] = {}
    uwf: dict[str, list[dict[str, Any]]] = {}
    for row in paired_conventional_rows:
        conventional.setdefault(row["patient_id"], []).append(row)
    for row in uwf_rows:
        uwf.setdefault(row["patient_id"], []).append(row)
    patients = sorted(set(conventional) & set(uwf), key=int)
    no_cfp_veto = 0
    uwf_detected = 0
    clean_separation = 0
    for patient in patients:
        cfp_clear = not any(
            row["uwf_score"] > threshold for row in conventional[patient]
        )
        detected = any(row["uwf_score"] > threshold for row in uwf[patient])
        no_cfp_veto += cfp_clear
        uwf_detected += detected
        clean_separation += cfp_clear and detected
    return {
        "paired_patients": len(patients),
        "patients_without_conventional_false_veto": no_cfp_veto,
        "patients_with_uwf_detected": uwf_detected,
        "patients_with_clean_modality_separation": clean_separation,
        "clean_modality_separation_rate": (
            clean_separation / len(patients) if patients else None
        ),
    }


def recommendation(
    official_metrics: dict[str, Any], paired_metrics: dict[str, Any]
) -> dict[str, Any]:
    compelling = (
        official_metrics["false_uwf_rate_on_conventional_images"] <= 0.01
        and official_metrics["uwf_recall"] >= 0.95
        and paired_metrics["clean_modality_separation_rate"] is not None
        and paired_metrics["clean_modality_separation_rate"] >= 0.90
    )
    if compelling:
        return {
            "internal_evidence": "compelling-in-domain-separation",
            "runtime_integration": "do-not-integrate-yet",
            "reason": (
                "The predeclared in-domain operating targets are met, but the "
                "small UWF validation cohort and likely camera/domain shortcuts "
                "do not establish robustness to other conventional or widefield "
                "devices. Report this evidence before any runtime proposal."
            ),
        }
    return {
        "internal_evidence": "not-compelling",
        "runtime_integration": "do-not-integrate",
        "reason": (
            "The veto gate misses a predeclared operating target; keep it out of "
            "the deployed quality pipeline."
        ),
    }


def validate_sources(args: argparse.Namespace) -> None:
    paths = (
        args.regular_train_manifest,
        args.regular_val_manifest,
        args.uwf_train_csv,
        args.uwf_val_csv,
    )
    combined = " ".join(str(path).lower() for path in paths)
    if "test" in combined or "mshf" in combined:
        raise ValueError("UWF veto experiment refuses DeepDRiD test and MSHF sources")
    if args.regular_train_manifest.name != "train.csv":
        raise ValueError("regular training source must be data/manifests/train.csv")
    if args.regular_val_manifest.name != "val.csv":
        raise ValueError("regular evaluation source must be data/manifests/val.csv")
    if "ultra-widefield-training" not in str(args.uwf_train_csv):
        raise ValueError("unexpected UWF training source")
    if "ultra-widefield-validation" not in str(args.uwf_val_csv):
        raise ValueError("unexpected UWF validation source")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regular-train-manifest",
        type=Path,
        default=Path("data/manifests/train.csv"),
    )
    parser.add_argument(
        "--regular-val-manifest",
        type=Path,
        default=Path("data/manifests/val.csv"),
    )
    parser.add_argument(
        "--uwf-train-csv",
        type=Path,
        default=Path(
            "data/raw/deepdrid-v1.1/ultra-widefield_images/"
            "ultra-widefield-training/ultra-widefield-training.csv"
        ),
    )
    parser.add_argument(
        "--uwf-val-csv",
        type=Path,
        default=Path(
            "data/raw/deepdrid-v1.1/ultra-widefield_images/"
            "ultra-widefield-validation/ultra-widefield-validation.csv"
        ),
    )
    parser.add_argument(
        "--regular-cache-dir",
        type=Path,
        default=Path("ml/cache/quality-specialist"),
    )
    parser.add_argument(
        "--uwf-cache-dir", type=Path, default=Path("ml/cache/uwf-modality-gate")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/uwf-veto-gate")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tuning-fraction", type=float, default=0.12)
    parser.add_argument("--calibration-fraction", type=float, default=0.36)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--false-uwf-patient-risk", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_sources(args)
    if args.batch_size <= 0 or args.hidden_dim < 0:
        raise ValueError("batch size must be positive and hidden dim non-negative")
    if min(args.epochs, args.patience, args.ensemble_members) <= 0:
        raise ValueError("epochs, patience, and ensemble members must be positive")
    if not 0 < args.tuning_fraction < 0.5:
        raise ValueError("tuning fraction must be between zero and 0.5")
    if not 0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration fraction must be between zero and 0.5")
    if args.tuning_fraction + args.calibration_fraction >= 0.8:
        raise ValueError("split fractions leave too few fit patients")
    if not 0 < args.false_uwf_patient_risk < 1 or not 0 < args.delta < 1:
        raise ValueError("risk and delta must be between zero and one")

    import numpy as np
    import torch

    regular_train = regular_examples(args.regular_train_manifest, expected_split="train")
    regular_val = regular_examples(args.regular_val_manifest, expected_split="val")
    uwf_train, uwf_train_renamed = read_uwf_csv(
        args.uwf_train_csv, expected_source="uwf-train"
    )
    uwf_val, uwf_val_renamed = read_uwf_csv(
        args.uwf_val_csv, expected_source="uwf-val"
    )

    regular_train_patients = {example.patient_id for example in regular_train}
    regular_val_patients = {example.patient_id for example in regular_val}
    uwf_train_patients = {example.patient_id for example in uwf_train}
    uwf_val_patients = {example.patient_id for example in uwf_val}
    if uwf_train_patients & uwf_val_patients:
        raise ValueError("UWF train/validation patient overlap")
    evaluation_patients = regular_val_patients | uwf_val_patients
    paired_conventional = [
        example
        for example in regular_train
        if example.patient_id in uwf_val_patients
    ]
    conventional_development = [
        example
        for example in regular_train
        if example.patient_id not in evaluation_patients
    ]
    uwf_development = [
        example for example in uwf_train if example.patient_id not in evaluation_patients
    ]
    development_examples = [*conventional_development, *uwf_development]
    development_patients = {example.patient_id for example in development_examples}
    if development_patients & evaluation_patients:
        raise AssertionError("development/evaluation patient leakage")
    if {example.patient_id for example in paired_conventional} != uwf_val_patients:
        raise ValueError("paired conventional holdout does not cover every UWF validation patient")

    device = choose_device(torch, args.device)
    feature_started = time.perf_counter()
    regular_train_features = extract_features(
        regular_train,
        manifest_path=args.regular_train_manifest,
        cache_dir=args.regular_cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    regular_val_features = extract_features(
        regular_val,
        manifest_path=args.regular_val_manifest,
        cache_dir=args.regular_cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    uwf_train_features = extract_features(
        uwf_train,
        manifest_path=args.uwf_train_csv,
        cache_dir=args.uwf_cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    uwf_val_features = extract_features(
        uwf_val,
        manifest_path=args.uwf_val_csv,
        cache_dir=args.uwf_cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    feature_seconds = round(time.perf_counter() - feature_started, 3)

    regular_train_index = {
        example.record_id: index for index, example in enumerate(regular_train)
    }
    conventional_development_features = np.stack(
        [regular_train_features[regular_train_index[e.record_id]] for e in conventional_development]
    )
    paired_conventional_features = np.stack(
        [regular_train_features[regular_train_index[e.record_id]] for e in paired_conventional]
    )
    development_features = np.concatenate(
        (conventional_development_features, uwf_train_features), axis=0
    ).astype("float32", copy=False)

    fit_indices, tuning_indices, calibration_indices = patient_group_three_way_split(
        development_examples,
        tuning_fraction=args.tuning_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    selection_mean = development_features[fit_indices].mean(axis=0, keepdims=True)
    selection_std = development_features[fit_indices].std(axis=0, keepdims=True)
    selection_std[selection_std < 1e-6] = 1.0
    selection_standardized = (
        development_features - selection_mean
    ) / selection_std
    targets = np.asarray(
        [example.target for example in development_examples], dtype="float32"
    )
    fit_x = torch.from_numpy(selection_standardized[fit_indices]).float()
    tuning_x = torch.from_numpy(selection_standardized[tuning_indices]).float()
    fit_targets = torch.from_numpy(targets[fit_indices]).float()
    tuning_targets = torch.from_numpy(targets[tuning_indices]).float()

    members: list[dict[str, Any]] = []
    for member in range(args.ensemble_members):
        _, metadata = train_member(
            fit_x=fit_x,
            fit_targets=fit_targets,
            tuning_x=tuning_x,
            tuning_targets=tuning_targets,
            input_dim=development_features.shape[1],
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + member,
            torch=torch,
        )
        members.append(metadata)
        print(f"Selected UWF member {member + 1}: {metadata}")

    development_indices = [*fit_indices, *tuning_indices]
    feature_mean = development_features[development_indices].mean(axis=0, keepdims=True)
    feature_std = development_features[development_indices].std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    standardized_development = (
        development_features - feature_mean
    ) / feature_std
    states: list[dict[str, Any]] = []
    development_x = torch.from_numpy(
        standardized_development[development_indices]
    ).float()
    development_targets = torch.from_numpy(targets[development_indices]).float()
    for member, metadata in enumerate(members):
        states.append(
            refit_member(
                development_x=development_x,
                development_targets=development_targets,
                input_dim=development_features.shape[1],
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=metadata["best_epoch"],
                seed=args.seed + member,
                torch=torch,
            )
        )

    def standardized(array: Any) -> Any:
        return torch.from_numpy(((array - feature_mean) / feature_std).astype("float32"))

    calibration_examples = [development_examples[index] for index in calibration_indices]
    calibration_scores = predict_ensemble(
        standardized(development_features[calibration_indices]),
        states=states,
        input_dim=development_features.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    regular_val_scores = predict_ensemble(
        standardized(regular_val_features),
        states=states,
        input_dim=development_features.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    uwf_val_scores = predict_ensemble(
        standardized(uwf_val_features),
        states=states,
        input_dim=development_features.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    paired_scores = predict_ensemble(
        standardized(paired_conventional_features),
        states=states,
        input_dim=development_features.shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )

    calibration_rows = score_rows(calibration_examples, calibration_scores)
    regular_val_rows = score_rows(regular_val, regular_val_scores)
    uwf_val_rows = score_rows(uwf_val, uwf_val_scores)
    paired_rows = score_rows(paired_conventional, paired_scores)
    policy = calibrate_veto_threshold(
        calibration_rows,
        false_uwf_patient_risk=args.false_uwf_patient_risk,
        delta=args.delta,
    )
    threshold = policy["threshold"]
    official_rows = [*regular_val_rows, *uwf_val_rows]
    official_metrics = threshold_metrics(official_rows, threshold, delta=args.delta)
    paired_eval_rows = [*paired_rows, *uwf_val_rows]
    paired_threshold_metrics = threshold_metrics(
        paired_eval_rows, threshold, delta=args.delta
    )
    paired_metrics = paired_patient_metrics(paired_rows, uwf_val_rows, threshold)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment": "DeepDRiD UWF-vs-conventional-CFP veto-only modality gate",
        "status": "experimental-only; not integrated",
        "policy_semantics": {
            "positive_class": "UWF",
            "if_positive": "force LIMITED",
            "if_negative": "leave existing quality-gate decision unchanged",
            "can_promote_ready": False,
            "can_promote_retake": False,
        },
        "data": {
            "regular_train_manifest": str(args.regular_train_manifest),
            "regular_train_sha256": sha256_file(resolve(args.regular_train_manifest)),
            "regular_val_manifest": str(args.regular_val_manifest),
            "regular_val_sha256": sha256_file(resolve(args.regular_val_manifest)),
            "uwf_train_csv": str(args.uwf_train_csv),
            "uwf_train_sha256": sha256_file(resolve(args.uwf_train_csv)),
            "uwf_val_csv": str(args.uwf_val_csv),
            "uwf_val_sha256": sha256_file(resolve(args.uwf_val_csv)),
            "regular_train_images": len(regular_train),
            "regular_val_images": len(regular_val),
            "uwf_train_images": len(uwf_train),
            "uwf_val_images": len(uwf_val),
            "uwf_train_patients": len(uwf_train_patients),
            "uwf_val_patients": len(uwf_val_patients),
            "regular_training_images_excluded_for_uwf_validation_patients": len(
                paired_conventional
            ),
            "regular_training_patients_excluded_for_uwf_validation": len(
                uwf_val_patients
            ),
            "development_evaluation_patient_overlap": 0,
            "paired_holdout_patients": len(uwf_val_patients),
            "uwf_paths_resolved_from_local_rename": uwf_train_renamed + uwf_val_renamed,
            "deepdrid_test_used": False,
            "mshf_used": False,
        },
        "development_split": {
            "fit_images": len(fit_indices),
            "fit_patients": len(
                {development_examples[index].patient_id for index in fit_indices}
            ),
            "tuning_images": len(tuning_indices),
            "tuning_patients": len(
                {development_examples[index].patient_id for index in tuning_indices}
            ),
            "calibration_images": len(calibration_indices),
            "calibration_patients": len(
                {development_examples[index].patient_id for index in calibration_indices}
            ),
            "patient_overlap": 0,
        },
        "model": {
            "architecture": "frozen DenseNet-121 global features plus MLP ensemble",
            "input_dim": int(development_features.shape[1]),
            "hidden_dim": args.hidden_dim,
            "ensemble_members": args.ensemble_members,
            "backbone_sha256": hashlib.sha256(
                (
                    PROJECT_ROOT
                    / "models/retinaready-quality-specialist/densenet121-a639ec97.pth"
                ).read_bytes()
            ).hexdigest(),
            "feature_device": device,
            "feature_seconds_including_regular_cache_load": feature_seconds,
            "members": members,
        },
        "calibrated_veto_policy": policy,
        "calibration_ranking": ranking_metrics(calibration_rows),
        "calibration_threshold_metrics": threshold_metrics(
            calibration_rows, threshold, delta=args.delta
        ),
        "official_validation": {
            "scope": "regular official validation + UWF official validation",
            "ranking": ranking_metrics(official_rows),
            "threshold_metrics": official_metrics,
        },
        "paired_conventional_holdout": {
            "scope": (
                "regular training images from UWF-validation patients excluded "
                "entirely from development, paired with their UWF validation images"
            ),
            "ranking": ranking_metrics(paired_eval_rows),
            "threshold_metrics": paired_threshold_metrics,
            "patient_pair_metrics": paired_metrics,
        },
        "predeclared_operating_targets": {
            "maximum_false_uwf_rate_on_conventional_images": 0.01,
            "minimum_uwf_recall": 0.95,
            "minimum_paired_patient_clean_separation": 0.90,
        },
        "recommendation": recommendation(official_metrics, paired_metrics),
        "limitations": [
            "This is a modality safety-veto experiment, not diagnosis or clinical validation.",
            "The gate may learn device, border, color, resolution, or cohort shortcuts rather than a general UWF concept.",
            "Only 25 patients and 50 UWF images are available in the official UWF validation split.",
            "Official regular and UWF validation cohorts are different patients; the paired holdout partially probes this confound but comes from the regular training directory.",
            "The local UWF training CSV contains 154 images from 77 patients although its readme says 152 images from 76 patients; the experiment follows the CSV and verifies every file.",
            "Risk calibration is internal and historically non-fresh, so its finite-sample bound is nominal rather than a deployment guarantee.",
            "No DeepDRiD test or MSHF data was used, and no artifact is installed into the runtime.",
        ],
        "calibration_results": calibration_rows,
        "official_validation_results": official_rows,
        "paired_conventional_results": paired_rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "schema_version": 1,
            "experimental_only": True,
            "runtime_integration_authorized": False,
            "architecture": payload["model"]["architecture"],
            "input_dim": int(development_features.shape[1]),
            "hidden_dim": args.hidden_dim,
            "members": states,
            "feature_mean": torch.from_numpy(feature_mean.astype("float32")),
            "feature_std": torch.from_numpy(feature_std.astype("float32")),
            "veto_policy": policy,
        },
        output_dir / "uwf-veto-experiment.pt",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "official_validation": payload["official_validation"],
                "paired_conventional_holdout": payload["paired_conventional_holdout"],
                "recommendation": payload["recommendation"],
                "report": str(report_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
