#!/usr/bin/env python3
"""Train and evaluate a conservative retinal review-priority baseline.

The model is a lightweight ensemble over frozen ImageNet DenseNet-121 global
features.  It predicts a ranking score for the derived DeepDRiD review-priority
target, then a patient-grouped calibration cohort supplies two strict
thresholds:

* below the lower threshold -> ROUTINE
* above the upper threshold -> PRIORITY
* otherwise -> UNCERTAIN

These are workflow priorities, not diagnoses, disease exclusions, treatment
recommendations, or clinical validation.  The script refuses test, MSHF, and
UWF inputs and never modifies RetinaReady's deployed quality pipeline or UI.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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
TRUTH_LABELS = ("ROUTINE", "PRIORITY")
DECISIONS = ("ROUTINE", "PRIORITY", "UNCERTAIN")
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


@dataclass(frozen=True)
class EscalationExample:
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

    @property
    def priority_target(self) -> float:
        return float(self.escalation_label == "PRIORITY")


def validate_sources(paths: Iterable[Path]) -> None:
    combined = " ".join(str(path).lower() for path in paths)
    if any(forbidden in combined for forbidden in ("test", "mshf", "widefield")):
        raise ValueError("escalation experiment refuses test, MSHF, and UWF sources")


def read_escalation_manifest(path: Path, *, expected_split: str) -> list[EscalationExample]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{resolved} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{resolved} contains no rows")

    examples: list[EscalationExample] = []
    seen: set[str] = set()
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(
                f"{resolved}: expected split={expected_split!r}, got {row['split']!r}"
            )
        if row["image_id"] in seen:
            raise ValueError(f"{resolved}: duplicate image ID {row['image_id']}")
        seen.add(row["image_id"])
        if row["escalation_label"] not in TRUTH_LABELS:
            raise ValueError(f"{row['image_id']}: invalid escalation label")
        if row["dr_grade"] not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"{row['image_id']}: invalid DR grade")
        grade = int(row["dr_grade"])
        expected_label = "PRIORITY" if grade >= 2 else "ROUTINE"
        if row["escalation_label"] != expected_label:
            raise ValueError(f"{row['image_id']}: grade/priority mapping mismatch")
        if row["overall_quality"] not in {"0", "1"}:
            raise ValueError(f"{row['image_id']}: invalid overall quality")
        if row["grade_source_field"] not in {
            "left_eye_DR_Level",
            "right_eye_DR_Level",
        }:
            raise ValueError(f"{row['image_id']}: invalid grade source field")
        if row["filename_side_matches_grade_field"] not in {"true", "false"}:
            raise ValueError(f"{row['image_id']}: invalid side audit flag")
        image_path = resolve(Path(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        examples.append(
            EscalationExample(
                split=row["split"],
                patient_id=row["patient_id"],
                image_id=row["image_id"],
                image_path=row["image_path"],
                dr_grade=grade,
                escalation_label=row["escalation_label"],
                overall_quality=int(row["overall_quality"]),
                source_split=row["source_split"],
                grade_source_field=row["grade_source_field"],
                filename_side_matches_grade_field=(
                    row["filename_side_matches_grade_field"] == "true"
                ),
            )
        )
    if {example.escalation_label for example in examples} != set(TRUTH_LABELS):
        raise ValueError(f"{resolved}: both priority classes are required")
    return examples


def assert_patient_disjoint(partitions: dict[str, list[EscalationExample]]) -> dict[str, int]:
    patients = {
        split: {example.patient_id for example in examples}
        for split, examples in partitions.items()
    }
    overlaps = {
        f"{left}_{right}": len(patients[left] & patients[right])
        for index, left in enumerate(partitions)
        for right in list(partitions)[index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"patient leakage across escalation partitions: {overlaps}")
    return overlaps


def make_head(torch: Any, input_dim: int, hidden_dim: int) -> Any:
    if hidden_dim:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_dim, 1),
        )
    return torch.nn.Linear(input_dim, 1)


def patient_ranking(
    examples: list[EscalationExample], scores: Iterable[float]
) -> dict[str, Any]:
    patients: dict[str, list[tuple[EscalationExample, float]]] = {}
    for example, score in zip(examples, scores, strict=True):
        patients.setdefault(example.patient_id, []).append((example, float(score)))
    patient_scores = [max(score for _, score in rows) for rows in patients.values()]
    patient_targets = [
        float(any(example.escalation_label == "PRIORITY" for example, _ in rows))
        for rows in patients.values()
    ]
    return {
        "patients": len(patients),
        "priority_patients": int(sum(patient_targets)),
        "routine_patients": len(patient_targets) - int(sum(patient_targets)),
        "roc_auc_priority_positive": auc(patient_scores, patient_targets),
    }


def train_member(
    *,
    train_x: Any,
    train_targets: Any,
    tuning_x: Any,
    tuning_targets: Any,
    tuning_examples: list[EscalationExample],
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
    model = make_head(torch, input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positives = float(train_targets.sum().item())
    negatives = float(len(train_targets) - positives)
    if min(positives, negatives) <= 0:
        raise ValueError("training requires both priority classes")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives)
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(128, len(train_x))

    best_score = -math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            logits = model(train_x[indices]).squeeze(1)
            loss = loss_fn(logits, train_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(model(tuning_x).squeeze(1)).numpy()
        selection_score = patient_ranking(tuning_examples, scores)[
            "roc_auc_priority_positive"
        ]
        if selection_score > best_score + 1e-6:
            best_score = selection_score
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
        raise RuntimeError("escalation training produced no checkpoint")
    return best_state, {
        "seed": seed,
        "best_epoch": best_epoch + 1,
        "selection_metric": "tuning_patient_roc_auc",
        "selection_score": best_score,
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
    model = make_head(torch, input_dim, hidden_dim)
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
        model = make_head(torch, input_dim, hidden_dim)
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            predictions.append(torch.sigmoid(model(features).squeeze(1)).numpy())
    return sum(predictions) / len(predictions)


def score_rows(
    examples: list[EscalationExample], scores: Iterable[float]
) -> list[dict[str, Any]]:
    return [
        {
            "split": example.split,
            "patient_id": example.patient_id,
            "image_id": example.image_id,
            "image_path": example.image_path,
            "dr_grade": example.dr_grade,
            "truth_review_priority": example.escalation_label,
            "overall_quality": example.overall_quality,
            "source_split": example.source_split,
            "review_priority_score": float(score),
        }
        for example, score in zip(examples, scores, strict=True)
    ]


def calibrate_thresholds(
    rows: list[dict[str, Any]],
    *,
    false_routine_risk: float,
    false_priority_risk: float,
    delta: float,
) -> dict[str, Any]:
    """Calibrate strict thresholds from independent patient-level events."""

    patients: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        patients.setdefault(row["patient_id"], []).append(row)
    priority_patient_minima = sorted(
        min(
            row["review_priority_score"]
            for row in patient_rows
            if row["truth_review_priority"] == "PRIORITY"
        )
        for patient_rows in patients.values()
        if any(row["truth_review_priority"] == "PRIORITY" for row in patient_rows)
    )
    routine_patient_maxima = sorted(
        (
            max(
                row["review_priority_score"]
                for row in patient_rows
                if row["truth_review_priority"] == "ROUTINE"
            )
            for patient_rows in patients.values()
            if any(row["truth_review_priority"] == "ROUTINE" for row in patient_rows)
        ),
        reverse=True,
    )
    if not priority_patient_minima or not routine_patient_maxima:
        raise ValueError("calibration requires both review-priority classes")

    false_routine_errors = maximum_certified_errors(
        len(priority_patient_minima), false_routine_risk, delta
    )
    false_priority_errors = maximum_certified_errors(
        len(routine_patient_maxima), false_priority_risk, delta
    )
    false_routine_constraint = (
        priority_patient_minima[false_routine_errors]
        if false_routine_errors >= 0
        else 0.0
    )
    false_priority_constraint = (
        routine_patient_maxima[false_priority_errors]
        if false_priority_errors >= 0
        else 1.0
    )
    # Each calibrated constraint points toward the largest acceptance region
    # for one class.  If those regions overlap, accepting both would make the
    # decision order-dependent.  Widen the abstention interval to the outer
    # constraints instead.  This can only remove accepted decisions, so it
    # cannot add calibration errors for either adverse event.
    routine_threshold = min(false_routine_constraint, false_priority_constraint)
    priority_threshold = max(false_routine_constraint, false_priority_constraint)
    return {
        "semantics": "review priority only; not diagnosis, disease exclusion, or treatment advice",
        "calibration_unit": "patient adverse event over image decisions",
        "routine_if_score_strictly_less_than": routine_threshold,
        "priority_if_score_strictly_greater_than": priority_threshold,
        "otherwise": "UNCERTAIN",
        "overlap_resolution": (
            "use the outer calibrated constraints as a fail-closed abstention interval"
        ),
        "raw_false_routine_constraint": false_routine_constraint,
        "raw_false_priority_constraint": false_priority_constraint,
        "per_gate_delta": delta,
        "simultaneous_confidence_lower_bound": max(0.0, 1.0 - 2.0 * delta),
        "false_routine": {
            "risk_limit": false_routine_risk,
            "event": "any PRIORITY-truth image for a patient is called ROUTINE",
            "calibration_patients": len(priority_patient_minima),
            "maximum_certified_errors": false_routine_errors,
            "observed_strict_threshold_errors": sum(
                score < routine_threshold for score in priority_patient_minima
            ),
            "upper_bound_at_maximum_errors": (
                exact_upper_bound(
                    false_routine_errors, len(priority_patient_minima), delta
                )
                if false_routine_errors >= 0
                else None
            ),
        },
        "false_priority": {
            "risk_limit": false_priority_risk,
            "event": "any ROUTINE-truth image for a patient is called PRIORITY",
            "calibration_patients": len(routine_patient_maxima),
            "maximum_certified_errors": false_priority_errors,
            "observed_strict_threshold_errors": sum(
                score > priority_threshold for score in routine_patient_maxima
            ),
            "upper_bound_at_maximum_errors": (
                exact_upper_bound(
                    false_priority_errors, len(routine_patient_maxima), delta
                )
                if false_priority_errors >= 0
                else None
            ),
        },
    }


def assign(score: float, policy: dict[str, Any]) -> str:
    if score < policy["routine_if_score_strictly_less_than"]:
        return "ROUTINE"
    if score > policy["priority_if_score_strictly_greater_than"]:
        return "PRIORITY"
    return "UNCERTAIN"


def ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["review_priority_score"] for row in rows]
    targets = [float(row["truth_review_priority"] == "PRIORITY") for row in rows]
    return {
        "images": len(rows),
        "priority_images": int(sum(targets)),
        "routine_images": len(targets) - int(sum(targets)),
        "roc_auc_priority_positive": auc(scores, targets),
    }


def selective_metrics(
    rows: list[dict[str, Any]], policy: dict[str, Any], *, delta: float
) -> dict[str, Any]:
    decisions = [assign(row["review_priority_score"], policy) for row in rows]
    truth_counts = {
        label: sum(row["truth_review_priority"] == label for row in rows)
        for label in TRUTH_LABELS
    }
    if not all(truth_counts.values()):
        raise ValueError("selective evaluation requires both truth classes")
    decision_counts = {
        decision: sum(value == decision for value in decisions)
        for decision in DECISIONS
    }
    false_routine = sum(
        row["truth_review_priority"] == "PRIORITY" and decision == "ROUTINE"
        for row, decision in zip(rows, decisions, strict=True)
    )
    false_priority = sum(
        row["truth_review_priority"] == "ROUTINE" and decision == "PRIORITY"
        for row, decision in zip(rows, decisions, strict=True)
    )
    true_routine = sum(
        row["truth_review_priority"] == "ROUTINE" and decision == "ROUTINE"
        for row, decision in zip(rows, decisions, strict=True)
    )
    true_priority = sum(
        row["truth_review_priority"] == "PRIORITY" and decision == "PRIORITY"
        for row, decision in zip(rows, decisions, strict=True)
    )
    accepted = decision_counts["ROUTINE"] + decision_counts["PRIORITY"]
    correct = true_routine + true_priority
    priority_missed_or_uncertain = truth_counts["PRIORITY"] - true_priority
    routine_missed_or_uncertain = truth_counts["ROUTINE"] - true_routine
    return {
        "images": len(rows),
        "truth_counts": truth_counts,
        "decision_counts": decision_counts,
        "coverage": accepted / len(rows),
        "accepted_accuracy": correct / accepted if accepted else None,
        "false_routine_count": false_routine,
        "false_routine_rate_given_priority": false_routine / truth_counts["PRIORITY"],
        "false_routine_rate_exact_upper_95": exact_upper_bound(
            false_routine, truth_counts["PRIORITY"], delta
        ),
        "false_priority_count": false_priority,
        "false_priority_rate_given_routine": false_priority / truth_counts["ROUTINE"],
        "false_priority_rate_exact_upper_95": exact_upper_bound(
            false_priority, truth_counts["ROUTINE"], delta
        ),
        "priority_recall": true_priority / truth_counts["PRIORITY"],
        "priority_recall_exact_lower_95": 1.0
        - exact_upper_bound(priority_missed_or_uncertain, truth_counts["PRIORITY"], delta),
        "routine_recall": true_routine / truth_counts["ROUTINE"],
        "routine_recall_exact_lower_95": 1.0
        - exact_upper_bound(routine_missed_or_uncertain, truth_counts["ROUTINE"], delta),
        "priority_precision": (
            true_priority / decision_counts["PRIORITY"]
            if decision_counts["PRIORITY"]
            else None
        ),
        "routine_precision": (
            true_routine / decision_counts["ROUTINE"]
            if decision_counts["ROUTINE"]
            else None
        ),
    }


def patient_event_metrics(
    rows: list[dict[str, Any]], policy: dict[str, Any], *, delta: float
) -> dict[str, Any]:
    patients: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        patients.setdefault(row["patient_id"], []).append(row)
    priority_patients = [
        patient_rows
        for patient_rows in patients.values()
        if any(row["truth_review_priority"] == "PRIORITY" for row in patient_rows)
    ]
    routine_image_patients = [
        patient_rows
        for patient_rows in patients.values()
        if any(row["truth_review_priority"] == "ROUTINE" for row in patient_rows)
    ]
    false_routine_events = sum(
        any(
            row["truth_review_priority"] == "PRIORITY"
            and assign(row["review_priority_score"], policy) == "ROUTINE"
            for row in patient_rows
        )
        for patient_rows in priority_patients
    )
    false_priority_events = sum(
        any(
            row["truth_review_priority"] == "ROUTINE"
            and assign(row["review_priority_score"], policy) == "PRIORITY"
            for row in patient_rows
        )
        for patient_rows in routine_image_patients
    )
    return {
        "patients": len(patients),
        "priority_truth_patients": len(priority_patients),
        "patients_with_routine_truth_images": len(routine_image_patients),
        "false_routine_patient_events": false_routine_events,
        "false_routine_patient_event_rate": false_routine_events
        / len(priority_patients),
        "false_routine_patient_event_exact_upper_95": exact_upper_bound(
            false_routine_events, len(priority_patients), delta
        ),
        "false_priority_patient_events": false_priority_events,
        "false_priority_patient_event_rate": false_priority_events
        / len(routine_image_patients),
        "false_priority_patient_event_exact_upper_95": exact_upper_bound(
            false_priority_events, len(routine_image_patients), delta
        ),
    }


def patient_aggregate_metrics(
    rows: list[dict[str, Any]], policy: dict[str, Any], *, delta: float
) -> dict[str, Any]:
    patients: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        patients.setdefault(row["patient_id"], []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for patient_id, patient_rows in patients.items():
        truth = (
            "PRIORITY"
            if any(row["truth_review_priority"] == "PRIORITY" for row in patient_rows)
            else "ROUTINE"
        )
        image_decisions = [
            assign(row["review_priority_score"], policy) for row in patient_rows
        ]
        if "PRIORITY" in image_decisions:
            decision = "PRIORITY"
        elif "UNCERTAIN" in image_decisions:
            decision = "UNCERTAIN"
        else:
            decision = "ROUTINE"
        aggregate_rows.append(
            {
                "patient_id": patient_id,
                "truth_review_priority": truth,
                "review_priority_score": max(
                    row["review_priority_score"] for row in patient_rows
                ),
                "decision": decision,
            }
        )
    truth_counts = {
        label: sum(row["truth_review_priority"] == label for row in aggregate_rows)
        for label in TRUTH_LABELS
    }
    decision_counts = {
        decision: sum(row["decision"] == decision for row in aggregate_rows)
        for decision in DECISIONS
    }
    false_routine = sum(
        row["truth_review_priority"] == "PRIORITY" and row["decision"] == "ROUTINE"
        for row in aggregate_rows
    )
    false_priority = sum(
        row["truth_review_priority"] == "ROUTINE" and row["decision"] == "PRIORITY"
        for row in aggregate_rows
    )
    correct = sum(
        row["decision"] == row["truth_review_priority"] for row in aggregate_rows
    )
    accepted = decision_counts["ROUTINE"] + decision_counts["PRIORITY"]
    return {
        "patients": len(aggregate_rows),
        "aggregation": "PRIORITY if any image is PRIORITY; else UNCERTAIN if any image is UNCERTAIN; else ROUTINE",
        "truth_counts": truth_counts,
        "decision_counts": decision_counts,
        "coverage": accepted / len(aggregate_rows),
        "accepted_accuracy": correct / accepted if accepted else None,
        "false_routine_count": false_routine,
        "false_routine_rate_given_priority": false_routine / truth_counts["PRIORITY"],
        "false_routine_rate_exact_upper_95": exact_upper_bound(
            false_routine, truth_counts["PRIORITY"], delta
        ),
        "false_priority_count": false_priority,
        "false_priority_rate_given_routine": false_priority / truth_counts["ROUTINE"],
        "false_priority_rate_exact_upper_95": exact_upper_bound(
            false_priority, truth_counts["ROUTINE"], delta
        ),
    }


def result_distribution_by_grade(
    rows: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for grade in range(5):
        grade_rows = [row for row in rows if row["dr_grade"] == grade]
        output[str(grade)] = {
            "images": len(grade_rows),
            "decisions": {
                decision: sum(
                    assign(row["review_priority_score"], policy) == decision
                    for row in grade_rows
                )
                for decision in DECISIONS
            },
        }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("data/escalation-manifests/train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/escalation-manifests/val.csv"))
    parser.add_argument("--calibration-manifest", type=Path, default=Path("data/escalation-manifests/calibration.csv"))
    parser.add_argument("--eval-manifest", type=Path, default=Path("data/escalation-manifests/eval.csv"))
    parser.add_argument("--manifest-summary", type=Path, default=Path("data/escalation-manifests/summary.json"))
    parser.add_argument("--base-train-manifest", type=Path, default=Path("data/manifests/train.csv"))
    parser.add_argument("--base-eval-manifest", type=Path, default=Path("data/manifests/val.csv"))
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("ml/cache/quality-specialist"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/escalation-baseline"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--false-routine-risk", type=float, default=0.05)
    parser.add_argument("--false-priority-risk", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_paths = [
        args.train_manifest,
        args.val_manifest,
        args.calibration_manifest,
        args.eval_manifest,
        args.base_train_manifest,
        args.base_eval_manifest,
    ]
    validate_sources(manifest_paths)
    if args.batch_size <= 0 or args.hidden_dim < 0:
        raise ValueError("batch size must be positive and hidden dim non-negative")
    if min(args.epochs, args.patience, args.ensemble_members) <= 0:
        raise ValueError("epochs, patience, and ensemble members must be positive")
    if not 0 < args.false_routine_risk < 1:
        raise ValueError("false ROUTINE risk must be between zero and one")
    if not 0 < args.false_priority_risk < 1:
        raise ValueError("false PRIORITY risk must be between zero and one")
    if not 0 < args.delta < 0.5:
        raise ValueError("delta must be between zero and 0.5")

    import numpy as np
    import torch

    partitions = {
        "train": read_escalation_manifest(args.train_manifest, expected_split="train"),
        "val": read_escalation_manifest(args.val_manifest, expected_split="val"),
        "calibration": read_escalation_manifest(
            args.calibration_manifest, expected_split="calibration"
        ),
        "eval": read_escalation_manifest(args.eval_manifest, expected_split="eval"),
    }
    overlaps = assert_patient_disjoint(partitions)
    if any(example.source_split != "regular-fundus-training" for split in ("train", "val", "calibration") for example in partitions[split]):
        raise ValueError("development partitions must come only from official training")
    if any(example.source_split != "regular-fundus-validation" for example in partitions["eval"]):
        raise ValueError("evaluation must come only from official validation")

    base_train = read_manifest(args.base_train_manifest, expected_split="train")
    base_eval = read_manifest(args.base_eval_manifest, expected_split="val")
    device = choose_device(torch, args.device)
    feature_started = time.perf_counter()
    base_train_features = extract_features(
        base_train,
        manifest_path=args.base_train_manifest,
        cache_dir=args.feature_cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    base_eval_features = extract_features(
        base_eval,
        manifest_path=args.base_eval_manifest,
        cache_dir=args.feature_cache_dir,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
        np=np,
    )
    feature_seconds = round(time.perf_counter() - feature_started, 3)
    base_train_index = {
        example.image_id: index for index, example in enumerate(base_train)
    }
    base_eval_index = {
        example.image_id: index for index, example in enumerate(base_eval)
    }

    feature_arrays: dict[str, Any] = {}
    for split in ("train", "val", "calibration"):
        feature_arrays[split] = np.stack(
            [base_train_features[base_train_index[e.image_id]] for e in partitions[split]]
        ).astype("float32", copy=False)
    feature_arrays["eval"] = np.stack(
        [base_eval_features[base_eval_index[e.image_id]] for e in partitions["eval"]]
    ).astype("float32", copy=False)

    train_mean = feature_arrays["train"].mean(axis=0, keepdims=True)
    train_std = feature_arrays["train"].std(axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0
    train_x = torch.from_numpy((feature_arrays["train"] - train_mean) / train_std).float()
    val_x = torch.from_numpy((feature_arrays["val"] - train_mean) / train_std).float()
    train_targets = torch.tensor(
        [example.priority_target for example in partitions["train"]], dtype=torch.float32
    )
    val_targets = torch.tensor(
        [example.priority_target for example in partitions["val"]], dtype=torch.float32
    )

    member_metadata: list[dict[str, Any]] = []
    selection_states: list[dict[str, Any]] = []
    for member in range(args.ensemble_members):
        selection_state, metadata = train_member(
            train_x=train_x,
            train_targets=train_targets,
            tuning_x=val_x,
            tuning_targets=val_targets,
            tuning_examples=partitions["val"],
            input_dim=feature_arrays["train"].shape[1],
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + member,
            torch=torch,
        )
        selection_states.append(selection_state)
        member_metadata.append(metadata)
        print(f"Selected escalation member {member + 1}: {metadata}")

    val_selection_scores = predict_ensemble(
        val_x,
        states=selection_states,
        input_dim=feature_arrays["train"].shape[1],
        hidden_dim=args.hidden_dim,
        torch=torch,
    )
    val_selection_rows = score_rows(partitions["val"], val_selection_scores)

    development_examples = [*partitions["train"], *partitions["val"]]
    development_features = np.concatenate(
        (feature_arrays["train"], feature_arrays["val"]), axis=0
    )
    feature_mean = development_features.mean(axis=0, keepdims=True)
    feature_std = development_features.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    development_x = torch.from_numpy(
        ((development_features - feature_mean) / feature_std).astype("float32")
    )
    development_targets = torch.tensor(
        [example.priority_target for example in development_examples],
        dtype=torch.float32,
    )
    states = [
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
        for member, metadata in enumerate(member_metadata)
    ]

    def standardized(array: Any) -> Any:
        return torch.from_numpy(((array - feature_mean) / feature_std).astype("float32"))

    scores = {
        split: predict_ensemble(
            standardized(feature_arrays[split]),
            states=states,
            input_dim=development_features.shape[1],
            hidden_dim=args.hidden_dim,
            torch=torch,
        )
        for split in partitions
    }
    rows = {
        split: score_rows(partitions[split], scores[split]) for split in partitions
    }
    policy = calibrate_thresholds(
        rows["calibration"],
        false_routine_risk=args.false_routine_risk,
        false_priority_risk=args.false_priority_risk,
        delta=args.delta,
    )
    for split_rows in rows.values():
        for row in split_rows:
            row["decision"] = assign(row["review_priority_score"], policy)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = json.loads(resolve(args.manifest_summary).read_text())
    checkpoint = PROJECT_ROOT / "models/retinaready-quality-specialist/densenet121-a639ec97.pth"
    eval_rows = rows["eval"]
    report = {
        "schema_version": 1,
        "experiment": "standalone conservative retinal review-priority baseline",
        "status": "experimental-only; not integrated",
        "semantics": {
            "ROUTINE": "lower review priority; not a finding of no disease",
            "PRIORITY": "higher review priority; not a diagnosis or treatment recommendation",
            "UNCERTAIN": "abstain and route for human review",
            "truth_mapping": "DeepDRiD eye grades 0-1 -> ROUTINE; grades 2-4 -> PRIORITY",
        },
        "data_and_license_audit": {
            "manifest_summary": str(args.manifest_summary),
            "manifest_summary_sha256": sha256_file(resolve(args.manifest_summary)),
            "license": summary_payload["license"],
            "grade_derivation": summary_payload["grade_derivation"],
            "partitions": summary_payload["partitions"],
            "patient_overlap": overlaps,
            "official_validation_used_only_for_final_evaluation": True,
            "test_used": False,
            "mshf_used": False,
            "uwf_used": False,
        },
        "model": {
            "architecture": (
                "frozen ImageNet DenseNet-121 1024-D global features plus "
                f"{args.ensemble_members}-member MLP ensemble"
            ),
            "input_dim": int(development_features.shape[1]),
            "hidden_dim": args.hidden_dim,
            "ensemble_members": args.ensemble_members,
            "backbone_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "feature_device": device,
            "feature_seconds_including_cache_load": feature_seconds,
            "members": member_metadata,
            "model_selection_data": "escalation val manifest only",
            "threshold_selection_data": "escalation calibration manifest only",
        },
        "policy": policy,
        "internal_validation_selection_ranking": {
            "note": "train-only member checkpoints scored before refitting on train plus val",
            "image": ranking_metrics(val_selection_rows),
            "patient": patient_ranking(partitions["val"], val_selection_scores),
        },
        "calibration": {
            "ranking": ranking_metrics(rows["calibration"]),
            "selective_image_metrics": selective_metrics(
                rows["calibration"], policy, delta=args.delta
            ),
            "patient_adverse_event_metrics": patient_event_metrics(
                rows["calibration"], policy, delta=args.delta
            ),
        },
        "official_validation_evaluation": {
            "ranking": ranking_metrics(eval_rows),
            "patient_ranking": patient_ranking(partitions["eval"], scores["eval"]),
            "selective_image_metrics": selective_metrics(
                eval_rows, policy, delta=args.delta
            ),
            "patient_adverse_event_metrics": patient_event_metrics(
                eval_rows, policy, delta=args.delta
            ),
            "patient_aggregate_metrics": patient_aggregate_metrics(
                eval_rows, policy, delta=args.delta
            ),
            "by_overall_quality": {
                str(quality): {
                    "ranking": ranking_metrics(
                        [row for row in eval_rows if row["overall_quality"] == quality]
                    ),
                    "selective_image_metrics": selective_metrics(
                        [row for row in eval_rows if row["overall_quality"] == quality],
                        policy,
                        delta=args.delta,
                    ),
                }
                for quality in (0, 1)
            },
            "decision_distribution_by_dr_grade": result_distribution_by_grade(
                eval_rows, policy
            ),
        },
        "recommendation": {
            "runtime_integration": "do-not-integrate",
            "clinical_use": "not-authorized",
            "next_evidence_needed": (
                "external multi-device patient-disjoint evaluation, prospective workflow "
                "testing, clinician review of the priority mapping, and fresh calibration"
            ),
        },
        "limitations": [
            "ROUTINE and PRIORITY are research workflow labels derived from dataset grades, not diagnoses or validated urgency categories.",
            "DeepDRiD is a diabetes-screening dataset from three studies; this does not establish transport to other populations, devices, or retinal conditions.",
            "Each eye grade is repeated across correlated dual-view images; all splits and calibration are therefore patient-grouped.",
            "The official validation cohort has 100 patients and has been used elsewhere in this project for technical-quality research, so it is untouched in this run but not project-level fresh.",
            "Frozen ImageNet features may learn acquisition or quality shortcuts rather than retinal findings.",
            "Poor-quality images remain in evaluation; quality-stratified results are descriptive and this experiment does not alter or consume the deployed quality gate.",
            "Exact calibration bounds are nominal for the historical calibration cohort, not clinical deployment guarantees.",
            "No external disease-grading set, test split, MSHF, UWF, cloud service, or paid compute was used.",
        ],
        "calibration_results": rows["calibration"],
        "official_validation_results": eval_rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "schema_version": 1,
            "experimental_only": True,
            "runtime_integration_authorized": False,
            "diagnostic_use_authorized": False,
            "architecture": report["model"]["architecture"],
            "input_dim": int(development_features.shape[1]),
            "hidden_dim": args.hidden_dim,
            "members": states,
            "feature_mean": torch.from_numpy(feature_mean.astype("float32")),
            "feature_std": torch.from_numpy(feature_std.astype("float32")),
            "policy": policy,
            "label_semantics": report["semantics"],
        },
        output_dir / "escalation-baseline-experimental.pt",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "policy": policy,
                "official_validation_evaluation": report[
                    "official_validation_evaluation"
                ],
                "recommendation": report["recommendation"],
                "report": str(report_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
