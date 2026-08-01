#!/usr/bin/env python3
"""Calibrate and evaluate a Gemma retinal review-priority adapter safely.

This utility consumes two *complete* decision-logit reports produced by
``evaluate_decision_logits.py --task escalation``.  Thresholds are selected
only from the patient-grouped calibration report.  A separate, patient-
disjoint evaluation report is scored after the policy is frozen.

The output is non-diagnostic research evidence.  ``ROUTINE`` means only a
lower place in a clinician review queue; it does not mean a healthy retina.
Threshold equality always abstains as ``UNCERTAIN``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from experiment_escalation_baseline import (
    DECISIONS,
    TRUTH_LABELS,
    assign,
    calibrate_thresholds,
    patient_aggregate_metrics,
    patient_event_metrics,
    ranking_metrics,
    read_escalation_manifest,
    result_distribution_by_grade,
    selective_metrics,
)
from train_quality_specialist import resolve, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CalibrationInputError(ValueError):
    """Raised when an input cannot prove the calibration contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationInputError(f"{name} must be a JSON object")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationInputError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationInputError(f"{name} must be an integer")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationInputError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CalibrationInputError(f"{name} must be finite and within [0, 1]")
    return number


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationInputError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationInputError(f"{name} must be finite")
    return number


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CalibrationInputError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path, name: str) -> dict[str, Any]:
    resolved = resolve(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationInputError(f"cannot read {name} {resolved}: {exc}") from exc
    return _object(payload, name)


def _reported_path_matches(recorded: str, manifest: Path) -> bool:
    """Accept the same project-relative path across local and remote roots."""

    resolved = resolve(manifest).resolve()
    recorded_path = Path(recorded)
    if (
        recorded_path == resolved
        or str(recorded_path) == str(resolved)
        or (recorded_path.is_absolute() and recorded_path.resolve() == resolved)
    ):
        return True
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return False
    recorded_parts = recorded_path.parts
    relative_parts = relative.parts
    return (
        len(recorded_parts) >= len(relative_parts)
        and recorded_parts[-len(relative_parts) :] == relative_parts
    )


def _adapter_binding(run: dict[str, Any], report_name: str) -> dict[str, Any]:
    adapter = _object(run.get("adapter"), f"{report_name}.run.adapter")
    training = _object(
        run.get("training_contract"), f"{report_name}.run.training_contract"
    )
    if training.get("task") != "escalation":
        raise CalibrationInputError(
            f"{report_name} adapter training contract is not task='escalation'"
        )
    if training.get("loss_scope") != "decision_token":
        raise CalibrationInputError(
            f"{report_name} adapter was not trained with decision-token loss"
        )
    return {
        "model_id": _nonempty_string(run.get("model_id"), f"{report_name}.model_id"),
        "model_revision": _nonempty_string(
            run.get("model_revision"), f"{report_name}.model_revision"
        ),
        "processor_id": _nonempty_string(
            run.get("processor_id"), f"{report_name}.processor_id"
        ),
        "processor_revision": _nonempty_string(
            run.get("processor_revision"), f"{report_name}.processor_revision"
        ),
        "adapter_weights_sha256": _sha256(
            adapter.get("weights_sha256"),
            f"{report_name}.run.adapter.weights_sha256",
        ),
        "training_provenance_sha256": _sha256(
            training.get("provenance_sha256"),
            f"{report_name}.run.training_contract.provenance_sha256",
        ),
        "loss_scope": "decision_token",
        "task": "escalation",
    }


def _manifest_binding(
    *,
    report_name: str,
    run: dict[str, Any],
    manifest: Path,
    expected_split: str,
    row_count: int,
) -> dict[str, Any]:
    resolved = resolve(manifest).resolve()
    actual_sha = sha256_file(resolved)
    reported_sha = _sha256(
        run.get("manifest_sha256"), f"{report_name}.run.manifest_sha256"
    )
    if reported_sha != actual_sha:
        raise CalibrationInputError(
            f"{report_name} manifest SHA mismatch: report={reported_sha}, "
            f"actual={actual_sha}"
        )
    recorded_path = _nonempty_string(
        run.get("manifest"), f"{report_name}.run.manifest"
    )
    if not _reported_path_matches(recorded_path, resolved):
        raise CalibrationInputError(
            f"{report_name} manifest path does not identify {resolved}: "
            f"{recorded_path!r}"
        )
    reported_split = run.get("expected_split")
    if reported_split is not None and reported_split != expected_split:
        raise CalibrationInputError(
            f"{report_name} expected_split={reported_split!r}, not "
            f"{expected_split!r}"
        )
    available = _integer(
        run.get("available_rows"), f"{report_name}.run.available_rows"
    )
    selected = _integer(
        run.get("selected_rows"), f"{report_name}.run.selected_rows"
    )
    if available != row_count or selected != row_count or selected != available:
        raise CalibrationInputError(
            f"{report_name} must contain the full manifest: manifest={row_count}, "
            f"available={available}, selected={selected}"
        )
    return {
        "path": str(resolved),
        "reported_path": recorded_path,
        "sha256": actual_sha,
        "split": expected_split,
        "split_provenance": (
            "explicit_in_decision_report"
            if reported_split is not None
            else "legacy_report_inferred_and_verified_from_full_hash_bound_manifest"
        ),
        "images": row_count,
    }


def load_scored_partition(
    report_path: Path,
    manifest_path: Path,
    *,
    expected_split: str,
    report_name: str,
) -> dict[str, Any]:
    """Validate one full report and convert its probabilities to policy rows."""

    report_resolved = resolve(report_path).resolve()
    manifest_resolved = resolve(manifest_path).resolve()
    report = _load_json(report_resolved, report_name)
    run = _object(report.get("run"), f"{report_name}.run")
    if run.get("mode") != "decision-token-logits":
        raise CalibrationInputError(
            f"{report_name} mode must be 'decision-token-logits'"
        )
    if run.get("task") != "escalation":
        raise CalibrationInputError(f"{report_name} task must be 'escalation'")
    if run.get("roc_auc_positive_class") != "PRIORITY":
        raise CalibrationInputError(
            f"{report_name} positive class must be 'PRIORITY'"
        )

    examples = read_escalation_manifest(
        manifest_resolved, expected_split=expected_split
    )
    manifest_binding = _manifest_binding(
        report_name=report_name,
        run=run,
        manifest=manifest_resolved,
        expected_split=expected_split,
        row_count=len(examples),
    )
    adapter_binding = _adapter_binding(run, report_name)

    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise CalibrationInputError(f"{report_name}.results must be an array")
    if len(raw_results) != len(examples):
        raise CalibrationInputError(
            f"{report_name} has {len(raw_results)} results for "
            f"{len(examples)} manifest rows"
        )
    summary = _object(report.get("summary"), f"{report_name}.summary")
    if _integer(summary.get("samples"), f"{report_name}.summary.samples") != len(
        examples
    ):
        raise CalibrationInputError(f"{report_name} summary sample count mismatch")

    by_image = {example.image_id: example for example in examples}
    seen_images: set[str] = set()
    patient_by_image: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_results):
        result = _object(raw, f"{report_name}.results[{index}]")
        image_id = _nonempty_string(
            result.get("image_id"), f"{report_name}.results[{index}].image_id"
        )
        patient_id = _nonempty_string(
            result.get("patient_id"), f"{report_name}.results[{index}].patient_id"
        )
        if image_id in seen_images:
            raise CalibrationInputError(
                f"{report_name} contains duplicate image_id {image_id!r}"
            )
        seen_images.add(image_id)
        if image_id not in by_image:
            raise CalibrationInputError(
                f"{report_name} image {image_id!r} is absent from its manifest"
            )
        example = by_image[image_id]
        if patient_id != example.patient_id:
            raise CalibrationInputError(
                f"{report_name} {image_id}: patient_id does not match manifest"
            )
        patient_by_image[image_id] = patient_id
        if result.get("image_path") != example.image_path:
            raise CalibrationInputError(
                f"{report_name} {image_id}: image_path does not match manifest"
            )
        if result.get("truth") != example.escalation_label:
            raise CalibrationInputError(
                f"{report_name} {image_id}: truth does not match manifest"
            )
        if result.get("positive_label") != "PRIORITY":
            raise CalibrationInputError(
                f"{report_name} {image_id}: positive_label must be PRIORITY"
            )
        if result.get("negative_label") != "ROUTINE":
            raise CalibrationInputError(
                f"{report_name} {image_id}: negative_label must be ROUTINE"
            )
        score = _probability(
            result.get("positive_probability"),
            f"{report_name} {image_id}.positive_probability",
        )
        positive_logit = _finite_number(
            result.get("positive_logit"),
            f"{report_name} {image_id}.positive_logit",
        )
        negative_logit = _finite_number(
            result.get("negative_logit"),
            f"{report_name} {image_id}.negative_logit",
        )
        margin = _finite_number(
            result.get("positive_minus_negative_logit"),
            f"{report_name} {image_id}.positive_minus_negative_logit",
        )
        if not math.isclose(
            positive_logit - negative_logit, margin, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise CalibrationInputError(
                f"{report_name} {image_id}: logit margin is internally inconsistent"
            )
        if not math.isclose(
            _stable_sigmoid(margin), score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CalibrationInputError(
                f"{report_name} {image_id}: positive_probability does not match "
                "the two-class logit margin"
            )
        if "negative_probability" in result:
            negative = _probability(
                result["negative_probability"],
                f"{report_name} {image_id}.negative_probability",
            )
            if not math.isclose(score + negative, 1.0, abs_tol=1e-9):
                raise CalibrationInputError(
                    f"{report_name} {image_id}: class probabilities do not sum to one"
                )
        rows.append(
            {
                "split": expected_split,
                "patient_id": example.patient_id,
                "image_id": example.image_id,
                "image_path": example.image_path,
                "dr_grade": example.dr_grade,
                "truth_review_priority": example.escalation_label,
                "overall_quality": example.overall_quality,
                "source_split": example.source_split,
                # This is deliberately renamed: the two-class softmax is a
                # ranking score, not a calibrated probability of disease.
                "review_priority_score": score,
            }
        )

    if seen_images != set(by_image):
        missing = sorted(set(by_image) - seen_images)
        raise CalibrationInputError(
            f"{report_name} does not cover the full manifest; missing {missing[:5]}"
        )
    report_patients = set(patient_by_image.values())
    manifest_patients = {example.patient_id for example in examples}
    if report_patients != manifest_patients:
        raise CalibrationInputError(
            f"{report_name} patient set does not match its manifest"
        )
    return {
        "rows": rows,
        "patient_ids": report_patients,
        "image_ids": seen_images,
        "report_binding": {
            "path": str(report_resolved),
            "sha256": sha256_file(report_resolved),
        },
        "manifest_binding": manifest_binding,
        "adapter_binding": adapter_binding,
    }


def build_report(
    *,
    calibration_report: Path,
    evaluation_report: Path,
    calibration_manifest: Path,
    evaluation_manifest: Path,
    false_routine_risk: float = 0.05,
    false_priority_risk: float = 0.05,
    delta: float = 0.05,
) -> dict[str, Any]:
    """Freeze on calibration, then measure a disjoint evaluation partition."""

    for value, name in (
        (false_routine_risk, "false_routine_risk"),
        (false_priority_risk, "false_priority_risk"),
    ):
        if not 0.0 < value < 1.0:
            raise CalibrationInputError(f"{name} must be strictly between zero and one")
    if not 0.0 < delta < 0.5:
        raise CalibrationInputError("delta must be strictly between zero and 0.5")
    if resolve(calibration_report).resolve() == resolve(evaluation_report).resolve():
        raise CalibrationInputError(
            "calibration and evaluation reports must be different artifacts"
        )
    if resolve(calibration_manifest).resolve() == resolve(evaluation_manifest).resolve():
        raise CalibrationInputError(
            "calibration and evaluation manifests must be different artifacts"
        )

    calibration = load_scored_partition(
        calibration_report,
        calibration_manifest,
        expected_split="calibration",
        report_name="calibration_report",
    )
    calibration_rows = calibration["rows"]
    # Freeze the policy before the evaluation report is even loaded. This is
    # stronger than merely excluding evaluation rows from the fitting call and
    # keeps the implementation aligned with the sealed-evaluation provenance.
    policy = calibrate_thresholds(
        calibration_rows,
        false_routine_risk=false_routine_risk,
        false_priority_risk=false_priority_risk,
        delta=delta,
    )
    lower = float(policy["routine_if_score_strictly_less_than"])
    upper = float(policy["priority_if_score_strictly_greater_than"])
    if not 0.0 <= lower <= upper <= 1.0:
        raise CalibrationInputError("calibration produced invalid ordered thresholds")
    if assign(lower, policy) != "UNCERTAIN" or assign(upper, policy) != "UNCERTAIN":
        raise CalibrationInputError("threshold equality must map to UNCERTAIN")

    evaluation = load_scored_partition(
        evaluation_report,
        evaluation_manifest,
        expected_split="eval",
        report_name="evaluation_report",
    )
    if calibration["adapter_binding"] != evaluation["adapter_binding"]:
        raise CalibrationInputError(
            "calibration and evaluation reports do not bind the same adapter, "
            "model, processor, and training provenance"
        )
    patient_overlap = calibration["patient_ids"] & evaluation["patient_ids"]
    image_overlap = calibration["image_ids"] & evaluation["image_ids"]
    if patient_overlap:
        raise CalibrationInputError(
            "calibration/evaluation patient leakage: "
            f"{sorted(patient_overlap)[:5]}"
        )
    if image_overlap:
        raise CalibrationInputError(
            "calibration/evaluation image leakage: "
            f"{sorted(image_overlap)[:5]}"
        )
    evaluation_rows = evaluation["rows"]

    for rows in (calibration_rows, evaluation_rows):
        for row in rows:
            row["decision"] = assign(row["review_priority_score"], policy)
            if row["decision"] not in DECISIONS:
                raise CalibrationInputError("policy emitted an unknown decision")

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "ranking": ranking_metrics(rows),
            "selective_image_metrics": selective_metrics(rows, policy, delta=delta),
            "patient_adverse_event_metrics": patient_event_metrics(
                rows, policy, delta=delta
            ),
            "patient_aggregate_metrics": patient_aggregate_metrics(
                rows, policy, delta=delta
            ),
            "decision_distribution_by_dr_grade": result_distribution_by_grade(
                rows, policy
            ),
        }

    core: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "gemma_escalation_selective_policy_evaluation",
        "status": "research-demo-only; not clinically validated",
        "generated_at": utc_now(),
        "semantics": {
            "ROUTINE": (
                "lower clinician review priority; not a finding of a healthy "
                "retina and not disease exclusion"
            ),
            "PRIORITY": (
                "higher clinician review priority; not a diagnosis, urgency "
                "determination, or treatment recommendation"
            ),
            "UNCERTAIN": "abstain and route for human review",
            "score": (
                "two-class Gemma decision-token softmax used only as an "
                "uncalibrated ranking score"
            ),
            "truth_mapping": "DeepDRiD grades 0-1 -> ROUTINE; grades 2-4 -> PRIORITY",
        },
        "inputs": {
            "adapter": calibration["adapter_binding"],
            "calibration_decision_report": calibration["report_binding"],
            "evaluation_decision_report": evaluation["report_binding"],
            "calibration_manifest": calibration["manifest_binding"],
            "evaluation_manifest": evaluation["manifest_binding"],
        },
        "separation_and_freeze_audit": {
            "calibration_expected_split": "calibration",
            "evaluation_expected_split": "eval",
            "calibration_patients": len(calibration["patient_ids"]),
            "evaluation_patients": len(evaluation["patient_ids"]),
            "patient_overlap": 0,
            "image_overlap": 0,
            "threshold_selection_source": "calibration decision report only",
            "evaluation_used_for_threshold_selection": False,
            "evaluation_sequence": "policy frozen before evaluation metrics",
            "full_manifest_coverage_required": True,
        },
        "policy": policy,
        "calibration": {
            "metrics": metrics(calibration_rows),
            "rows": calibration_rows,
        },
        "evaluation": {
            "role": "frozen-policy evaluation only; never threshold selection",
            "metrics": metrics(evaluation_rows),
            "rows": evaluation_rows,
        },
        "recommendation": {
            "runtime_integration": "not-authorized-by-this-report",
            "clinical_use": "not-authorized",
            "next_evidence_needed": (
                "fresh external multi-device patient-disjoint validation, "
                "prospective workflow testing, clinician review of the priority "
                "mapping, and one-time calibration on an independent cohort"
            ),
        },
        "limitations": [
            "This is a retrospective research/demo evaluation, not clinical validation.",
            "ROUTINE and PRIORITY are workflow labels derived from diabetic-retinopathy grades, not diagnoses or validated urgency categories.",
            "The decision-token softmax is renamed review_priority_score because it is not a calibrated probability of disease or harm.",
            "The nominal patient-event bounds assume exchangeability and do not establish safety on another population, device, modality, or acquisition protocol.",
            "DeepDRiD images and patients are correlated; the utility therefore refuses incomplete reports and any calibration/evaluation patient overlap.",
            "Poor-quality images remain present; a separate technical-quality gate must fail closed before any review-priority output is released.",
        ],
    }
    core_hash = canonical_sha256(core)
    core["integrity"] = {
        "algorithm": "SHA-256",
        "canonicalization": (
            "UTF-8 JSON, keys sorted, separators=(',', ':'), allow_nan=false, "
            "with the top-level integrity object omitted"
        ),
        "canonical_report_without_integrity_sha256": core_hash,
        "input_binding_complete": True,
    }
    return core


def verify_integrity(report: dict[str, Any]) -> bool:
    """Verify the report's internal canonical content binding."""

    integrity = report.get("integrity")
    if not isinstance(integrity, dict):
        return False
    expected = integrity.get("canonical_report_without_integrity_sha256")
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        return False
    core = {key: value for key, value in report.items() if key != "integrity"}
    try:
        return canonical_sha256(core) == expected
    except (TypeError, ValueError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a strict patient-level Gemma escalation policy on a complete "
            "calibration report, then score a disjoint complete evaluation report."
        )
    )
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("data/escalation-manifests/calibration.csv"),
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=Path("data/escalation-manifests/eval.csv"),
    )
    parser.add_argument("--false-routine-risk", type=float, default=0.05)
    parser.add_argument("--false-priority-risk", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = build_report(
            calibration_report=args.calibration_report,
            evaluation_report=args.evaluation_report,
            calibration_manifest=args.calibration_manifest,
            evaluation_manifest=args.evaluation_manifest,
            false_routine_risk=args.false_routine_risk,
            false_priority_risk=args.false_priority_risk,
            delta=args.delta,
        )
        output = resolve(args.output).resolve()
        if output in {
            resolve(args.calibration_report).resolve(),
            resolve(args.evaluation_report).resolve(),
            resolve(args.calibration_manifest).resolve(),
            resolve(args.evaluation_manifest).resolve(),
        }:
            raise CalibrationInputError("--output must not overwrite an input artifact")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (CalibrationInputError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_file(output),
                "policy": report["policy"],
                "evaluation": report["evaluation"]["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
