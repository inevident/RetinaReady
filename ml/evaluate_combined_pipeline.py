#!/usr/bin/env python3
"""Evaluate the frozen RetinaReady + RetinaPriority policy entirely offline.

This evaluator does not run model inference. It hash-verifies and reuses the
existing 400-image score reports, independently reapplies both frozen strict
threshold policies, joins them to the official DeepDRiD validation truths, and
then simulates the quality-first workflow. Review-priority labels are research
workflow labels, not diagnoses, disease exclusions, or treatment advice.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from calibrate_selective_policy import exact_upper_bound


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = PROJECT_ROOT / "ml/configs/combined_offline_evaluation_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/combined-offline-evaluation/report.json"

QUALITY_TRUTHS = ("READY", "RETAKE")
QUALITY_DECISIONS = ("READY", "RETAKE", "LIMITED")
REVIEW_TRUTHS = ("ROUTINE", "PRIORITY")
ESCALATION_DECISIONS = ("ROUTINE", "PRIORITY", "UNCERTAIN")
FINAL_STATES = (
    "RETAKE",
    "LIMITED",
    "ROUTINE_REVIEW",
    "PRIORITY_REVIEW",
    "UNCERTAIN",
)
EXPECTED_INPUT_KEYS = {
    "quality_bundle_manifest",
    "quality_backbone",
    "quality_decision_head",
    "quality_factor_head",
    "quality_evaluation_manifest",
    "quality_image_level_report",
    "quality_report_hash_attestation",
    "escalation_evaluation_manifest",
    "escalation_image_level_report",
    "escalation_model_artifact",
    "escalation_promotion_manifest",
}


class EvaluationInputError(ValueError):
    """Raised when a pinned input or evaluation contract is invalid."""


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationInputError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationInputError(f"{path}: expected a JSON object")
    return value


def verify_locked_inputs(lock_path: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    lock = load_json(lock_path)
    if lock.get("schema_version") != 1:
        raise EvaluationInputError(f"{lock_path}: unsupported lock schema")
    inputs = lock.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != EXPECTED_INPUT_KEYS:
        raise EvaluationInputError(
            f"{lock_path}: input keys must be exactly {sorted(EXPECTED_INPUT_KEYS)}"
        )

    paths: dict[str, Path] = {}
    verified: dict[str, Any] = {}
    for name in sorted(inputs):
        entry = inputs[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise EvaluationInputError(f"{lock_path}: malformed input entry {name}")
        relative_path = entry["path"]
        expected = entry["sha256"]
        if not isinstance(relative_path, str) or not relative_path:
            raise EvaluationInputError(f"{lock_path}: invalid path for {name}")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise EvaluationInputError(f"{lock_path}: invalid SHA-256 for {name}")
        path = resolve(relative_path)
        if not path.is_file():
            raise EvaluationInputError(f"locked input is missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise EvaluationInputError(
                f"SHA-256 mismatch for {relative_path}: expected {expected}, got {actual}"
            )
        paths[name] = path
        verified[name] = {
            "path": relative_path,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": True,
        }
    return lock, paths, verified


def read_csv_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required_columns - set(reader.fieldnames or ())
            if missing:
                raise EvaluationInputError(f"{path}: missing columns {sorted(missing)}")
            rows = list(reader)
    except OSError as error:
        raise EvaluationInputError(f"cannot read CSV {path}: {error}") from error
    if not rows:
        raise EvaluationInputError(f"{path}: contains no rows")
    return rows


def index_unique(rows: Iterable[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise EvaluationInputError(f"{source}: every result must be an object")
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise EvaluationInputError(f"{source}: invalid image_id")
        if image_id in indexed:
            raise EvaluationInputError(f"{source}: duplicate image_id {image_id}")
        indexed[image_id] = row
    return indexed


def finite_score(value: Any, *, source: str, image_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationInputError(f"{source}: invalid score for {image_id}")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise EvaluationInputError(f"{source}: score outside [0, 1] for {image_id}")
    return score


def quality_decision(score: float, policy: dict[str, Any]) -> str:
    high = policy.get("ready_threshold_strictly_greater_than")
    low = policy.get("retake_threshold_strictly_less_than")
    if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
        raise EvaluationInputError("quality policy has invalid thresholds")
    if not 0.0 <= float(low) < float(high) <= 1.0:
        raise EvaluationInputError("quality policy thresholds are not ordered")
    if score > float(high):
        return "READY"
    if score < float(low):
        return "RETAKE"
    return "LIMITED"


def escalation_decision(score: float, policy: dict[str, Any]) -> str:
    low = policy.get("routine_if_score_strictly_less_than")
    high = policy.get("priority_if_score_strictly_greater_than")
    if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
        raise EvaluationInputError("escalation policy has invalid thresholds")
    if not 0.0 <= float(low) < float(high) <= 1.0:
        raise EvaluationInputError("escalation policy thresholds are not ordered")
    if score < float(low):
        return "ROUTINE"
    if score > float(high):
        return "PRIORITY"
    return "UNCERTAIN"


def pipeline_state(quality: str, escalation: str) -> str:
    if quality == "RETAKE":
        return "RETAKE"
    if quality == "LIMITED":
        return "LIMITED"
    if quality != "READY":
        raise EvaluationInputError(f"unknown quality decision {quality!r}")
    mapping = {
        "ROUTINE": "ROUTINE_REVIEW",
        "PRIORITY": "PRIORITY_REVIEW",
        "UNCERTAIN": "UNCERTAIN",
    }
    try:
        return mapping[escalation]
    except KeyError as error:
        raise EvaluationInputError(
            f"unknown escalation decision {escalation!r}"
        ) from error


def _same_number(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=0.0
    )


def validate_policy_provenance(
    *,
    paths: dict[str, Path],
    quality_bundle: dict[str, Any],
    quality_report: dict[str, Any],
    escalation_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for source, payload in (
        ("quality bundle", quality_bundle),
        ("quality report", quality_report),
        ("escalation report", escalation_report),
    ):
        if payload.get("schema_version") != 1:
            raise EvaluationInputError(f"{source} has an unsupported schema")
    bundle_policy = quality_bundle.get("policy")
    report_policy = quality_report.get("selective_policy")
    escalation_policy = escalation_report.get("policy")
    if not all(isinstance(value, dict) for value in (bundle_policy, report_policy, escalation_policy)):
        raise EvaluationInputError("one or more frozen policies are missing")

    quality_keys = (
        "ready_threshold_strictly_greater_than",
        "retake_threshold_strictly_less_than",
        "per_gate_delta",
    )
    for key in quality_keys:
        if not _same_number(bundle_policy.get(key), report_policy.get(key)):
            raise EvaluationInputError(
                f"quality bundle/report policy mismatch for {key}"
            )
    if bundle_policy.get("calibration_unit") != report_policy.get("calibration_unit"):
        raise EvaluationInputError("quality bundle/report calibration-unit mismatch")

    data = quality_report.get("data")
    if not isinstance(data, dict):
        raise EvaluationInputError("quality report is missing data provenance")
    quality_manifest_relative = paths["quality_evaluation_manifest"].relative_to(PROJECT_ROOT).as_posix()
    if data.get("evaluation_manifest") != quality_manifest_relative:
        raise EvaluationInputError("quality report points to a different evaluation manifest")
    if data.get("evaluation_manifest_sha256") != sha256_file(paths["quality_evaluation_manifest"]):
        raise EvaluationInputError("quality report evaluation-manifest hash mismatch")
    if data.get("evaluation_images") != 400 or data.get("evaluation_patients") != 100:
        raise EvaluationInputError("quality report is not the 400-image/100-patient evaluation")

    source_report = (
        quality_bundle.get("decision_head", {}).get("source_report")
        if isinstance(quality_bundle.get("decision_head"), dict)
        else None
    )
    expected_source = paths["quality_image_level_report"].relative_to(PROJECT_ROOT).as_posix()
    if source_report != expected_source:
        raise EvaluationInputError("quality bundle does not identify the pinned score report")

    quality_attestation = load_json(paths["quality_report_hash_attestation"])
    if quality_attestation.get("schema_version") != 1:
        raise EvaluationInputError("quality hash attestation has an unsupported schema")
    attested_specialist = quality_attestation.get("specialist")
    if not isinstance(attested_specialist, dict):
        raise EvaluationInputError("quality hash attestation is missing specialist provenance")
    if attested_specialist.get("report") != expected_source:
        raise EvaluationInputError("quality hash attestation identifies a different report")
    if attested_specialist.get("report_sha256") != sha256_file(
        paths["quality_image_level_report"]
    ):
        raise EvaluationInputError("quality report does not match its prior hash attestation")
    attested_policy = attested_specialist.get("policy")
    if not isinstance(attested_policy, dict):
        raise EvaluationInputError("quality hash attestation is missing its policy")
    for key in quality_keys:
        if not _same_number(attested_policy.get(key), report_policy.get(key)):
            raise EvaluationInputError(
                f"quality hash attestation/report policy mismatch for {key}"
            )
    for bundle_key, input_key in (
        ("backbone", "quality_backbone"),
        ("decision_head", "quality_decision_head"),
        ("factor_head", "quality_factor_head"),
    ):
        entry = quality_bundle.get(bundle_key)
        if not isinstance(entry, dict) or entry.get("sha256") != sha256_file(paths[input_key]):
            raise EvaluationInputError(f"quality bundle hash mismatch for {bundle_key}")

    audit = escalation_report.get("data_and_license_audit")
    if not isinstance(audit, dict):
        raise EvaluationInputError("escalation report is missing data provenance")
    eval_partition = audit.get("partitions", {}).get("eval")
    if not isinstance(eval_partition, dict):
        raise EvaluationInputError("escalation report is missing eval provenance")
    escalation_manifest_relative = paths["escalation_evaluation_manifest"].relative_to(PROJECT_ROOT).as_posix()
    if eval_partition.get("manifest") != escalation_manifest_relative:
        raise EvaluationInputError("escalation report points to a different eval manifest")
    if eval_partition.get("manifest_sha256") != sha256_file(paths["escalation_evaluation_manifest"]):
        raise EvaluationInputError("escalation report eval-manifest hash mismatch")
    if eval_partition.get("images") != 400 or eval_partition.get("patients") != 100:
        raise EvaluationInputError("escalation report is not the 400-image/100-patient evaluation")
    model = escalation_report.get("model")
    if not isinstance(model, dict) or model.get("backbone_sha256") != sha256_file(paths["quality_backbone"]):
        raise EvaluationInputError("escalation report backbone does not match the pinned backbone")
    if escalation_report.get("status") != "experimental-only; not integrated":
        raise EvaluationInputError("unexpected escalation artifact status")
    recommendation = escalation_report.get("recommendation")
    if not isinstance(recommendation, dict) or recommendation.get("clinical_use") != "not-authorized":
        raise EvaluationInputError("escalation artifact lacks the non-clinical-use contract")

    promotion = load_json(paths["escalation_promotion_manifest"])
    if promotion.get("schema_version") != 1:
        raise EvaluationInputError("escalation promotion manifest has an unsupported schema")
    if promotion.get("network_required") is not False or promotion.get("fail_closed_decision") != "UNCERTAIN":
        raise EvaluationInputError("escalation promotion manifest is not local and fail-closed")
    bindings = promotion.get("bindings")
    if not isinstance(bindings, dict):
        raise EvaluationInputError("escalation promotion manifest is missing bindings")
    for binding_key, input_key in (
        ("artifact", "escalation_model_artifact"),
        ("report", "escalation_image_level_report"),
        ("backbone", "quality_backbone"),
    ):
        binding = bindings.get(binding_key)
        if not isinstance(binding, dict):
            raise EvaluationInputError(f"escalation promotion binding is missing {binding_key}")
        expected_path = paths[input_key].relative_to(PROJECT_ROOT).as_posix()
        if binding.get("file") != expected_path or binding.get("sha256") != sha256_file(
            paths[input_key]
        ):
            raise EvaluationInputError(f"escalation promotion binding mismatch for {binding_key}")
    promoted_policy = promotion.get("policy_contract")
    if not isinstance(promoted_policy, dict):
        raise EvaluationInputError("escalation promotion manifest is missing its policy")
    for key in (
        "routine_if_score_strictly_less_than",
        "priority_if_score_strictly_greater_than",
    ):
        if not _same_number(promoted_policy.get(key), escalation_policy.get(key)):
            raise EvaluationInputError(
                f"escalation promotion/report policy mismatch for {key}"
            )
    if promoted_policy.get("equality_and_between_thresholds") != "UNCERTAIN":
        raise EvaluationInputError("escalation promotion equality policy is not fail-closed")
    if promoted_policy.get("internal_labels") != ["ROUTINE", "PRIORITY", "UNCERTAIN"]:
        raise EvaluationInputError("escalation promotion internal labels changed")
    if promoted_policy.get("output_mapping") != {
        "ROUTINE": "ROUTINE_REVIEW",
        "PRIORITY": "PRIORITY_REVIEW",
        "UNCERTAIN": "UNCERTAIN",
    }:
        raise EvaluationInputError("escalation promotion output mapping changed")

    return report_policy, escalation_policy


def join_records(
    *,
    quality_manifest_rows: list[dict[str, str]],
    escalation_manifest_rows: list[dict[str, str]],
    quality_results: list[dict[str, Any]],
    escalation_results: list[dict[str, Any]],
    quality_policy: dict[str, Any],
    escalation_policy: dict[str, Any],
    expected_images: int,
    expected_patients: int,
    expected_source_split: str,
) -> list[dict[str, Any]]:
    quality_manifest = index_unique(quality_manifest_rows, source="quality manifest")
    escalation_manifest = index_unique(escalation_manifest_rows, source="escalation manifest")
    quality_scores = index_unique(quality_results, source="quality score report")
    escalation_scores = index_unique(escalation_results, source="escalation score report")
    expected_ids = set(quality_manifest)
    sources = {
        "escalation manifest": set(escalation_manifest),
        "quality score report": set(quality_scores),
        "escalation score report": set(escalation_scores),
    }
    if len(expected_ids) != expected_images:
        raise EvaluationInputError(
            f"quality manifest has {len(expected_ids)} images; expected {expected_images}"
        )
    for source, image_ids in sources.items():
        if image_ids != expected_ids:
            missing = sorted(expected_ids - image_ids)
            extra = sorted(image_ids - expected_ids)
            raise EvaluationInputError(
                f"{source} image IDs do not match; missing={missing[:5]}, extra={extra[:5]}"
            )

    records: list[dict[str, Any]] = []
    for image_id, quality_truth_row in quality_manifest.items():
        escalation_truth_row = escalation_manifest[image_id]
        quality_score_row = quality_scores[image_id]
        escalation_score_row = escalation_scores[image_id]

        patient_id = quality_truth_row.get("patient_id")
        image_path = quality_truth_row.get("image_path")
        source_split = quality_truth_row.get("source_split")
        if not isinstance(patient_id, str) or not patient_id:
            raise EvaluationInputError(f"{image_id}: invalid patient ID")
        if not isinstance(image_path, str) or not image_path:
            raise EvaluationInputError(f"{image_id}: invalid image path")
        if quality_truth_row.get("split") != "val":
            raise EvaluationInputError(f"{image_id}: quality manifest is not official val")
        if escalation_truth_row.get("split") != "eval":
            raise EvaluationInputError(f"{image_id}: escalation manifest is not eval")
        if escalation_score_row.get("split") != "eval":
            raise EvaluationInputError(f"{image_id}: escalation score row is not eval")
        for source, row in (
            ("escalation manifest", escalation_truth_row),
            ("quality score report", quality_score_row),
            ("escalation score report", escalation_score_row),
        ):
            if row.get("patient_id") != patient_id:
                raise EvaluationInputError(f"{image_id}: patient mismatch in {source}")
            if row.get("image_path") != image_path:
                raise EvaluationInputError(f"{image_id}: image-path mismatch in {source}")
        if source_split != expected_source_split or escalation_truth_row.get("source_split") != source_split:
            raise EvaluationInputError(f"{image_id}: unexpected or mismatched source split")
        if escalation_score_row.get("source_split") != source_split:
            raise EvaluationInputError(f"{image_id}: escalation score source split mismatch")

        quality_truth = quality_truth_row.get("quality_label")
        if quality_truth not in QUALITY_TRUTHS:
            raise EvaluationInputError(f"{image_id}: invalid technical-quality truth")
        expected_overall = "1" if quality_truth == "READY" else "0"
        if quality_truth_row.get("overall_quality") != expected_overall:
            raise EvaluationInputError(f"{image_id}: inconsistent technical-quality truth")
        if escalation_truth_row.get("overall_quality") != expected_overall:
            raise EvaluationInputError(f"{image_id}: quality truth differs across manifests")
        if quality_score_row.get("truth") != quality_truth:
            raise EvaluationInputError(f"{image_id}: quality report truth mismatch")
        if escalation_score_row.get("overall_quality") != int(expected_overall):
            raise EvaluationInputError(f"{image_id}: escalation report quality truth mismatch")

        grade_text = escalation_truth_row.get("dr_grade")
        if grade_text not in {"0", "1", "2", "3", "4"}:
            raise EvaluationInputError(f"{image_id}: invalid DR grade")
        dr_grade = int(grade_text)
        review_truth = escalation_truth_row.get("escalation_label")
        expected_review_truth = "PRIORITY" if dr_grade >= 2 else "ROUTINE"
        if review_truth != expected_review_truth:
            raise EvaluationInputError(f"{image_id}: grade/review truth mismatch")
        if escalation_score_row.get("dr_grade") != dr_grade:
            raise EvaluationInputError(f"{image_id}: escalation report grade mismatch")
        if escalation_score_row.get("truth_review_priority") != review_truth:
            raise EvaluationInputError(f"{image_id}: escalation report truth mismatch")
        if escalation_truth_row.get("grade_source_field") not in {
            "left_eye_DR_Level",
            "right_eye_DR_Level",
        }:
            raise EvaluationInputError(f"{image_id}: invalid DR grade source field")
        if escalation_truth_row.get("filename_side_matches_grade_field") != "true":
            raise EvaluationInputError(
                f"{image_id}: unexpected official-validation filename-side mismatch"
            )

        quality_score = finite_score(
            quality_score_row.get("decision_score"),
            source="quality score report",
            image_id=image_id,
        )
        review_score = finite_score(
            escalation_score_row.get("review_priority_score"),
            source="escalation score report",
            image_id=image_id,
        )
        quality_call = quality_decision(quality_score, quality_policy)
        review_call = escalation_decision(review_score, escalation_policy)
        if escalation_score_row.get("decision") != review_call:
            raise EvaluationInputError(
                f"{image_id}: stored escalation decision does not reproduce from score"
            )
        final_state = pipeline_state(quality_call, review_call)
        gate_passed = quality_call == "READY"
        records.append(
            {
                "patient_id": patient_id,
                "image_id": image_id,
                "image_path": image_path,
                "source_split": source_split,
                "technical_quality_truth": quality_truth,
                "overall_quality": int(expected_overall),
                "dr_grade": dr_grade,
                "truth_review_priority": review_truth,
                "quality_score": quality_score,
                "quality_decision": quality_call,
                "quality_gate_passed": gate_passed,
                "cached_standalone_review_priority_score": review_score,
                "cached_standalone_review_priority_decision": review_call,
                "review_priority_stage_executed_in_simulated_pipeline": gate_passed,
                "final_pipeline_state": final_state,
                "quality_block_reason": quality_call if not gate_passed else None,
            }
        )
    if len({row["patient_id"] for row in records}) != expected_patients:
        raise EvaluationInputError("joined cohort does not have the expected patient count")
    images_per_patient = Counter(row["patient_id"] for row in records)
    if set(images_per_patient.values()) != {4}:
        raise EvaluationInputError("official validation must contain four images per patient")
    return records


def count_all(values: Iterable[str], labels: Iterable[str]) -> dict[str, int]:
    labels = tuple(labels)
    counts = Counter(values)
    unknown = set(counts) - set(labels)
    if unknown:
        raise EvaluationInputError(f"unexpected values while counting: {sorted(unknown)}")
    return {label: counts[label] for label in labels}


def matrix(
    records: list[dict[str, Any]], *, truth_key: str, truth_labels: Iterable[Any]
) -> dict[str, dict[str, int]]:
    return {
        str(truth): count_all(
            (row["final_pipeline_state"] for row in records if row[truth_key] == truth),
            FINAL_STATES,
        )
        for truth in truth_labels
    }


def rate_payload(
    count: int,
    denominator: int,
    *,
    exact_delta: float = 0.05,
    independent_unit: str = "image",
) -> dict[str, Any]:
    if denominator < 0 or not 0 <= count <= denominator:
        raise EvaluationInputError("invalid metric count or denominator")
    if denominator == 0:
        return {
            "count": 0,
            "denominator": 0,
            "rate": None,
            "one_sided_exact_upper_95": None,
            "bound_interpretation": "not computed because the stratum is empty",
        }
    if independent_unit not in {"image", "patient"}:
        raise EvaluationInputError("confidence-bound unit must be image or patient")
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator,
        "one_sided_exact_upper_95": exact_upper_bound(count, denominator, exact_delta),
        "bound_interpretation": (
            "nominal/descriptive only; correlated dual-view images violate an independent-image interpretation"
            if independent_unit == "image"
            else "patient adverse-event unit; still exploratory and not a deployment guarantee"
        ),
    }


def patient_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["patient_id"]].append(record)
    return dict(groups)


def patient_aggregate_state(rows: list[dict[str, Any]]) -> str:
    states = {row["final_pipeline_state"] for row in rows}
    # Reporting-only, conservative extension of the existing escalation
    # aggregation: an available PRIORITY flag wins; otherwise uncertainty or a
    # quality block prevents ROUTINE release.
    for state in ("PRIORITY_REVIEW", "UNCERTAIN", "RETAKE", "LIMITED", "ROUTINE_REVIEW"):
        if state in states:
            return state
    raise EvaluationInputError("patient has no pipeline state")


def patient_review_queue_state(rows: list[dict[str, Any]]) -> str:
    """Aggregate review release conservatively without hiding blocked images."""

    states = {row["final_pipeline_state"] for row in rows}
    if "PRIORITY_REVIEW" in states:
        return "PRIORITY_REVIEW"
    if states == {"ROUTINE_REVIEW"}:
        return "ROUTINE_REVIEW"
    return "UNCERTAIN"


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total <= 0:
        raise EvaluationInputError("cannot evaluate an empty cohort")
    quality_truth_counts = count_all(
        (row["technical_quality_truth"] for row in records), QUALITY_TRUTHS
    )
    review_truth_counts = count_all(
        (row["truth_review_priority"] for row in records), REVIEW_TRUTHS
    )
    grade_counts = count_all((str(row["dr_grade"]) for row in records), map(str, range(5)))
    quality_decision_counts = count_all(
        (row["quality_decision"] for row in records), QUALITY_DECISIONS
    )
    standalone_review_counts = count_all(
        (row["cached_standalone_review_priority_decision"] for row in records),
        ESCALATION_DECISIONS,
    )
    final_counts = count_all(
        (row["final_pipeline_state"] for row in records), FINAL_STATES
    )
    if sum(final_counts.values()) != total:
        raise EvaluationInputError("final pipeline states do not retain every image")

    quality_ready_rows = [row for row in records if row["quality_gate_passed"]]
    eligible_review_truth_counts = count_all(
        (row["truth_review_priority"] for row in quality_ready_rows), REVIEW_TRUTHS
    )
    eligible_review_decisions = count_all(
        (
            row["cached_standalone_review_priority_decision"]
            for row in quality_ready_rows
        ),
        ESCALATION_DECISIONS,
    )
    eligible_false_routine = sum(
        row["truth_review_priority"] == "PRIORITY"
        and row["cached_standalone_review_priority_decision"] == "ROUTINE"
        for row in quality_ready_rows
    )
    eligible_false_priority = sum(
        row["truth_review_priority"] == "ROUTINE"
        and row["cached_standalone_review_priority_decision"] == "PRIORITY"
        for row in quality_ready_rows
    )

    false_ready = sum(
        row["technical_quality_truth"] == "RETAKE" and row["quality_decision"] == "READY"
        for row in records
    )
    false_retake = sum(
        row["technical_quality_truth"] == "READY" and row["quality_decision"] == "RETAKE"
        for row in records
    )
    decisive_quality = quality_decision_counts["READY"] + quality_decision_counts["RETAKE"]
    correct_quality = sum(
        row["quality_decision"] in QUALITY_TRUTHS
        and row["quality_decision"] == row["technical_quality_truth"]
        for row in records
    )

    false_routine = sum(
        row["truth_review_priority"] == "PRIORITY"
        and row["final_pipeline_state"] == "ROUTINE_REVIEW"
        for row in records
    )
    false_priority = sum(
        row["truth_review_priority"] == "ROUTINE"
        and row["final_pipeline_state"] == "PRIORITY_REVIEW"
        for row in records
    )
    decisive_review = final_counts["ROUTINE_REVIEW"] + final_counts["PRIORITY_REVIEW"]
    correct_review = sum(
        (row["truth_review_priority"] == "ROUTINE" and row["final_pipeline_state"] == "ROUTINE_REVIEW")
        or (row["truth_review_priority"] == "PRIORITY" and row["final_pipeline_state"] == "PRIORITY_REVIEW")
        for row in records
    )
    blocked = final_counts["RETAKE"] + final_counts["LIMITED"]
    no_review_release = blocked + final_counts["UNCERTAIN"]
    abstentions = final_counts["LIMITED"] + final_counts["UNCERTAIN"]

    priority_rows = [row for row in records if row["truth_review_priority"] == "PRIORITY"]
    priority_blocked = [row for row in priority_rows if not row["quality_gate_passed"]]
    standalone_priority = [
        row
        for row in records
        if row["cached_standalone_review_priority_decision"] == "PRIORITY"
    ]
    standalone_priority_blocked = [row for row in standalone_priority if not row["quality_gate_passed"]]

    patients = patient_groups(records)
    priority_patients = {
        patient_id: rows
        for patient_id, rows in patients.items()
        if any(row["truth_review_priority"] == "PRIORITY" for row in rows)
    }
    routine_image_patients = {
        patient_id: rows
        for patient_id, rows in patients.items()
        if any(row["truth_review_priority"] == "ROUTINE" for row in rows)
    }
    retake_bearing_patients = {
        patient_id: rows
        for patient_id, rows in patients.items()
        if any(row["technical_quality_truth"] == "RETAKE" for row in rows)
    }
    ready_bearing_patients = {
        patient_id: rows
        for patient_id, rows in patients.items()
        if any(row["technical_quality_truth"] == "READY" for row in rows)
    }
    false_ready_patient_events = sum(
        any(
            row["technical_quality_truth"] == "RETAKE" and row["quality_decision"] == "READY"
            for row in rows
        )
        for rows in retake_bearing_patients.values()
    )
    false_retake_patient_events = sum(
        any(
            row["technical_quality_truth"] == "READY" and row["quality_decision"] == "RETAKE"
            for row in rows
        )
        for rows in ready_bearing_patients.values()
    )
    false_routine_patient_events = sum(
        any(
            row["truth_review_priority"] == "PRIORITY"
            and row["final_pipeline_state"] == "ROUTINE_REVIEW"
            for row in rows
        )
        for rows in priority_patients.values()
    )
    false_priority_patient_events = sum(
        any(
            row["truth_review_priority"] == "ROUTINE"
            and row["final_pipeline_state"] == "PRIORITY_REVIEW"
            for row in rows
        )
        for rows in routine_image_patients.values()
    )
    patient_states = {
        patient_id: patient_aggregate_state(rows) for patient_id, rows in patients.items()
    }
    patient_state_counts = count_all(patient_states.values(), FINAL_STATES)
    patient_review_truths = {
        patient_id: (
            "PRIORITY"
            if any(row["truth_review_priority"] == "PRIORITY" for row in rows)
            else "ROUTINE"
        )
        for patient_id, rows in patients.items()
    }
    patient_decisive = patient_state_counts["ROUTINE_REVIEW"] + patient_state_counts["PRIORITY_REVIEW"]
    patient_correct = sum(
        (patient_review_truths[patient_id] == "PRIORITY" and state == "PRIORITY_REVIEW")
        or (patient_review_truths[patient_id] == "ROUTINE" and state == "ROUTINE_REVIEW")
        for patient_id, state in patient_states.items()
    )
    patient_queue_states = {
        patient_id: patient_review_queue_state(rows)
        for patient_id, rows in patients.items()
    }
    patient_queue_counts = count_all(
        patient_queue_states.values(),
        ("ROUTINE_REVIEW", "PRIORITY_REVIEW", "UNCERTAIN"),
    )
    patient_queue_decisive = (
        patient_queue_counts["ROUTINE_REVIEW"]
        + patient_queue_counts["PRIORITY_REVIEW"]
    )
    patient_queue_correct = sum(
        (
            patient_review_truths[patient_id] == "PRIORITY"
            and state == "PRIORITY_REVIEW"
        )
        or (
            patient_review_truths[patient_id] == "ROUTINE"
            and state == "ROUTINE_REVIEW"
        )
        for patient_id, state in patient_queue_states.items()
    )
    patient_queue_false_routine = sum(
        patient_review_truths[patient_id] == "PRIORITY"
        and state == "ROUTINE_REVIEW"
        for patient_id, state in patient_queue_states.items()
    )
    patient_queue_false_priority = sum(
        patient_review_truths[patient_id] == "ROUTINE"
        and state == "PRIORITY_REVIEW"
        for patient_id, state in patient_queue_states.items()
    )
    priority_patients_all_blocked = sum(
        all(not row["quality_gate_passed"] for row in rows if row["truth_review_priority"] == "PRIORITY")
        for rows in priority_patients.values()
    )
    priority_patients_any_passed = len(priority_patients) - priority_patients_all_blocked
    priority_patients_final_priority = sum(
        any(row["final_pipeline_state"] == "PRIORITY_REVIEW" for row in rows)
        for rows in priority_patients.values()
    )

    return {
        "image_level": {
            "all_images_denominator": total,
            "technical_quality_truth_counts": quality_truth_counts,
            "review_priority_truth_counts": review_truth_counts,
            "dr_grade_truth_counts": grade_counts,
            "quality_gate": {
                "decision_counts": quality_decision_counts,
                "state_by_technical_quality_truth": {
                    truth: count_all(
                        (row["quality_decision"] for row in records if row["technical_quality_truth"] == truth),
                        QUALITY_DECISIONS,
                    )
                    for truth in QUALITY_TRUTHS
                },
                "selective_coverage_all_images": decisive_quality / total,
                "limited_abstention_rate_all_images": quality_decision_counts["LIMITED"] / total,
                "accepted_accuracy": correct_quality / decisive_quality if decisive_quality else None,
                "false_ready": rate_payload(false_ready, quality_truth_counts["RETAKE"]),
                "false_retake": rate_payload(false_retake, quality_truth_counts["READY"]),
            },
            "standalone_escalation_before_quality_gate": {
                "decision_counts": standalone_review_counts,
                "selective_coverage_all_images": (
                    standalone_review_counts["ROUTINE"] + standalone_review_counts["PRIORITY"]
                )
                / total,
                "uncertain_rate_all_images": standalone_review_counts["UNCERTAIN"] / total,
            },
            "review_stage_given_quality_ready": {
                "eligible_images": len(quality_ready_rows),
                "truth_counts": eligible_review_truth_counts,
                "decision_counts": eligible_review_decisions,
                "selective_coverage": (
                    eligible_review_decisions["ROUTINE"]
                    + eligible_review_decisions["PRIORITY"]
                )
                / len(quality_ready_rows)
                if quality_ready_rows
                else None,
                "uncertain_rate": (
                    eligible_review_decisions["UNCERTAIN"] / len(quality_ready_rows)
                    if quality_ready_rows
                    else None
                ),
                "false_routine_release": rate_payload(
                    eligible_false_routine, eligible_review_truth_counts["PRIORITY"]
                ),
                "false_priority_release": rate_payload(
                    eligible_false_priority, eligible_review_truth_counts["ROUTINE"]
                ),
                "selection_note": (
                    "conditional on the frozen quality gate; not a replacement for full-cohort metrics"
                ),
            },
            "full_pipeline": {
                "final_state_counts": final_counts,
                "state_by_technical_quality_truth": matrix(
                    records, truth_key="technical_quality_truth", truth_labels=QUALITY_TRUTHS
                ),
                "state_by_review_priority_truth": matrix(
                    records, truth_key="truth_review_priority", truth_labels=REVIEW_TRUTHS
                ),
                "state_by_dr_grade": matrix(
                    records, truth_key="dr_grade", truth_labels=range(5)
                ),
                "quality_blocked_count": blocked,
                "quality_blocked_rate_all_images": blocked / total,
                "downstream_uncertain_count": final_counts["UNCERTAIN"],
                "downstream_uncertain_rate_all_images": final_counts["UNCERTAIN"] / total,
                "abstention_count": abstentions,
                "abstention_rate_all_images": abstentions / total,
                "no_decisive_review_release_count": no_review_release,
                "no_decisive_review_release_rate_all_images": no_review_release / total,
                "decisive_review_count": decisive_review,
                "decisive_review_coverage_all_images": decisive_review / total,
                "decisive_review_coverage_given_quality_ready": (
                    decisive_review / quality_decision_counts["READY"]
                    if quality_decision_counts["READY"]
                    else None
                ),
                "accepted_review_accuracy": correct_review / decisive_review if decisive_review else None,
                "false_routine_danger": rate_payload(
                    false_routine, review_truth_counts["PRIORITY"]
                ),
                "false_priority_workload": rate_payload(
                    false_priority, review_truth_counts["ROUTINE"]
                ),
            },
        },
        "patient_level": {
            "all_patients_denominator": len(patients),
            "review_priority_truth_counts": {
                "ROUTINE": len(patients) - len(priority_patients),
                "PRIORITY": len(priority_patients),
            },
            "technical_quality_adverse_events": {
                "false_ready": rate_payload(
                    false_ready_patient_events,
                    len(retake_bearing_patients),
                    independent_unit="patient",
                ),
                "false_retake": rate_payload(
                    false_retake_patient_events,
                    len(ready_bearing_patients),
                    independent_unit="patient",
                ),
            },
            "combined_adverse_events_over_image_decisions": {
                "false_routine_danger": rate_payload(
                    false_routine_patient_events,
                    len(priority_patients),
                    independent_unit="patient",
                ),
                "false_priority_workload": rate_payload(
                    false_priority_patient_events,
                    len(routine_image_patients),
                    independent_unit="patient",
                ),
            },
            "conservative_aggregate": {
                "rule": (
                    "PRIORITY_REVIEW if any image is PRIORITY_REVIEW; else UNCERTAIN if any "
                    "image is UNCERTAIN; else RETAKE if any image is RETAKE; else LIMITED if "
                    "any image is LIMITED; else ROUTINE_REVIEW"
                ),
                "state_counts": patient_state_counts,
                "decisive_review_coverage": patient_decisive / len(patients),
                "accepted_review_accuracy": patient_correct / patient_decisive if patient_decisive else None,
            },
            "fail_closed_review_queue_aggregate": {
                "rule": (
                    "PRIORITY_REVIEW if any image is PRIORITY_REVIEW; ROUTINE_REVIEW only "
                    "if every image is ROUTINE_REVIEW; otherwise UNCERTAIN, including any "
                    "quality-blocked image"
                ),
                "decision_counts": patient_queue_counts,
                "decisive_review_coverage": patient_queue_decisive / len(patients),
                "accepted_review_accuracy": (
                    patient_queue_correct / patient_queue_decisive
                    if patient_queue_decisive
                    else None
                ),
                "false_routine_danger": rate_payload(
                    patient_queue_false_routine,
                    len(priority_patients),
                    independent_unit="patient",
                ),
                "false_priority_workload": rate_payload(
                    patient_queue_false_priority,
                    len(patients) - len(priority_patients),
                    independent_unit="patient",
                ),
            },
        },
        "quality_gate_effect_on_dataset_defined_priority": {
            "interpretation": (
                "yes: the quality-first contract prevents review-priority scoring from being "
                "used for some grade-derived PRIORITY-truth images; this is workflow blockage, "
                "not evidence about disease or clinical urgency"
                if priority_blocked
                else "no grade-derived PRIORITY-truth images were quality-blocked"
            ),
            "priority_truth_images": len(priority_rows),
            "blocked_priority_truth_images": len(priority_blocked),
            "blocked_priority_truth_image_rate": len(priority_blocked) / len(priority_rows),
            "blocked_as_retake": sum(row["quality_decision"] == "RETAKE" for row in priority_blocked),
            "blocked_as_limited": sum(row["quality_decision"] == "LIMITED" for row in priority_blocked),
            "priority_truth_images_passed_to_escalation": len(priority_rows) - len(priority_blocked),
            "standalone_priority_review_decisions": len(standalone_priority),
            "standalone_priority_review_decisions_blocked_by_quality": len(standalone_priority_blocked),
            "standalone_priority_review_decisions_blocked_rate": (
                len(standalone_priority_blocked) / len(standalone_priority)
                if standalone_priority
                else None
            ),
            "priority_truth_patients": len(priority_patients),
            "priority_truth_patients_with_all_priority_images_quality_blocked": priority_patients_all_blocked,
            "priority_truth_patients_with_any_priority_image_passing_quality": priority_patients_any_passed,
            "priority_truth_patients_with_any_final_priority_review": priority_patients_final_priority,
        },
    }


def validate_reproduced_baselines(
    records: list[dict[str, Any]],
    quality_report: dict[str, Any],
    escalation_report: dict[str, Any],
) -> None:
    quality_expected = quality_report.get("exploratory_validation_selective_metrics")
    escalation_expected = escalation_report.get("official_validation_evaluation", {}).get(
        "selective_image_metrics"
    )
    if not isinstance(quality_expected, dict) or not isinstance(escalation_expected, dict):
        raise EvaluationInputError("baseline reports are missing selective metrics")
    quality_counts = count_all((row["quality_decision"] for row in records), QUALITY_DECISIONS)
    if quality_counts != quality_expected.get("decision_counts"):
        raise EvaluationInputError("reapplied quality policy does not reproduce its source report")
    escalation_counts = count_all(
        (row["cached_standalone_review_priority_decision"] for row in records),
        ESCALATION_DECISIONS,
    )
    if escalation_counts != escalation_expected.get("decision_counts"):
        raise EvaluationInputError("reapplied escalation policy does not reproduce its source report")


def build_report(lock_path: Path = DEFAULT_LOCK, *, limit: int | None = None) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock, paths, verified = verify_locked_inputs(lock_path)
    dataset = lock.get("dataset")
    if not isinstance(dataset, dict):
        raise EvaluationInputError("lock is missing dataset contract")
    if dataset.get("name") != "DeepDRiD" or dataset.get("source_split") != "regular-fundus-validation":
        raise EvaluationInputError("only the pinned DeepDRiD official validation cohort is allowed")
    expected_images = dataset.get("images")
    expected_patients = dataset.get("patients")
    if expected_images != 400 or expected_patients != 100:
        raise EvaluationInputError("full contract must contain 400 images and 100 patients")

    quality_bundle = load_json(paths["quality_bundle_manifest"])
    quality_report = load_json(paths["quality_image_level_report"])
    escalation_report = load_json(paths["escalation_image_level_report"])
    quality_policy, review_policy = validate_policy_provenance(
        paths=paths,
        quality_bundle=quality_bundle,
        quality_report=quality_report,
        escalation_report=escalation_report,
    )
    quality_manifest_rows = read_csv_rows(
        paths["quality_evaluation_manifest"],
        {
            "split",
            "patient_id",
            "image_id",
            "image_path",
            "overall_quality",
            "quality_label",
            "source_split",
        },
    )
    escalation_manifest_rows = read_csv_rows(
        paths["escalation_evaluation_manifest"],
        {
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
        },
    )
    quality_results = quality_report.get("results")
    escalation_results = escalation_report.get("official_validation_results")
    if not isinstance(quality_results, list) or not isinstance(escalation_results, list):
        raise EvaluationInputError("pinned reports do not contain image-level score rows")
    records = join_records(
        quality_manifest_rows=quality_manifest_rows,
        escalation_manifest_rows=escalation_manifest_rows,
        quality_results=quality_results,
        escalation_results=escalation_results,
        quality_policy=quality_policy,
        escalation_policy=review_policy,
        expected_images=expected_images,
        expected_patients=expected_patients,
        expected_source_split=dataset["source_split"],
    )
    validate_reproduced_baselines(records, quality_report, escalation_report)
    full_records = records
    if limit is not None:
        if isinstance(limit, bool) or not 1 <= limit <= len(records):
            raise EvaluationInputError(f"limit must be between 1 and {len(records)}")
        if limit % 4:
            raise EvaluationInputError(
                "smoke limit must be a multiple of four to retain complete patient bundles"
            )
        records = records[:limit]
        smoke_patient_counts = Counter(row["patient_id"] for row in records)
        if set(smoke_patient_counts.values()) != {4}:
            raise EvaluationInputError(
                "smoke subset does not contain complete four-image patient bundles"
            )

    report = {
        "schema_version": 1,
        "status": "exploratory_offline_research_evaluation_only",
        "semantics": {
            "technical_quality": "capture-quality research only",
            "ROUTINE_REVIEW": (
                "lower clinician-review queue priority; not a finding that disease is absent"
            ),
            "PRIORITY_REVIEW": (
                "higher clinician-review queue priority; not a diagnosis or treatment recommendation"
            ),
            "UNCERTAIN": "abstain and route for human prioritization",
            "truth_mapping": "DeepDRiD eye grade 0-1 -> ROUTINE; grade 2-4 -> PRIORITY",
        },
        "provenance": {
            "evaluation_name": lock.get("evaluation_name"),
            "evaluation_lock": lock_path.relative_to(PROJECT_ROOT).as_posix(),
            "evaluation_lock_sha256": sha256_file(lock_path),
            "evaluator_script": "ml/evaluate_combined_pipeline.py",
            "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
            "all_locked_inputs_verified": True,
            "verified_inputs": verified,
            "image_level_quality_report_rows": len(quality_results),
            "image_level_escalation_report_rows": len(escalation_results),
            "joined_full_cohort_rows": len(full_records),
            "reused_existing_image_level_scores": True,
            "new_model_inference_executed": False,
            "cloud_resources_used": False,
            "model_family": "frozen ImageNet DenseNet-121 baselines already present in the project",
            "composition_scope": (
                "specialist-only offline policy composition; not the upload API or hybrid Gemma-veto runtime"
            ),
            "full_command": "python3 ml/evaluate_combined_pipeline.py",
            "smoke_command": (
                "python3 ml/evaluate_combined_pipeline.py --limit 8 "
                "--output outputs/combined-offline-evaluation/smoke.json"
            ),
        },
        "evaluation_set": {
            **dataset,
            "evaluated_images": len(records),
            "evaluated_patients": len({row["patient_id"] for row in records}),
            "is_smoke_subset": limit is not None,
            "all_images_retained_in_full_run": limit is None and len(records) == expected_images,
        },
        "pipeline_contract": {
            "quality_policy": quality_policy,
            "quality_gate": "only exact READY proceeds; RETAKE and LIMITED are final quality-block states",
            "review_priority_policy": review_policy,
            "final_states": list(FINAL_STATES),
            "cached_standalone_scores_note": (
                "Scores exist for all 400 images from the standalone baseline report, but the "
                "simulated combined pipeline marks review-priority execution false for every "
                "quality-blocked image and never uses those cached scores for its final state."
            ),
        },
        "integrity_checks": {
            "locked_input_files_verified": len(verified),
            "full_cohort_expected_images": expected_images,
            "full_cohort_joined_images": len(full_records),
            "full_cohort_unique_image_ids": len(
                {row["image_id"] for row in full_records}
            ),
            "full_cohort_unique_patients": len(
                {row["patient_id"] for row in full_records}
            ),
            "duplicate_image_ids": 0,
            "missing_or_extra_image_ids_across_four_sources": 0,
            "patient_id_mismatches": 0,
            "image_path_mismatches": 0,
            "source_split_mismatches": 0,
            "technical_quality_truth_mismatches": 0,
            "dr_grade_or_review_truth_mismatches": 0,
            "quality_report_prior_hash_attestation_verified": True,
            "quality_bundle_policy_reproduced": True,
            "escalation_promotion_bindings_verified": True,
            "escalation_policy_reproduced": True,
            "final_state_denominator_retained": len(records),
        },
        "metrics": compute_metrics(records),
        "limitations": [
            "This is retrospective exploratory evidence on a repeatedly viewed DeepDRiD validation cohort, not a fresh test or clinical validation.",
            "Both component policies retain their original historical calibration limitations; composing them does not create a new risk guarantee.",
            "The grade-derived review-priority truth is a research workflow convention and not a clinically validated urgency taxonomy.",
            "The evaluator reuses hash-pinned image-level scores and therefore measures the frozen recorded baselines, not a fresh inference-runtime reliability run.",
            "Results do not establish transport across populations, devices, clinics, or retinal conditions.",
        ],
        "records": records,
    }
    if sum(report["metrics"]["image_level"]["full_pipeline"]["final_state_counts"].values()) != len(records):
        raise RuntimeError("internal denominator invariant failed")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        help="hash/provenance lock (default: pinned v1 lock)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="machine-readable JSON output path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="deterministic first-N smoke subset; all 400 inputs are still verified and joined",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(resolve(args.lock), limit=args.limit)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    full = report["metrics"]["image_level"]["full_pipeline"]
    gate = report["metrics"]["quality_gate_effect_on_dataset_defined_priority"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "images": report["evaluation_set"]["evaluated_images"],
                "patients": report["evaluation_set"]["evaluated_patients"],
                "final_state_counts": full["final_state_counts"],
                "decisive_review_coverage_all_images": full[
                    "decisive_review_coverage_all_images"
                ],
                "false_routine_danger": full["false_routine_danger"],
                "false_priority_workload": full["false_priority_workload"],
                "blocked_priority_truth_images": gate["blocked_priority_truth_images"],
                "output": output.relative_to(PROJECT_ROOT).as_posix()
                if output.is_relative_to(PROJECT_ROOT)
                else str(output),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
