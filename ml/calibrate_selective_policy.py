#!/usr/bin/env python3
"""Calibrate a three-way READY / RETAKE / LIMITED quality policy.

The policy is intentionally asymmetric:

* READY only above a class-conditional, high-confidence upper threshold.
* RETAKE only below a separately calibrated lower threshold.
* LIMITED everywhere in between.

Thresholds use one-sided exact binomial upper confidence bounds.  This avoids
calling a noisy point estimate such as 10 / 218 = 4.6% a guaranteed 5% risk.
The input must contain out-of-sample calibration scores; never calibrate on the
sealed evaluation set.

Related methods:
* Angelopoulos et al., Conformal Risk Control (ICLR 2024)
* Learn-Then-Test risk-controlling prediction (Angelopoulos et al.)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LABELS = ("READY", "RETAKE")


def binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    return sum(
        math.comb(n, index)
        * probability**index
        * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def exact_upper_bound(errors: int, samples: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper confidence bound."""

    if not 0 <= errors <= samples:
        raise ValueError("errors must be between zero and samples")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be between zero and one")
    if errors == samples:
        return 1.0
    low = errors / samples
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if binomial_cdf(errors, samples, midpoint) > delta:
            low = midpoint
        else:
            high = midpoint
    return high


def maximum_certified_errors(samples: int, risk: float, delta: float) -> int:
    if not 0.0 < risk < 1.0:
        raise ValueError("risk must be between zero and one")
    certified = -1
    for errors in range(samples + 1):
        if exact_upper_bound(errors, samples, delta) <= risk:
            certified = errors
        else:
            break
    return certified


def decision_score(result: dict[str, Any]) -> float:
    """Read the uncalibrated ranking score, accepting legacy result files."""

    score = result.get("decision_score", result.get("ready_probability"))
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError(f"invalid decision score for {result.get('image_id')}")
    return float(score)


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path}: missing non-empty results list")
    seen: set[str] = set()
    for result in results:
        image_id = result.get("image_id")
        truth = result.get("truth")
        score = result.get("decision_score", result.get("ready_probability"))
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"{path}: invalid image_id")
        if image_id in seen:
            raise ValueError(f"{path}: duplicate image_id {image_id}")
        seen.add(image_id)
        if truth not in LABELS:
            raise ValueError(f"{path}: invalid truth {truth!r}")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{path}: invalid score for {image_id}")
    return results


def calibration_scores(
    results: list[dict[str, Any]], *, unit: str
) -> tuple[list[float], list[float]]:
    """Return adverse-class scores at the requested independent risk unit.

    At patient level, a false-READY event means that *any* RETAKE image from a
    patient is released as READY, so the patient's maximum RETAKE-image score
    is the relevant statistic. The dual false-RETAKE event uses the minimum
    score among that patient's READY images.
    """

    if unit == "image":
        retake_scores = [
            decision_score(result)
            for result in results
            if result["truth"] == "RETAKE"
        ]
        ready_scores = [
            decision_score(result)
            for result in results
            if result["truth"] == "READY"
        ]
    elif unit == "patient":
        patients: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            patient_id = result.get("patient_id")
            if not isinstance(patient_id, str) or not patient_id:
                raise ValueError("patient-level calibration requires patient_id")
            patients.setdefault(patient_id, []).append(result)
        retake_scores = [
            max(
                decision_score(result)
                for result in patient_results
                if result["truth"] == "RETAKE"
            )
            for patient_results in patients.values()
            if any(result["truth"] == "RETAKE" for result in patient_results)
        ]
        ready_scores = [
            min(
                decision_score(result)
                for result in patient_results
                if result["truth"] == "READY"
            )
            for patient_results in patients.values()
            if any(result["truth"] == "READY" for result in patient_results)
        ]
    else:
        raise ValueError("calibration unit must be 'image' or 'patient'")
    return sorted(retake_scores, reverse=True), sorted(ready_scores)


def calibrate_thresholds(
    results: list[dict[str, Any]],
    *,
    false_ready_risk: float,
    false_retake_risk: float,
    delta: float,
    unit: str = "image",
) -> dict[str, Any]:
    retake_scores, ready_scores = calibration_scores(results, unit=unit)
    if not retake_scores or not ready_scores:
        raise ValueError("calibration requires both READY and RETAKE examples")

    ready_errors = maximum_certified_errors(
        len(retake_scores), false_ready_risk, delta
    )
    retake_errors = maximum_certified_errors(
        len(ready_scores), false_retake_risk, delta
    )
    ready_threshold = (
        retake_scores[ready_errors]
        if ready_errors >= 0 and ready_errors < len(retake_scores)
        else math.inf
    )
    retake_threshold = (
        ready_scores[retake_errors]
        if retake_errors >= 0 and retake_errors < len(ready_scores)
        else -math.inf
    )
    if retake_threshold >= ready_threshold:
        raise ValueError(
            "calibrated thresholds overlap; tighter risks or a better-ranking model "
            "are required"
        )
    return {
        "calibration_unit": unit,
        "per_gate_delta": delta,
        "simultaneous_confidence_lower_bound": max(0.0, 1.0 - 2.0 * delta),
        "ready_threshold_strictly_greater_than": ready_threshold,
        "retake_threshold_strictly_less_than": retake_threshold,
        "false_ready": {
            "risk_limit": false_ready_risk,
            "delta": delta,
            "calibration_class": "RETAKE",
            "calibration_samples": len(retake_scores),
            "event": (
                "any RETAKE image for a patient is released as READY"
                if unit == "patient"
                else "a RETAKE image is released as READY"
            ),
            "maximum_certified_errors": ready_errors,
            "upper_bound_at_maximum_errors": (
                exact_upper_bound(ready_errors, len(retake_scores), delta)
                if ready_errors >= 0
                else None
            ),
        },
        "false_retake": {
            "risk_limit": false_retake_risk,
            "delta": delta,
            "calibration_class": "READY",
            "calibration_samples": len(ready_scores),
            "event": (
                "any READY image for a patient is sent to RETAKE"
                if unit == "patient"
                else "a READY image is sent to RETAKE"
            ),
            "maximum_certified_errors": retake_errors,
            "upper_bound_at_maximum_errors": (
                exact_upper_bound(retake_errors, len(ready_scores), delta)
                if retake_errors >= 0
                else None
            ),
        },
    }


def assign(score: float, policy: dict[str, Any]) -> str:
    if score > policy["ready_threshold_strictly_greater_than"]:
        return "READY"
    if score < policy["retake_threshold_strictly_less_than"]:
        return "RETAKE"
    return "LIMITED"


def evaluate(results: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    decisions = [assign(decision_score(result), policy) for result in results]
    truth_counts = {
        label: sum(r["truth"] == label for r in results) for label in LABELS
    }
    counts = {
        decision: sum(value == decision for value in decisions)
        for decision in ("READY", "RETAKE", "LIMITED")
    }
    accepted = counts["READY"] + counts["RETAKE"]
    correct = sum(
        decision in LABELS and decision == result["truth"]
        for result, decision in zip(results, decisions, strict=True)
    )
    false_ready = sum(
        result["truth"] == "RETAKE" and decision == "READY"
        for result, decision in zip(results, decisions, strict=True)
    )
    false_retake = sum(
        result["truth"] == "READY" and decision == "RETAKE"
        for result, decision in zip(results, decisions, strict=True)
    )
    true_ready = sum(
        result["truth"] == "READY" and decision == "READY"
        for result, decision in zip(results, decisions, strict=True)
    )
    true_retake = sum(
        result["truth"] == "RETAKE" and decision == "RETAKE"
        for result, decision in zip(results, decisions, strict=True)
    )
    return {
        "samples": len(results),
        "truth_counts": truth_counts,
        "decision_counts": counts,
        "coverage": accepted / len(results),
        "accepted_accuracy": correct / accepted if accepted else None,
        "false_ready_rate_given_retake": false_ready / truth_counts["RETAKE"],
        "false_retake_rate_given_ready": false_retake / truth_counts["READY"],
        "ready_recall": true_ready / truth_counts["READY"],
        "retake_recall": true_retake / truth_counts["RETAKE"],
        "ready_precision": (
            true_ready / counts["READY"] if counts["READY"] else None
        ),
        "retake_precision": (
            true_retake / counts["RETAKE"] if counts["RETAKE"] else None
        ),
        "false_ready_count": false_ready,
        "false_retake_count": false_retake,
    }


def evaluate_patient_events(
    results: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate patient-level any-error events used by clustered calibration."""

    patients: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        patient_id = result.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient-level evaluation requires patient_id")
        patients.setdefault(patient_id, []).append(result)

    retake_bearing = 0
    ready_bearing = 0
    false_ready_patients = 0
    false_retake_patients = 0
    for patient_results in patients.values():
        rows = [
            (result, assign(decision_score(result), policy))
            for result in patient_results
        ]
        if any(result["truth"] == "RETAKE" for result, _ in rows):
            retake_bearing += 1
            false_ready_patients += any(
                result["truth"] == "RETAKE" and decision == "READY"
                for result, decision in rows
            )
        if any(result["truth"] == "READY" for result, _ in rows):
            ready_bearing += 1
            false_retake_patients += any(
                result["truth"] == "READY" and decision == "RETAKE"
                for result, decision in rows
            )
    return {
        "patients": len(patients),
        "retake_bearing_patients": retake_bearing,
        "ready_bearing_patients": ready_bearing,
        "false_ready_patient_count": false_ready_patients,
        "false_retake_patient_count": false_retake_patients,
        "false_ready_patient_rate": (
            false_ready_patients / retake_bearing if retake_bearing else None
        ),
        "false_retake_patient_rate": (
            false_retake_patients / ready_bearing if ready_bearing else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-results", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--false-ready-risk", type=float, default=0.05)
    parser.add_argument("--false-retake-risk", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--unit", choices=("image", "patient"), default="image")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    calibration = load_results(args.calibration_results)
    evaluation = (
        load_results(args.evaluation_results)
        if args.evaluation_results
        else calibration
    )
    policy = calibrate_thresholds(
        calibration,
        false_ready_risk=args.false_ready_risk,
        false_retake_risk=args.false_retake_risk,
        delta=args.delta,
        unit=args.unit,
    )
    payload = {
        "schema_version": 1,
        "policy": policy,
        "calibration": {
            "source": str(args.calibration_results),
            "metrics": evaluate(calibration, policy),
            "patient_event_metrics": evaluate_patient_events(calibration, policy),
        },
        "evaluation": {
            "source": str(args.evaluation_results or args.calibration_results),
            "exploratory_if_previously_opened": bool(args.evaluation_results),
            "metrics": evaluate(evaluation, policy),
            "patient_event_metrics": evaluate_patient_events(evaluation, policy),
        },
        "notes": [
            "Rates are class-conditional error rates, not READY-call precision.",
            "Each gate uses a one-sided exact bound; the simultaneous confidence lower bound uses a union bound.",
            "The guarantee applies only under exchangeability at the stated calibration unit.",
            "The local llama.cpp free-generation path does not expose these direct logits.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
