#!/usr/bin/env python3
"""Evaluate the deployed RetinaReady hybrid through its real HTTP boundary.

Every manifest row stays in the denominator. HTTP failures, timeouts, malformed
responses, specialist abstentions, and Gemma vetoes are all recorded as
LIMITED rather than dropped. The already-open DeepDRiD validation split is the
only accepted default; this tool refuses the official test manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_DECISIONS = frozenset({"READY", "RETAKE", "LIMITED"})
REQUIRED_HEALTH = {
    "status": "ready",
    "mode": "hybrid-local",
    "privacy": "local-only",
    "network_required": False,
    "model_verified": True,
    "lora_verified": True,
    "specialist_verified": True,
}
QUALITY_ATTENTION_LABEL = "Model quality attention \u2014 not pathology localization."
ATTENTION_FACTORS = frozenset({"artifact", "clarity", "field_definition"})
ATTENTION_METHODS = frozenset(
    {"factor-grad-cam", "factor-gradient-sensitivity"}
)
TRACE_SPECIALIST_STATES = frozenset(
    {"READY candidate", "RETAKE candidate", "Abstained", "Input rejected"}
)
TRACE_GEMMA_STATES = frozenset({"Confirmed", "Skipped", "No confirmation"})


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def read_json_url(url: str, *, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def verify_health(base_url: str, *, timeout: float) -> dict[str, Any]:
    health = read_json_url(f"{base_url.rstrip('/')}/api/health", timeout=timeout)
    mismatches = {
        key: {"expected": expected, "actual": health.get(key)}
        for key, expected in REQUIRED_HEALTH.items()
        if health.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"hybrid health verification failed: {mismatches}")
    return health


def attention_metadata(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Validate optional explanation metadata without retaining its image."""

    attention = payload.get("quality_attention")
    if attention is None:
        return None
    if not isinstance(attention, dict):
        raise ValueError("quality_attention was not a JSON object")
    image_data_url = attention.get("image_data_url")
    if (
        attention.get("label") != QUALITY_ATTENTION_LABEL
        or attention.get("factor") not in ATTENTION_FACTORS
        or attention.get("method") not in ATTENTION_METHODS
        or not isinstance(attention.get("factor_label"), str)
        or not isinstance(image_data_url, str)
        or not image_data_url.startswith("data:image/png;base64,")
    ):
        raise ValueError("quality_attention failed the local PNG schema")
    return {
        "factor": attention["factor"],
        "method": attention["method"],
        "encoded_chars": len(image_data_url),
    }


def decision_trace_metadata(payload: dict[str, Any]) -> dict[str, str]:
    """Validate the application-authored, fail-closed decision trace."""

    trace = payload.get("decision_trace")
    if not isinstance(trace, dict):
        raise ValueError("API response omitted decision_trace")
    specialist = trace.get("specialist")
    gemma = trace.get("gemma")
    policy = trace.get("policy")
    decision = payload.get("status")
    if (
        specialist not in TRACE_SPECIALIST_STATES
        or gemma not in TRACE_GEMMA_STATES
        or policy not in VALID_DECISIONS
        or policy != decision
    ):
        raise ValueError("decision_trace failed its enumerated schema")
    if gemma == "Confirmed":
        if specialist != f"{policy} candidate" or policy == "LIMITED":
            raise ValueError("confirmed decision_trace is inconsistent")
    elif gemma == "Skipped":
        if specialist not in {"Abstained", "Input rejected"} or policy != "LIMITED":
            raise ValueError("skipped decision_trace is inconsistent")
    elif not specialist.endswith(" candidate") or policy != "LIMITED":
        raise ValueError("unconfirmed decision_trace is inconsistent")
    return {"specialist": specialist, "gemma": gemma, "policy": policy}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    required = {"split", "patient_id", "image_id", "image_path", "quality_label"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if path.name == "test.csv" or any(row["split"] == "test" for row in rows):
        raise ValueError("hybrid evaluator refuses the already-open official test split")
    for row in rows:
        if row["split"] != "val":
            raise ValueError(f"expected val row, found {row['split']!r}")
        if row["quality_label"] not in {"READY", "RETAKE"}:
            raise ValueError(f"invalid quality label: {row['quality_label']!r}")
        image_path = resolve(Path(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    return rows


def load_specialist_decisions(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    policy = report["selective_policy"]
    ready_threshold = float(policy["ready_threshold_strictly_greater_than"])
    retake_threshold = float(policy["retake_threshold_strictly_less_than"])
    decisions: dict[str, str] = {}
    for result in report["results"]:
        score = float(result["decision_score"])
        decision = (
            "READY"
            if score > ready_threshold
            else "RETAKE"
            if score < retake_threshold
            else "LIMITED"
        )
        decisions[result["image_id"]] = decision
    return decisions, policy


def request_analysis(
    base_url: str,
    *,
    image_path: Path,
    timeout: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    request = Request(
        f"{base_url.rstrip('/')}/api/analyze",
        data=image_path.read_bytes(),
        headers={"Content-Type": "image/jpeg", "X-Filename": image_path.name},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("API response was not a JSON object")
        decision = payload.get("status")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"invalid API decision: {decision!r}")
        meta = payload.get("meta")
        if not isinstance(meta, dict) or not isinstance(meta.get("latency_ms"), (int, float)):
            raise ValueError("API response omitted numeric meta.latency_ms")
        attention_metadata(payload)
        decision_trace_metadata(payload)
        return payload, None, (time.perf_counter() - started) * 1000
    except HTTPError as error:
        return None, f"HTTP {error.code}", (time.perf_counter() - started) * 1000
    except (URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as error:
        return None, f"{type(error).__name__}: {error}", (time.perf_counter() - started) * 1000


def _patient_events(records: list[dict[str, Any]]) -> dict[str, Any]:
    patients: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        patients.setdefault(record["patient_id"], []).append(record)
    retake_bearing = [
        rows for rows in patients.values() if any(row["truth"] == "RETAKE" for row in rows)
    ]
    ready_bearing = [
        rows for rows in patients.values() if any(row["truth"] == "READY" for row in rows)
    ]
    false_ready = sum(
        any(row["truth"] == "RETAKE" and row["decision"] == "READY" for row in rows)
        for rows in retake_bearing
    )
    false_retake = sum(
        any(row["truth"] == "READY" and row["decision"] == "RETAKE" for row in rows)
        for rows in ready_bearing
    )
    return {
        "patients": len(patients),
        "retake_bearing_patients": len(retake_bearing),
        "ready_bearing_patients": len(ready_bearing),
        "false_ready_patient_count": false_ready,
        "false_ready_patient_rate": false_ready / len(retake_bearing) if retake_bearing else None,
        "false_retake_patient_count": false_retake,
        "false_retake_patient_rate": false_retake / len(ready_bearing) if ready_bearing else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    truth = Counter(record["truth"] for record in records)
    decisions = Counter(record["decision"] for record in records)
    accepted = [record for record in records if record["decision"] != "LIMITED"]
    false_ready = sum(
        record["truth"] == "RETAKE" and record["decision"] == "READY"
        for record in records
    )
    false_retake = sum(
        record["truth"] == "READY" and record["decision"] == "RETAKE"
        for record in records
    )
    wall_latencies = [float(record["wall_latency_ms"]) for record in records]
    api_latencies = [
        float(record["api_latency_ms"])
        for record in records
        if isinstance(record.get("api_latency_ms"), (int, float))
    ]
    attention_records = [
        record for record in records if bool(record.get("attention_present"))
    ]
    trace_records = [record for record in records if record.get("trace_policy")]
    return {
        "images": len(records),
        "truth_counts": {"READY": truth["READY"], "RETAKE": truth["RETAKE"]},
        "decision_counts": {
            "READY": decisions["READY"],
            "RETAKE": decisions["RETAKE"],
            "LIMITED": decisions["LIMITED"],
        },
        "coverage": len(accepted) / len(records),
        "accepted_accuracy": (
            sum(record["decision"] == record["truth"] for record in accepted) / len(accepted)
            if accepted
            else None
        ),
        "false_ready_count": false_ready,
        "false_ready_rate_given_retake": false_ready / truth["RETAKE"] if truth["RETAKE"] else None,
        "false_retake_count": false_retake,
        "false_retake_rate_given_ready": false_retake / truth["READY"] if truth["READY"] else None,
        "patient_event_metrics": _patient_events(records),
        "flow": {
            "specialist_limited": sum(record["flow"] == "specialist_limited" for record in records),
            "gemma_confirmed": sum(record["flow"] == "gemma_confirmed" for record in records),
            "gemma_veto_or_internal_abstention": sum(
                record["flow"] == "gemma_veto_or_internal_abstention" for record in records
            ),
            "http_or_schema_failure": sum(record["flow"] == "http_or_schema_failure" for record in records),
        },
        "latency_ms": {
            "wall_median": statistics.median(wall_latencies),
            "wall_p95": percentile(wall_latencies, 0.95),
            "api_median": statistics.median(api_latencies) if api_latencies else None,
            "api_p95": percentile(api_latencies, 0.95),
        },
        "quality_attention": {
            "present": len(attention_records),
            "retake_decisions": sum(
                record["decision"] == "RETAKE" for record in attention_records
            ),
            "unexpected_non_retake": sum(
                record["decision"] != "RETAKE" for record in attention_records
            ),
            "retake_without_attention": sum(
                record["decision"] == "RETAKE"
                and not bool(record.get("attention_present"))
                for record in records
            ),
            "methods": dict(
                Counter(
                    str(record["attention_method"])
                    for record in attention_records
                )
            ),
        },
        "decision_trace": {
            "present": len(trace_records),
            "missing": len(records) - len(trace_records),
            "paths": dict(
                Counter(
                    " -> ".join(
                        (
                            str(record["trace_specialist"]),
                            str(record["trace_gemma"]),
                            str(record["trace_policy"]),
                        )
                    )
                    for record in trace_records
                )
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/val.csv"))
    parser.add_argument(
        "--specialist-report",
        type=Path,
        default=Path("outputs/quality-specialist-rigorous-factors/report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hybrid-validation-exploratory.json"),
    )
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("timeout and limit must be positive")
    manifest_path = resolve(args.manifest)
    specialist_report_path = resolve(args.specialist_report)
    output_path = resolve(args.output)
    rows = read_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    specialist_decisions, policy = load_specialist_decisions(specialist_report_path)
    missing = [row["image_id"] for row in rows if row["image_id"] not in specialist_decisions]
    if missing:
        raise ValueError(f"specialist report is missing manifest images: {missing[:5]}")

    health = verify_health(args.base_url, timeout=min(args.timeout, 10.0))
    records: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        image_path = resolve(Path(row["image_path"]))
        payload, error, wall_latency = request_analysis(
            args.base_url, image_path=image_path, timeout=args.timeout
        )
        specialist_decision = specialist_decisions[row["image_id"]]
        if payload is None:
            decision = "LIMITED"
            api_latency = None
            flow = "http_or_schema_failure"
            attention = None
            trace = None
        else:
            decision = str(payload["status"])
            api_latency = float(payload["meta"]["latency_ms"])
            if specialist_decision == "LIMITED":
                flow = "specialist_limited"
            elif decision == specialist_decision:
                flow = "gemma_confirmed"
            else:
                flow = "gemma_veto_or_internal_abstention"
            attention = attention_metadata(payload)
            trace = decision_trace_metadata(payload)
        records.append(
            {
                "patient_id": row["patient_id"],
                "image_id": row["image_id"],
                "truth": row["quality_label"],
                "specialist_decision": specialist_decision,
                "decision": decision,
                "flow": flow,
                "api_latency_ms": api_latency,
                "wall_latency_ms": wall_latency,
                "error": error,
                "attention_present": attention is not None,
                "attention_factor": attention["factor"] if attention else None,
                "attention_method": attention["method"] if attention else None,
                "attention_encoded_chars": (
                    attention["encoded_chars"] if attention else None
                ),
                "trace_specialist": trace["specialist"] if trace else None,
                "trace_gemma": trace["gemma"] if trace else None,
                "trace_policy": trace["policy"] if trace else None,
            }
        )
        if index % 25 == 0 or index == len(rows):
            print(f"Evaluated {index}/{len(rows)} images")

    report = {
        "schema_version": 1,
        "evaluation_type": "exploratory end-to-end deployed hybrid evaluation",
        "health": health,
        "data": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(manifest_path),
            "images": len(rows),
            "official_test_used": False,
            "validation_history": "viewed repeatedly during development; exploratory only",
        },
        "specialist": {
            "report": str(args.specialist_report),
            "report_sha256": sha256_file(specialist_report_path),
            "policy": policy,
        },
        "metrics": summarize(records),
        "runtime": {"total_seconds": time.perf_counter() - run_started},
        "limitations": [
            "DeepDRiD validation was viewed during model development; metrics are exploratory.",
            "The API cannot distinguish a Gemma disagreement from an internal Gemma timeout after fail-closed normalization.",
            "The current patient thresholds are nominal/post-hoc, not a fresh deployment guarantee.",
            "This is technical capture-quality research, not diagnosis or clinical validation.",
        ],
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": report["metrics"], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
