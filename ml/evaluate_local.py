#!/usr/bin/env python3
"""Evaluate local RetinaReady inference against a DeepDRiD manifest.

This is deliberately a safety-first evaluation:

* every selected manifest row stays in the metric denominator;
* schema-invalid responses and request failures fail closed to LIMITED;
* LIMITED is not treated as a correct binary prediction;
* false-READY rate is measured over every ground-truth RETAKE image.

The script only sends requests to the configured llama.cpp endpoint. With the
default 127.0.0.1 URL, image data remains on the local machine.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from infer_local import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    extract_object,
    image_data_url,
    limited_fallback,
    normalize,
    request_json,
)

GROUND_TRUTH_LABELS = ("READY", "RETAKE")
PREDICTION_LABELS = ("READY", "RETAKE", "LIMITED")
REQUIRED_COLUMNS = {"image_id", "image_path", "patient_id", "quality_label"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_project(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else project_root() / value


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"manifest is empty: {path}")

    image_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        if not image_id:
            raise ValueError(f"{path}:{index}: image_id is empty")
        if image_id in image_ids:
            raise ValueError(f"{path}:{index}: duplicate image_id {image_id!r}")
        image_ids.add(image_id)

        label = row["quality_label"].strip().upper()
        if label not in GROUND_TRUTH_LABELS:
            raise ValueError(
                f"{path}:{index}: quality_label must be READY or RETAKE, got {label!r}"
            )
        row["quality_label"] = label

        image_path = resolve_from_project(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{path}:{index}: image does not exist: {image_path}"
            )
    return rows


def select_rows(
    rows: list[dict[str, str]],
    *,
    limit: int | None,
    sampling: str,
    seed: int,
) -> list[dict[str, str]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        raise ValueError("--limit must be greater than zero")

    if sampling == "sequential":
        return rows[:limit]

    rng = random.Random(seed)
    if sampling == "random":
        # Return samples in draw order, which is stable for a fixed Python
        # version, manifest, and seed.
        return rng.sample(rows, limit)

    if sampling != "stratified":
        raise ValueError(f"unsupported sampling strategy: {sampling!r}")

    groups = {
        label: [row for row in rows if row["quality_label"] == label]
        for label in GROUND_TRUTH_LABELS
    }
    for group in groups.values():
        rng.shuffle(group)

    # Allocate proportionally using largest remainders. When the limit can hold
    # both classes, reserve one row per present class so a smoke evaluation can
    # exercise both safety outcomes.
    nonempty = [label for label, group in groups.items() if group]
    counts = {label: 0 for label in GROUND_TRUTH_LABELS}
    remaining = limit
    if limit >= len(nonempty):
        for label in nonempty:
            counts[label] = 1
            remaining -= 1

    available = {
        label: len(groups[label]) - counts[label] for label in GROUND_TRUTH_LABELS
    }
    total_available = sum(available.values())
    if remaining and total_available:
        exact = {
            label: remaining * available[label] / total_available
            for label in GROUND_TRUTH_LABELS
        }
        for label in GROUND_TRUTH_LABELS:
            extra = min(available[label], int(exact[label]))
            counts[label] += extra
            remaining -= extra
        order = sorted(
            GROUND_TRUTH_LABELS,
            key=lambda label: (exact[label] - int(exact[label]), label),
            reverse=True,
        )
        while remaining:
            progressed = False
            for label in order:
                if counts[label] < len(groups[label]):
                    counts[label] += 1
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                break

    selected: list[dict[str, str]] = []
    for label in GROUND_TRUTH_LABELS:
        selected.extend(groups[label][: counts[label]])
    rng.shuffle(selected)
    return selected


def build_payload(row: dict[str, str], model: str, max_tokens: int) -> dict[str, Any]:
    image_path = resolve_from_project(row["image_path"])
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path)},
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def check_health(base_url: str, timeout: float) -> dict[str, Any]:
    """Check llama.cpp health once so an offline server does not fail per row."""

    parts = urllib.parse.urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    health_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, f"{path.rstrip('/')}/health", "", "")
    )
    with urllib.request.urlopen(health_url, timeout=min(timeout, 10)) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict) or value.get("status") != "ok":
        raise ValueError(f"model server is not ready: {value!r}")
    return value


def run_one(
    row: dict[str, str],
    *,
    endpoint: str,
    model: str,
    timeout: float,
    max_tokens: int,
    include_output: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    schema_valid = False
    request_succeeded = False
    error: str | None = None
    raw_content: str | None = None
    try:
        response = request_json(
            endpoint,
            build_payload(row, model=model, max_tokens=max_tokens),
            timeout,
        )
        request_succeeded = True
        raw_content = response["choices"][0]["message"]["content"]
        if not isinstance(raw_content, str):
            raise TypeError("model message content is not a string")
        prediction = normalize(extract_object(raw_content))
        schema_valid = True
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        urllib.error.URLError,
    ) as exc:
        error = f"{type(exc).__name__}: {exc}"
        prediction = limited_fallback(error)
    latency_ms = (time.perf_counter() - started) * 1000

    result: dict[str, Any] = {
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "image_path": row["image_path"],
        "truth": row["quality_label"],
        "prediction": prediction["decision"],
        "schema_valid": schema_valid,
        "request_succeeded": request_succeeded,
        "latency_ms": round(latency_ms, 3),
    }
    if error is not None:
        result["error"] = error
    if include_output:
        result["output"] = {
            key: value for key, value in prediction.items() if not key.startswith("_")
        }
        if raw_content is not None and not schema_valid:
            result["raw_content"] = raw_content
    return result


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def summarize(results: list[dict[str, Any]], wall_time_seconds: float) -> dict[str, Any]:
    truth_counts = {
        label: sum(result["truth"] == label for result in results)
        for label in GROUND_TRUTH_LABELS
    }
    prediction_counts = {
        label: sum(result["prediction"] == label for result in results)
        for label in PREDICTION_LABELS
    }
    outcome_matrix = {
        truth: {
            prediction: sum(
                result["truth"] == truth and result["prediction"] == prediction
                for result in results
            )
            for prediction in PREDICTION_LABELS
        }
        for truth in GROUND_TRUTH_LABELS
    }
    binary_matrix = {
        truth: {
            prediction: outcome_matrix[truth][prediction]
            for prediction in GROUND_TRUTH_LABELS
        }
        for truth in GROUND_TRUTH_LABELS
    }

    total = len(results)
    correct = sum(result["truth"] == result["prediction"] for result in results)
    binary_decisions = prediction_counts["READY"] + prediction_counts["RETAKE"]
    binary_correct = (
        binary_matrix["READY"]["READY"] + binary_matrix["RETAKE"]["RETAKE"]
    )
    actual_retake = truth_counts["RETAKE"]
    false_ready = outcome_matrix["RETAKE"]["READY"]
    predicted_retake_on_retake = outcome_matrix["RETAKE"]["RETAKE"]
    latencies = [float(result["latency_ms"]) for result in results]
    schema_valid = sum(bool(result["schema_valid"]) for result in results)
    request_succeeded = sum(bool(result["request_succeeded"]) for result in results)

    return {
        "samples": total,
        "truth_counts": truth_counts,
        "prediction_counts": prediction_counts,
        "request_success_count": request_succeeded,
        "request_success_rate": rounded(ratio(request_succeeded, total)),
        "schema_valid_count": schema_valid,
        "schema_valid_rate": rounded(ratio(schema_valid, total)),
        "outcome_matrix": outcome_matrix,
        "binary_confusion_matrix": binary_matrix,
        "limited_predictions": prediction_counts["LIMITED"],
        "metrics": {
            # LIMITED and invalid responses remain in this denominator.
            "accuracy": rounded(ratio(correct, total)),
            # Diagnostic only; excludes LIMITED predictions and must not replace
            # the conservative headline accuracy above.
            "accuracy_among_binary_decisions": rounded(
                ratio(binary_correct, binary_decisions)
            ),
            "retake_recall": rounded(
                ratio(predicted_retake_on_retake, actual_retake)
            ),
            "false_ready_rate": rounded(ratio(false_ready, actual_retake)),
            "false_ready_count": false_ready,
            "actual_retake_count": actual_retake,
        },
        "latency_ms": {
            "median": rounded(statistics.median(latencies), 3) if latencies else None,
            "p95": rounded(percentile(latencies, 0.95), 3),
            "min": rounded(min(latencies), 3) if latencies else None,
            "max": rounded(max(latencies), 3) if latencies else None,
            "wall_time_seconds": rounded(wall_time_seconds, 3),
        },
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the local RetinaReady llama.cpp endpoint on a DeepDRiD "
            "manifest. Invalid responses fail closed to LIMITED."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/val.csv"),
        help="CSV manifest, relative to retina-ready (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8081/v1",
        help="OpenAI-compatible llama.cpp base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RETINA_READY_MODEL_ALIAS", "retinaready-gemma4-26b"),
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate at most this many selected manifest rows",
    )
    parser.add_argument(
        "--sampling",
        choices=("sequential", "random", "stratified"),
        default="sequential",
        help="selection method when --limit is set (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for random or stratified selection (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the full JSON report to this path",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit normalized model outputs from per-image result records",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print progress every N samples; 0 disables it (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="skip the llama.cpp /health preflight (use only for compatible proxies)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")

    manifest = resolve_from_project(args.manifest)
    try:
        rows = read_manifest(manifest)
        selected = select_rows(
            rows,
            limit=args.limit,
            sampling=args.sampling,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if not args.skip_health_check:
        try:
            check_health(args.base_url, args.timeout)
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            parser.error(f"model server health check failed: {exc}")

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    started_at = utc_now()
    wall_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        result = run_one(
            row,
            endpoint=endpoint,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            include_output=not args.summary_only,
        )
        results.append(result)
        if args.progress_every and (
            index % args.progress_every == 0 or index == len(selected)
        ):
            print(
                f"[{index}/{len(selected)}] {result['image_id']}: "
                f"{result['truth']} -> {result['prediction']} "
                f"({result['latency_ms']:.0f} ms)",
                file=sys.stderr,
                flush=True,
            )

    wall_time_seconds = time.perf_counter() - wall_started
    report = {
        "run": {
            "manifest": str(manifest),
            "base_url": args.base_url,
            "model": args.model,
            "selection": {
                "available_rows": len(rows),
                "selected_rows": len(selected),
                "limit": args.limit,
                "sampling": args.sampling,
                "seed": args.seed,
            },
            "started_at": started_at,
            "completed_at": utc_now(),
        },
        "summary": summarize(results, wall_time_seconds),
        "results": results,
    }

    summary_text = json.dumps(report["summary"], indent=2, sort_keys=True)
    print(summary_text)
    if args.output is not None:
        output = resolve_from_project(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {output}", file=sys.stderr)

    # A completed evaluation can contain poor model results. Exit nonzero only
    # when every request failed, which usually means the endpoint was unavailable.
    if report["summary"]["request_success_count"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
