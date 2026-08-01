#!/usr/bin/env python3
"""Evaluate a quality or escalation decision-token PEFT adapter on CUDA.

The corrected classifier smoke run supervises exactly one token per image: the
first token of the JSON ``decision`` value. This evaluator reconstructs the
same multimodal training sequence, verifies the assistant mask, truncates the
sequence immediately before that token, and compares the two pinned class
logits directly. It does not generate the remainder of the JSON response.

Dry-run mode validates paths, manifest sampling, adapter provenance, and the
pinned scoring contract without importing CUDA libraries or loading weights.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from evaluate_peft import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_PROCESSOR,
    DEFAULT_PROCESSOR_REVISION,
    adapter_metadata,
    load_runtime,
    resolve_project_path,
    validate_adapter_base,
)
from train_qlora import (
    assistant_only_labels,
    decision_token_only_labels,
    read_rows,
    render_generation_prompt,
    render_training_text,
    select_rows_for_run,
    task_contract,
)


READY_TOKEN_ID = 156274
RETAKE_TOKEN_ID = 1357
QUALITY_CLASS_TOKEN_IDS = {"READY": READY_TOKEN_ID, "RETAKE": RETAKE_TOKEN_ID}


def task_scoring_contract(task: str) -> dict[str, Any]:
    contract = task_contract(task)
    if task == "quality":
        return {
            **contract,
            "positive_label": "READY",
            "negative_label": "RETAKE",
            "positive_probability_name": "ready_probability",
            "pinned_token_ids": QUALITY_CLASS_TOKEN_IDS,
            "equality_fails_to": "RETAKE",
        }
    return {
        **contract,
        "positive_label": "PRIORITY",
        "negative_label": "ROUTINE",
        "positive_probability_name": "priority_probability",
        "pinned_token_ids": None,
        "equality_fails_to": "PRIORITY",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_empty_thought_channel(
    effective_config: dict[str, Any], model_id: str
) -> bool:
    """Resolve the trainer's tri-state empty-thought-channel setting."""

    configured = effective_config.get("empty_thought_channel")
    if configured is not None:
        if not isinstance(configured, bool):
            raise ValueError(
                "adapter provenance effective_config.empty_thought_channel must "
                "be true, false, or null"
            )
        return configured
    lowered = model_id.lower()
    return lowered.endswith("-it") and ("26b" in lowered or "31b" in lowered)


def load_decision_training_contract(
    metadata: dict[str, Any], model_id: str, task: str = "quality"
) -> dict[str, Any]:
    """Validate that the adapter was completed under the one-token objective."""

    provenance_path = Path(metadata["provenance_path"])
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not isinstance(provenance, dict):
        raise ValueError("run_provenance.json must contain one JSON object")
    effective_config = provenance.get("effective_config")
    if not isinstance(effective_config, dict):
        raise ValueError("adapter provenance is missing effective_config")
    if effective_config.get("loss_scope") != "decision_token":
        raise ValueError(
            "adapter was not trained with loss_scope='decision_token'; direct "
            "class-logit evaluation would not match its training objective"
        )
    trained_task = effective_config.get("task", "quality")
    if trained_task != task:
        raise ValueError(
            f"adapter was trained for task={trained_task!r}, not requested task={task!r}"
        )
    return {
        "task": trained_task,
        "loss_scope": "decision_token",
        "empty_thought_channel": effective_empty_thought_channel(
            effective_config, model_id
        ),
        "stratified_sampling": effective_config.get("stratified_sampling"),
        "train_seed": effective_config.get("seed"),
        "provenance_sha256": sha256_file(provenance_path),
        "selected_data": provenance.get("selected_data"),
    }


def validate_class_tokens(tokenizer: Any, task: str = "quality") -> dict[str, Any]:
    """Fail if the pinned processor no longer implements the trained labels."""

    scoring = task_scoring_contract(task)
    labels = scoring["labels"]
    encoded = {
        label: tokenizer.encode(label, add_special_tokens=False)
        for label in labels
    }
    if any(not token_ids for token_ids in encoded.values()):
        raise ValueError(f"one or more {task} labels encoded to zero tokens: {encoded}")
    first_ids = {label: token_ids[0] for label, token_ids in encoded.items()}
    if len(set(first_ids.values())) != len(first_ids):
        raise ValueError(
            f"{task} class labels do not have distinct first tokens: {first_ids}"
        )
    pinned = scoring["pinned_token_ids"]
    if pinned is not None and first_ids != pinned:
        raise ValueError(
            f"quality class-token contract changed: expected {pinned}, got {first_ids}"
        )
    return {
        label: {
            "token_id": first_ids[label],
            "full_encoding": encoded[label],
            "decoded_token": tokenizer.decode([first_ids[label]]),
        }
        for label in labels
    }


def prepare_full_batch(
    row: dict[str, str],
    *,
    image: Any,
    processor: Any,
    torch: Any,
    include_empty_thought_channel: bool,
    task: str = "quality",
    class_token_ids: dict[str, int] | None = None,
) -> tuple[dict[str, Any], int]:
    """Rebuild the training batch and return its unique class-token position."""

    full_text = render_training_text(
        processor, row, include_empty_thought_channel, task=task
    )
    prompt_text = render_generation_prompt(processor, row, task=task)
    # Keep the nested image list identical to train_qlora.py's collator.
    images = [[image]]
    batch = processor(
        text=[full_text],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    prompt_batch = processor(
        text=[prompt_text],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    assistant_labels = assistant_only_labels(batch, prompt_batch, torch)
    decision_labels = decision_token_only_labels(
        batch,
        assistant_labels,
        [row],
        processor.tokenizer,
        torch,
        task=task,
    )
    positions = torch.nonzero(decision_labels[0] != -100, as_tuple=False).flatten()
    if positions.numel() != 1:
        raise ValueError(
            f"{row['image_id']}: expected exactly one decision-token position, "
            f"found {positions.numel()}"
        )
    position = int(positions.item())
    observed = int(batch["input_ids"][0, position].item())
    label_field = task_contract(task)["label_field"]
    if class_token_ids is None:
        class_token_ids = {
            label: processor.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in task_contract(task)["labels"]
        }
    expected = class_token_ids[row[label_field]]
    if observed != expected:
        raise ValueError(
            f"{row['image_id']}: decision token is {observed}, expected {expected} "
            f"for {row[label_field]}"
        )
    return dict(batch), position


def truncate_before_decision(
    batch: dict[str, Any], position: int
) -> dict[str, Any]:
    """Drop the ground-truth class token and all later assistant tokens."""

    if position <= 0:
        raise ValueError("decision token cannot be the first input token")
    sequence_length = int(batch["input_ids"].shape[1])
    if position >= sequence_length:
        raise ValueError("decision token position lies outside the input sequence")

    inputs: dict[str, Any] = {}
    # Gemma4Processor emits ``mm_token_type_ids`` by default. It participates
    # in causal/bidirectional mask construction and must be truncated with the
    # text sequence; leaving its full training length would either fail shape
    # validation or accidentally describe tokens that are no longer present.
    sequence_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
        "position_ids",
    }
    for key, value in batch.items():
        if key in sequence_keys:
            if getattr(value, "ndim", 0) != 2 or value.shape[1] != sequence_length:
                raise ValueError(
                    f"unexpected {key} shape for sequence truncation: "
                    f"{getattr(value, 'shape', None)}"
                )
            inputs[key] = value[:, :position]
        else:
            inputs[key] = value
    return inputs


def validate_prefix_does_not_encode_truth(
    row: dict[str, str],
    *,
    image: Any,
    processor: Any,
    torch: Any,
    include_empty_thought_channel: bool,
    task: str = "quality",
    class_token_ids: dict[str, int] | None = None,
) -> dict[str, int]:
    """Prove both task labels share the exact scored token prefix."""

    contract = task_contract(task)
    labels = contract["labels"]
    label_field = contract["label_field"]
    batches: dict[str, tuple[dict[str, Any], int]] = {}
    for label in labels:
        synthetic = dict(row)
        synthetic[label_field] = label
        if task == "escalation":
            synthetic["dr_grade"] = "2" if label == "PRIORITY" else "0"
        batches[label] = prepare_full_batch(
            synthetic,
            image=image,
            processor=processor,
            torch=torch,
            include_empty_thought_channel=include_empty_thought_channel,
            task=task,
            class_token_ids=class_token_ids,
        )
    positive, negative = labels
    positive_batch, positive_position = batches[positive]
    negative_batch, negative_position = batches[negative]
    positive_prefix = positive_batch["input_ids"][0, :positive_position]
    negative_prefix = negative_batch["input_ids"][0, :negative_position]
    if positive_position != negative_position or not torch.equal(
        positive_prefix, negative_prefix
    ):
        raise ValueError(
            "the teacher-forced prefix differs by class; refusing an evaluation "
            "that could leak the ground-truth decision"
        )
    return {
        "decision_position": positive_position,
        "prefix_tokens": int(positive_prefix.numel()),
    }


def stable_ready_probability(ready_minus_retake_logit: float) -> float:
    """Compute a stable two-class softmax probability for READY."""

    if ready_minus_retake_logit >= 0:
        return 1.0 / (1.0 + math.exp(-ready_minus_retake_logit))
    exponent = math.exp(ready_minus_retake_logit)
    return exponent / (1.0 + exponent)


def run_one(
    row: dict[str, str],
    *,
    model: Any,
    processor: Any,
    torch: Any,
    include_empty_thought_channel: bool,
    decision_threshold: float,
    task: str = "quality",
    class_token_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Score one image from the exact prefix immediately before its class token."""

    from PIL import Image

    image_path = resolve_project_path(row["image_path"])
    started = time.perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    try:
        batch, decision_position = prepare_full_batch(
            row,
            image=image,
            processor=processor,
            torch=torch,
            include_empty_thought_channel=include_empty_thought_channel,
            task=task,
            class_token_ids=class_token_ids,
        )
        inputs = truncate_before_decision(batch, decision_position)
        device = model.device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        torch.cuda.synchronize()
        with torch.inference_mode():
            # The pinned Gemma 4 implementation accepts logits_to_keep and PEFT
            # forwards it, avoiding allocation of logits for the entire prefix.
            output = model(
                **inputs,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        torch.cuda.synchronize()
    finally:
        image.close()

    scoring = task_scoring_contract(task)
    positive_label = scoring["positive_label"]
    negative_label = scoring["negative_label"]
    label_field = scoring["label_field"]
    if class_token_ids is None:
        class_token_ids = {
            label: processor.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in scoring["labels"]
        }
    logits = output.logits[0, -1]
    positive_logit = float(logits[class_token_ids[positive_label]].float().item())
    negative_logit = float(logits[class_token_ids[negative_label]].float().item())
    if not math.isfinite(positive_logit) or not math.isfinite(negative_logit):
        raise ValueError(f"{row['image_id']}: class logits are not finite")
    margin = positive_logit - negative_logit
    positive_probability = stable_ready_probability(margin)
    if task == "quality":
        prediction = (
            positive_label
            if positive_probability > decision_threshold
            else negative_label
        )
    else:
        # Equality fails closed to priority review; routine is released only below
        # a threshold frozen on patient-disjoint calibration data.
        prediction = (
            positive_label
            if positive_probability >= decision_threshold
            else negative_label
        )
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "image_id": row["image_id"],
        "patient_id": row["patient_id"],
        "image_path": row["image_path"],
        "truth": row[label_field],
        "prediction": prediction,
        "decision_position": decision_position,
        "prefix_tokens": int(inputs["input_ids"].shape[1]),
        # Preserve full Python-float precision so near-ties remain ordered for
        # ROC-AUC. Aggregate display metrics are rounded only in summarize().
        "positive_label": positive_label,
        "negative_label": negative_label,
        "positive_logit": positive_logit,
        "negative_logit": negative_logit,
        "positive_minus_negative_logit": margin,
        "positive_probability": positive_probability,
        "negative_probability": 1.0 - positive_probability,
        "latency_ms": round(latency_ms, 3),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def positive_roc_auc(
    results: list[dict[str, Any]], positive_label: str
) -> float | None:
    """Return ROC-AUC via the Mann-Whitney definition, including score ties."""

    positives = [
        float(result["positive_probability"])
        for result in results
        if result["truth"] == positive_label
    ]
    negatives = [
        float(result["positive_probability"])
        for result in results
        if result["truth"] != positive_label
    ]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def summarize(
    results: list[dict[str, Any]], wall_time_seconds: float, task: str = "quality"
) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize zero results")
    scoring = task_scoring_contract(task)
    labels = scoring["labels"]
    positive_label = scoring["positive_label"]
    negative_label = scoring["negative_label"]
    for result in results:
        if result.get("truth") not in labels:
            raise ValueError(f"invalid truth label in result: {result.get('truth')!r}")
        if result.get("prediction") not in labels:
            raise ValueError(
                f"invalid prediction label in result: {result.get('prediction')!r}"
            )

    matrix = {
        truth: {
            prediction: sum(
                result["truth"] == truth and result["prediction"] == prediction
                for result in results
            )
            for prediction in labels
        }
        for truth in labels
    }
    truth_counts = {
        truth: sum(result["truth"] == truth for result in results)
        for truth in labels
    }
    prediction_counts = {
        prediction: sum(result["prediction"] == prediction for result in results)
        for prediction in labels
    }
    positive_recall = ratio(
        matrix[positive_label][positive_label], truth_counts[positive_label]
    )
    negative_recall = ratio(
        matrix[negative_label][negative_label], truth_counts[negative_label]
    )
    balanced_accuracy = (
        (positive_recall + negative_recall) / 2
        if positive_recall is not None and negative_recall is not None
        else None
    )
    correct = sum(matrix[label][label] for label in labels)
    false_positive = matrix[negative_label][positive_label]
    false_negative = matrix[positive_label][negative_label]
    latencies = [float(result["latency_ms"]) for result in results]
    return {
        "samples": len(results),
        "truth_counts": truth_counts,
        "prediction_counts": prediction_counts,
        "confusion_matrix": matrix,
        "metrics": {
            "accuracy": rounded(ratio(correct, len(results))),
            "balanced_accuracy": rounded(balanced_accuracy),
            "positive_label": positive_label,
            "positive_recall": rounded(positive_recall),
            "negative_label": negative_label,
            "negative_recall": rounded(negative_recall),
            "false_positive_rate": rounded(
                ratio(false_positive, truth_counts[negative_label])
            ),
            "false_positive_count": false_positive,
            "false_negative_rate": rounded(
                ratio(false_negative, truth_counts[positive_label])
            ),
            "false_negative_count": false_negative,
            "roc_auc_positive": rounded(positive_roc_auc(results, positive_label)),
        },
        "latency_ms": {
            "mean": rounded(sum(latencies) / len(latencies), 3),
            "min": rounded(min(latencies), 3),
            "max": rounded(max(latencies), 3),
            "wall_time_seconds": rounded(wall_time_seconds, 3),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completed decision-token PEFT adapter by comparing the "
            "selected task's class logits without autoregressive generation."
        )
    )
    parser.add_argument(
        "--task", choices=("quality", "escalation"), default="quality"
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--processor-id", default=DEFAULT_PROCESSOR)
    parser.add_argument("--processor-revision", default=DEFAULT_PROCESSOR_REVISION)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/val.csv")
    )
    parser.add_argument("--expected-split", default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sampling",
        choices=("sequential", "random", "stratified"),
        default="stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decision-threshold",
        "--ready-threshold",
        dest="decision_threshold",
        type=float,
        default=0.5,
        help=(
            "pre-calibrated positive-class threshold; equality fails closed to "
            "RETAKE for quality and PRIORITY for escalation"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    task_contract(args.task)
    for name in (
        "model_id",
        "model_revision",
        "processor_id",
        "processor_revision",
    ):
        if not getattr(args, name):
            raise ValueError(f"{name} must not be empty")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be greater than zero")
    if args.progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    if not args.expected_split:
        raise ValueError("expected_split must not be empty")
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError("decision_threshold must be strictly between zero and one")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        metadata = adapter_metadata(args.adapter_dir)
        if metadata is None:  # Required by argparse; keeps type narrowing explicit.
            raise ValueError("--adapter-dir is required")
        validate_adapter_base(
            metadata,
            args.model_id,
            args.model_revision,
            args.processor_id,
            args.processor_revision,
        )
        training_contract = load_decision_training_contract(
            metadata, args.model_id, task=args.task
        )
        manifest = resolve_project_path(args.manifest)
        rows = read_rows(manifest, args.expected_split, task=args.task)
        if args.sampling == "sequential":
            selected = list(rows if args.limit is None else rows[: args.limit])
        else:
            selected = select_rows_for_run(
                rows,
                args.limit,
                seed=args.seed,
                stratified=args.sampling == "stratified",
                task=args.task,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    scoring = task_scoring_contract(args.task)
    setup = {
        "mode": "decision-token-logits",
        "task": args.task,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "processor_id": args.processor_id,
        "processor_revision": args.processor_revision,
        "adapter": {
            **metadata,
            "weights_sha256": sha256_file(Path(metadata["weights_path"])),
        },
        "training_contract": training_contract,
        "class_token_ids": scoring["pinned_token_ids"],
        "classification_rule": (
            f"positive={scoring['positive_label']}; equality fails closed to "
            f"{scoring['equality_fails_to']}"
        ),
        "decision_threshold": args.decision_threshold,
        "roc_auc_positive_class": scoring["positive_label"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        # Keep the split contract in the score artifact itself.  Older reports
        # did not include this field; downstream calibration can still verify
        # them against a hash-bound full manifest, but new reports are
        # self-describing.
        "expected_split": args.expected_split,
        "available_rows": len(rows),
        "selected_rows": len(selected),
        "sampling": args.sampling,
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "setup": setup}, indent=2))
        return 0

    model, processor, torch = load_runtime(args, metadata)
    processor.tokenizer.padding_side = "right"
    token_contract = validate_class_tokens(processor.tokenizer, task=args.task)
    class_token_ids = {
        label: int(details["token_id"])
        for label, details in token_contract.items()
    }

    # One class-invariance check is enough because every sorted target starts
    # with the same confidence/decision keys. It proves the scored prefix does
    # not contain the manifest's ground-truth label.
    from PIL import Image

    first_path = resolve_project_path(selected[0]["image_path"])
    with Image.open(first_path) as source:
        first_image = source.convert("RGB")
    try:
        prefix_validation = validate_prefix_does_not_encode_truth(
            selected[0],
            image=first_image,
            processor=processor,
            torch=torch,
            include_empty_thought_channel=training_contract[
                "empty_thought_channel"
            ],
            task=args.task,
            class_token_ids=class_token_ids,
        )
    finally:
        first_image.close()

    wall_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        try:
            result = run_one(
                row,
                model=model,
                processor=processor,
                torch=torch,
                include_empty_thought_channel=training_contract[
                    "empty_thought_channel"
                ],
                decision_threshold=args.decision_threshold,
                task=args.task,
                class_token_ids=class_token_ids,
            )
        except Exception as exc:
            raise RuntimeError(
                f"decision-logit evaluation failed on {row['image_id']}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        results.append(result)
        if args.progress_every and (
            index % args.progress_every == 0 or index == len(selected)
        ):
            print(
                f"[{index}/{len(selected)}] {result['image_id']}: "
                f"{result['truth']} -> {result['prediction']} "
                f"(P({result['positive_label']})="
                f"{result['positive_probability']:.4f}, "
                f"{result['latency_ms']:.0f} ms)",
                file=sys.stderr,
                flush=True,
            )

    summary = summarize(
        results, time.perf_counter() - wall_started, task=args.task
    )
    report = {
        "run": {
            **setup,
            "token_contract": token_contract,
            "prefix_validation": prefix_validation,
            "completed_at": utc_now(),
        },
        "summary": summary,
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output is not None:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
