#!/usr/bin/env python3
"""Gemma 4 multimodal QLoRA for RetinaReady quality or review-priority labels.

The training path intentionally follows Google's official Transformers/TRL
vision QLoRA guide. It requires Linux plus an NVIDIA CUDA GPU with BF16.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import random
import socket
import sys
from typing import Any

QUALITY_DISCLAIMER = "Technical image-quality assessment only; not a diagnosis."
ESCALATION_DISCLAIMER = (
    "Review-priority support only; not a diagnosis or treatment recommendation."
)
EMPTY_THOUGHT_CHANNEL = "<|channel>thought\n<channel|>"
OFFICIAL_MODEL = "google/gemma-4-26B-A4B-it"
OFFICIAL_MODEL_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
OFFICIAL_INSTRUCTION_PROCESSOR = "google/gemma-4-E2B-it"
OFFICIAL_PROCESSOR_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
PROVENANCE_FILENAME = "run_provenance.json"
QUALITY_SYSTEM_PROMPT = """You are RetinaReady, an offline technical image-quality assistant.
Assess only capture quality of color fundus photographs. Never diagnose disease,
infer that an eye is healthy, recommend treatment, or interpret retinal pathology.
Return exactly one JSON object. READY means technically sufficient. RETAKE means
technically insufficient. LIMITED is reserved for unsupported or uncertain inputs."""
QUALITY_USER_PROMPT = """Assess this color fundus photograph for capture quality.
Return the RetinaReady JSON object only, with no markdown or additional prose."""

ESCALATION_SYSTEM_PROMPT = """You are RetinaPriority, an offline retinal review-priority assistant.
Assess only whether a clinically usable conventional color fundus photograph should
enter ROUTINE or PRIORITY under the declared diabetic-retinopathy
screening threshold. Never claim a diagnosis, infer that an eye is healthy, recommend
treatment, or delay human review. PRIORITY corresponds to the released dataset's
referable threshold (DR grade 2-4); ROUTINE corresponds to grade 0-1. Return
exactly one JSON object. Unsupported or uncertain inputs must be handled by the
application's fail-closed safety policy rather than released as routine."""
ESCALATION_USER_PROMPT = """Assign review priority for this clinically usable conventional
color fundus photograph. Return the RetinaPriority JSON object only, with no markdown
or additional prose."""

QUALITY_REQUIRED_COLUMNS = {
    "split",
    "patient_id",
    "image_id",
    "image_path",
    "overall_quality",
    "quality_label",
    "artifact",
    "clarity",
    "field_definition",
    "source_split",
}

ESCALATION_REQUIRED_COLUMNS = {
    "split",
    "patient_id",
    "image_id",
    "image_path",
    "dr_grade",
    "escalation_label",
    "overall_quality",
    "source_split",
}


def task_contract(task: str) -> dict[str, Any]:
    if task == "quality":
        return {
            "label_field": "quality_label",
            "labels": ("READY", "RETAKE"),
            "required_columns": QUALITY_REQUIRED_COLUMNS,
            "system_prompt": QUALITY_SYSTEM_PROMPT,
            "user_prompt": QUALITY_USER_PROMPT,
            "disclaimer": QUALITY_DISCLAIMER,
        }
    if task == "escalation":
        return {
            "label_field": "escalation_label",
            # Keep the learned class tokens identical to the immutable source
            # manifests. The product adapter maps these internal research labels
            # to ROUTINE_REVIEW / PRIORITY_REVIEW at the UI boundary.
            "labels": ("ROUTINE", "PRIORITY"),
            "required_columns": ESCALATION_REQUIRED_COLUMNS,
            "system_prompt": ESCALATION_SYSTEM_PROMPT,
            "user_prompt": ESCALATION_USER_PROMPT,
            "disclaimer": ESCALATION_DISCLAIMER,
        }
    raise ValueError(f"unsupported task: {task!r}")


def validate_task_label_tokens(tokenizer: Any, task: str) -> dict[str, Any]:
    """Require distinct first tokens for the balanced decision-token objective."""

    labels = task_contract(task)["labels"]
    encoded = {
        label: tokenizer.encode(label, add_special_tokens=False) for label in labels
    }
    if any(not token_ids for token_ids in encoded.values()):
        raise ValueError(f"one or more {task} labels encoded to zero tokens: {encoded}")
    first_ids = {label: token_ids[0] for label, token_ids in encoded.items()}
    if len(set(first_ids.values())) != len(first_ids):
        raise ValueError(
            f"{task} labels do not have distinct first tokens: {first_ids}"
        )
    return {
        label: {
            "first_token_id": first_ids[label],
            "full_encoding": encoded[label],
        }
        for label in labels
    }


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_project(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root() / path


def load_config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("config must contain one JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--task",
        choices=("quality", "escalation"),
        default="quality",
        help="supervised contract to train; quality remains the backward-compatible default",
    )
    parser.add_argument("--model-id", default=OFFICIAL_MODEL)
    parser.add_argument("--model-revision", default=OFFICIAL_MODEL_REVISION)
    parser.add_argument("--processor-id", default=OFFICIAL_INSTRUCTION_PROCESSOR)
    parser.add_argument("--processor-revision", default=OFFICIAL_PROCESSOR_REVISION)
    parser.add_argument("--train-manifest", default="data/manifests/train.csv")
    parser.add_argument("--val-manifest", default="data/manifests/val.csv")
    parser.add_argument(
        "--calibration-manifest",
        help=(
            "escalation-only threshold-calibration manifest recorded in run "
            "provenance; it is never consumed as training data"
        ),
    )
    parser.add_argument(
        "--eval-manifest",
        help=(
            "escalation-only frozen evaluation manifest recorded in run "
            "provenance; it is never consumed as training data"
        ),
    )
    parser.add_argument("--output-dir", default="ml/runs/gemma4-26b-retina-qlora")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help=(
            "explicit Trainer checkpoint directory to resume; the script never "
            "auto-discovers or implicitly resumes a checkpoint"
        ),
    )
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-regex",
        help=(
            "optional PEFT regular expression for trainable modules; when omitted, "
            "PEFT's pinned Gemma default is used"
        ),
    )
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "linear", "cosine"),
        default="constant",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="evaluation events without improvement before stopping; zero disables",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--empty-thought-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "prepend Gemma's empty thought channel to assistant targets; by default "
            "this is enabled only for 26B/31B instruction checkpoints"
        ),
    )
    parser.add_argument(
        "--loss-scope",
        choices=("assistant", "decision_token"),
        default="assistant",
        help=(
            "assistant supervises the complete JSON target; decision_token "
            "supervises only the first distinguishing class-label token"
        ),
    )
    parser.add_argument(
        "--stratified-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="balance bounded train/eval subsets across the selected task's labels",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-hardware", action="store_true")
    return parser


def resolve_config_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return resolve_from_project(path)


def validated_config_defaults(
    parser: argparse.ArgumentParser, defaults: dict[str, Any]
) -> dict[str, Any]:
    """Reject config typos and JSON values that argparse would not type-check."""

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(defaults) - set(actions))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")

    validated: dict[str, Any] = {}
    for key, value in defaults.items():
        action = actions[key]
        if isinstance(value, bool):
            if not isinstance(
                action,
                (
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                    argparse.BooleanOptionalAction,
                ),
            ):
                raise ValueError(f"config key {key!r} must not be boolean")
            validated[key] = value
        elif isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse.BooleanOptionalAction,
            ),
        ):
            if value is not None or not isinstance(action, argparse.BooleanOptionalAction):
                raise ValueError(f"config key {key!r} must be boolean")
            validated[key] = None
        elif action.type is int:
            if not isinstance(value, int):
                raise ValueError(f"config key {key!r} must be an integer")
            validated[key] = value
        elif action.type is float:
            if not isinstance(value, (int, float)):
                raise ValueError(f"config key {key!r} must be numeric")
            validated[key] = float(value)
        elif action.type in {None, str}:
            if not isinstance(value, str):
                raise ValueError(f"config key {key!r} must be a string")
            validated[key] = value
        else:
            validated[key] = action.type(value)
    return validated


def validate_args(args: argparse.Namespace) -> None:
    task_contract(args.task)
    for key in (
        "model_id",
        "model_revision",
        "processor_id",
        "processor_revision",
    ):
        if not getattr(args, key):
            raise ValueError(f"{key} must not be empty")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or > 0")
    for key in (
        "max_train_samples",
        "max_eval_samples",
        "batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "lora_rank",
        "lora_alpha",
        "logging_steps",
        "eval_steps",
        "save_steps",
    ):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if args.lora_target_regex is not None and not args.lora_target_regex.strip():
        raise ValueError("lora_target_regex must not be blank")
    if args.loss_scope not in {"assistant", "decision_token"}:
        raise ValueError("loss_scope must be 'assistant' or 'decision_token'")
    declared_evaluation_manifests = (
        args.calibration_manifest,
        args.eval_manifest,
    )
    if args.task == "quality" and any(
        value is not None for value in declared_evaluation_manifests
    ):
        raise ValueError(
            "calibration_manifest and eval_manifest are escalation-only"
        )
    if args.task == "escalation":
        for key in ("calibration_manifest", "eval_manifest"):
            value = getattr(args, key)
            if value is not None and not value.strip():
                raise ValueError(f"{key} must not be blank")


def validate_resume_checkpoint(value: Path | None) -> Path | None:
    """Resolve and validate only a checkpoint explicitly supplied by the user."""

    if value is None:
        return None
    checkpoint = resolve_from_project(value).resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"resume checkpoint directory not found: {checkpoint}")
    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise ValueError(
            "resume checkpoint is missing trainer_state.json: "
            f"{trainer_state_path}"
        )
    try:
        with trainer_state_path.open(encoding="utf-8") as handle:
            trainer_state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"resume checkpoint has invalid trainer_state.json: {trainer_state_path}"
        ) from exc
    if not isinstance(trainer_state, dict):
        raise ValueError("resume checkpoint trainer_state.json must be an object")
    global_step = trainer_state.get("global_step")
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError(
            "resume checkpoint trainer_state.json must contain a non-negative "
            "integer global_step"
        )
    return checkpoint


def validate_output_target(
    output_dir_value: str | Path,
    resume_checkpoint: Path | None,
) -> None:
    """Prevent an unapproved run from overwriting an earlier run directory."""

    output_dir = resolve_from_project(output_dir_value).resolve()
    if resume_checkpoint is not None:
        if resume_checkpoint.parent.resolve() != output_dir:
            raise ValueError(
                "resume checkpoint must be directly inside output_dir; got "
                f"{resume_checkpoint} for {output_dir}"
            )
        return
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError(
            f"output_dir is not empty: {output_dir}. Choose a new output directory "
            "or explicitly resume a validated checkpoint."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir exists but is not a directory: {output_dir}")


def use_empty_thought_channel(args: argparse.Namespace) -> bool:
    if args.empty_thought_channel is not None:
        return args.empty_thought_channel
    model_name = args.model_id.lower()
    return model_name.endswith("-it") and ("26b" in model_name or "31b" in model_name)


def read_rows(
    manifest: Path, expected_split: str, task: str = "quality"
) -> list[dict[str, str]]:
    contract = task_contract(task)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = contract["required_columns"] - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{manifest} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{manifest} is empty")
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(
                f"{manifest}: expected split={expected_split!r}, got {row['split']!r}"
            )
        image = resolve_from_project(row["image_path"])
        if not image.is_file():
            raise FileNotFoundError(f"image listed in manifest is missing: {image}")
        label_field = contract["label_field"]
        if row[label_field] not in set(contract["labels"]):
            raise ValueError(f"unexpected {label_field}: {row[label_field]!r}")
        if task == "escalation":
            grade = as_int(row, "dr_grade")
            if grade not in {0, 1, 2, 3, 4}:
                raise ValueError(f"unexpected dr_grade: {grade}")
            expected_label = "PRIORITY" if grade >= 2 else "ROUTINE"
            if row[label_field] != expected_label:
                raise ValueError(
                    f"{row['image_id']}: dr_grade={grade} conflicts with "
                    f"{label_field}={row[label_field]!r}"
                )
    return rows


def select_rows_for_run(
    rows: list[dict[str, str]],
    limit: int | None,
    *,
    seed: int,
    stratified: bool,
    task: str = "quality",
) -> list[dict[str, str]]:
    """Select a deterministic bounded subset without changing the source list."""

    selected = list(rows)
    if limit is None:
        return selected
    if limit > len(selected):
        raise ValueError(f"requested {limit} rows but only {len(selected)} exist")

    rng = random.Random(seed)
    if not stratified:
        rng.shuffle(selected)
        return selected[:limit]

    contract = task_contract(task)
    label_field = contract["label_field"]
    labels = contract["labels"]
    groups = {
        label: [row for row in selected if row[label_field] == label]
        for label in labels
    }
    quota, remainder = divmod(limit, len(labels))
    quotas = {
        label: quota + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }
    for label in labels:
        if len(groups[label]) < quotas[label]:
            raise ValueError(
                f"cannot select {quotas[label]} {label} rows from {len(groups[label])}"
            )
        rng.shuffle(groups[label])
    selected = [
        row
        for label in labels
        for row in groups[label][: quotas[label]]
    ]
    rng.shuffle(selected)
    return selected


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except ValueError as exc:
        raise ValueError(f"{row.get('image_id')}: invalid {key}={row[key]!r}") from exc


def target_for(row: dict[str, str], task: str = "quality") -> dict[str, Any]:
    """Convert official factor scores to a documented presentation contract.

    DeepDRiD has no LIMITED label and no official factor cutoff. The binary
    decision remains the official overall label. Issue codes below are a
    deliberately simple UI heuristic, not new ground truth.
    """

    contract = task_contract(task)
    if task == "escalation":
        decision = row["escalation_label"]
        next_step = (
            "Route for priority clinician review."
            if decision == "PRIORITY"
            else "Keep in the routine clinician review queue."
        )
        return {
            "decision": decision,
            "confidence": None,
            "next_step": next_step,
            "disclaimer": contract["disclaimer"],
        }

    decision = row["quality_label"]
    artifact_raw = as_int(row, "artifact")
    clarity_raw = as_int(row, "clarity")
    field_raw = as_int(row, "field_definition")

    issues: list[str] = []
    if decision == "RETAKE":
        if artifact_raw >= 4:
            issues.append("artifact")
        if clarity_raw <= 6:
            issues.append("blur")
        if field_raw <= 6:
            issues.append("field_cutoff")
        if not issues:
            issues.append("uncertain")

    instructions: list[str] = []
    if "artifact" in issues:
        instructions.append("reduce glare or obstruction and clean the imaging lens")
    if "blur" in issues:
        instructions.append("stabilize the patient and refocus")
    if "field_cutoff" in issues:
        instructions.append("recenter the optic disc and macula")
    if "uncertain" in issues:
        instructions.append("review the capture and retake if technical quality is uncertain")

    if decision == "READY":
        instruction = None
    else:
        instruction = "Retake the image: " + "; ".join(instructions) + "."

    return {
        "decision": decision,
        "confidence": None,
        "issues": issues,
        "scores": {
            # Present all factors as 100=best while preserving their exact order.
            "artifact": 100 - 10 * artifact_raw,
            "clarity": 10 * clarity_raw,
            "field_definition": 10 * field_raw,
        },
        "retake_instruction": instruction,
        "disclaimer": contract["disclaimer"],
    }


def messages_for(
    row: dict[str, str],
    include_answer: bool = True,
    include_empty_thought_channel: bool = False,
    task: str = "quality",
) -> list[dict[str, Any]]:
    contract = task_contract(task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": contract["system_prompt"]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": contract["user_prompt"]},
                {"type": "image", "image": str(resolve_from_project(row["image_path"]))},
            ],
        },
    ]
    if include_answer:
        answer = json.dumps(
            target_for(row, task), separators=(",", ":"), sort_keys=True
        )
        if include_empty_thought_channel:
            # Google recommends this for 26B/31B instruction checkpoints when
            # the fine-tuning data deliberately contains no reasoning trace.
            answer = EMPTY_THOUGHT_CHANNEL + answer
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


def render_training_text(
    processor: Any,
    row: dict[str, str],
    include_empty_thought_channel: bool,
    task: str = "quality",
) -> str:
    """Render a prompt and retain Google's recommended empty thought channel.

    Some canonical chat-template revisions strip thought-channel text from an
    assistant message. Inject it after rendering when that happens.
    """

    messages = messages_for(
        row,
        include_empty_thought_channel=include_empty_thought_channel,
        task=task,
    )
    text = processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    if include_empty_thought_channel and EMPTY_THOUGHT_CHANNEL not in text:
        answer = json.dumps(
            target_for(row, task), separators=(",", ":"), sort_keys=True
        )
        position = text.rfind(answer)
        if position < 0:
            raise ValueError("chat template did not preserve the supervised JSON target")
        text = text[:position] + EMPTY_THOUGHT_CHANNEL + text[position:]
    return text.strip()


def render_generation_prompt(
    processor: Any, row: dict[str, str], task: str = "quality"
) -> str:
    """Render the exact prefix that should be excluded from supervised loss."""

    return processor.apply_chat_template(
        messages_for(row, include_answer=False, task=task),
        add_generation_prompt=True,
        tokenize=False,
    ).strip()


def assistant_only_labels(
    batch: dict[str, Any],
    prompt_batch: dict[str, Any],
    torch: Any,
) -> Any:
    """Build labels that supervise only content after the generation prompt."""

    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch_size = batch["input_ids"].shape[0]
    if prompt_batch["input_ids"].shape[0] != batch_size:
        raise ValueError("prompt/full batch sizes differ")

    for index in range(batch_size):
        prompt_length = int(prompt_batch["attention_mask"][index].sum().item())
        full_length = int(batch["attention_mask"][index].sum().item())
        if prompt_length >= full_length:
            raise ValueError(
                "generation prompt consumes the full supervised sequence; "
                "the assistant target is missing"
            )
        prompt_ids = prompt_batch["input_ids"][index, :prompt_length]
        full_prefix = batch["input_ids"][index, :prompt_length]
        if not torch.equal(prompt_ids, full_prefix):
            raise ValueError(
                "generation prompt tokens are not a prefix of the supervised "
                "sequence; refusing to train with an ambiguous loss mask"
            )
        # System/user text and the assistant-generation header remain context,
        # not prediction labels.
        labels[index, :prompt_length] = -100
    return labels


def decision_token_only_labels(
    batch: dict[str, Any],
    assistant_labels: Any,
    rows: list[dict[str, str]],
    tokenizer: Any,
    torch: Any,
    task: str = "quality",
) -> Any:
    """Supervise one balanced class token while retaining the full JSON context."""

    if len(rows) != batch["input_ids"].shape[0]:
        raise ValueError("row/token batch sizes differ")
    labels = torch.full_like(assistant_labels, -100)
    input_ids = batch["input_ids"]
    sequence_length = input_ids.shape[1]
    label_field = task_contract(task)["label_field"]
    for index, row in enumerate(rows):
        decision_ids = tokenizer.encode(
            row[label_field], add_special_tokens=False
        )
        if not decision_ids:
            raise ValueError("decision label encoded to zero tokens")
        width = len(decision_ids)
        matches: list[int] = []
        for start in range(sequence_length - width + 1):
            stop = start + width
            if torch.any(assistant_labels[index, start:stop] == -100):
                continue
            if input_ids[index, start:stop].tolist() == decision_ids:
                matches.append(start)
        if len(matches) != 1:
            raise ValueError(
                f"expected one supervised {row[label_field]} token span, "
                f"found {len(matches)}"
            )
        # Class labels may tokenize to different widths. Training only the first
        # distinguishing token gives every image equal class-loss weight.
        position = matches[0]
        labels[index, position] = input_ids[index, position]
    return labels


def show_dry_run(
    args: argparse.Namespace,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
) -> None:
    contract = task_contract(args.task)
    label_field = contract["label_field"]
    task_labels = contract["labels"]
    train_patients = {row["patient_id"] for row in train_rows}
    val_patients = {row["patient_id"] for row in val_rows}
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f"patient leakage between train and val: {sorted(overlap)[:5]}")

    counts: dict[str, int] = {}
    for row in train_rows:
        counts[row[label_field]] = counts.get(row[label_field], 0) + 1
    selected_train = select_rows_for_run(
        train_rows,
        args.max_train_samples,
        seed=args.seed,
        stratified=args.stratified_sampling,
        task=args.task,
    )
    selected_val = select_rows_for_run(
        val_rows,
        args.max_eval_samples,
        seed=args.seed + 1,
        stratified=args.stratified_sampling,
        task=args.task,
    )
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "task": args.task,
                "model_revision": args.model_revision,
                "processor_id": args.processor_id,
                "processor_revision": args.processor_revision,
                "resume_from_checkpoint": (
                    str(args.resume_from_checkpoint)
                    if args.resume_from_checkpoint is not None
                    else None
                ),
                "output_dir": str(resolve_from_project(args.output_dir)),
                "calibration_manifest": (
                    str(resolve_from_project(args.calibration_manifest))
                    if args.calibration_manifest is not None
                    else None
                ),
                "eval_manifest": (
                    str(resolve_from_project(args.eval_manifest))
                    if args.eval_manifest is not None
                    else None
                ),
                "strict_hardware": args.strict_hardware,
                "empty_thought_channel": use_empty_thought_channel(args),
                "training_plan": {
                    "epochs": args.epochs,
                    "max_steps": args.max_steps,
                    "max_train_samples": args.max_train_samples,
                    "max_eval_samples": args.max_eval_samples,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": (
                        args.gradient_accumulation_steps
                    ),
                    "max_seq_length": args.max_seq_length,
                    "learning_rate": args.learning_rate,
                    "lr_scheduler_type": args.lr_scheduler_type,
                    "warmup_ratio": args.warmup_ratio,
                    "lora_target_regex": args.lora_target_regex,
                    "early_stopping_patience": args.early_stopping_patience,
                    "loss_scope": args.loss_scope,
                    "stratified_sampling": args.stratified_sampling,
                },
                "train_images": len(train_rows),
                "train_patients": len(train_patients),
                "train_labels": counts,
                "val_images": len(val_rows),
                "val_patients": len(val_patients),
                "patient_overlap": 0,
                "selected_preview": {
                    "train_rows": len(selected_train),
                    "train_labels": {
                        label: sum(
                            row[label_field] == label
                            for row in selected_train
                        )
                        for label in task_labels
                    },
                    "validation_rows": len(selected_val),
                    "validation_labels": {
                        label: sum(
                            row[label_field] == label for row in selected_val
                        )
                        for label in task_labels
                    },
                },
                "example_messages": messages_for(
                    train_rows[0],
                    include_empty_thought_channel=use_empty_thought_channel(args),
                    task=args.task,
                ),
            },
            indent=2,
        )
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def package_versions() -> dict[str, str | None]:
    names = (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "huggingface-hub",
        "peft",
        "Pillow",
        "safetensors",
        "tensorboard",
        "torch",
        "transformers",
        "trl",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def manifest_metadata(
    path: Path, rows: list[dict[str, str]], task: str = "quality"
) -> dict[str, Any]:
    label_field = task_contract(task)["label_field"]
    labels: dict[str, int] = {}
    for row in rows:
        label = row[label_field]
        labels[label] = labels.get(label, 0) + 1
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "labels": labels,
    }


def selected_rows_sha256(
    rows: list[dict[str, str]], task: str = "quality"
) -> str:
    label_field = task_contract(task)["label_field"]
    selected = [
        {
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            label_field: row[label_field],
        }
        for row in rows
    ]
    payload = json.dumps(
        selected, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def gpu_metadata(torch: Any) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        device: dict[str, Any] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": properties.total_memory / 2**30,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "multiprocessor_count": properties.multi_processor_count,
        }
        uuid = getattr(properties, "uuid", None)
        if uuid is not None:
            device["uuid"] = str(uuid)
        devices.append(device)
    return {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "devices": devices,
    }


def checkpoint_metadata(checkpoint: Path | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    trainer_state = checkpoint / "trainer_state.json"
    with trainer_state.open(encoding="utf-8") as handle:
        state = json.load(handle)
    return {
        "path": str(checkpoint),
        "trainer_state_sha256": sha256_file(trainer_state),
        "global_step": state["global_step"],
    }


def build_run_provenance(
    args: argparse.Namespace,
    train_manifest: Path,
    val_manifest: Path,
    full_train_rows: list[dict[str, str]],
    full_val_rows: list[dict[str, str]],
    selected_train_rows: list[dict[str, str]],
    selected_val_rows: list[dict[str, str]],
    torch: Any,
) -> dict[str, Any]:
    contract = task_contract(args.task)
    label_field = contract["label_field"]
    task_labels = contract["labels"]
    script_path = Path(__file__).resolve()
    config_source: dict[str, Any] | None = None
    if args.config is not None:
        with args.config.open(encoding="utf-8") as handle:
            config_values = json.load(handle)
        config_source = {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
            "values": config_values,
        }

    return {
        "schema_version": 1,
        "status": "initializing",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "run": {
            "argv": sys.argv,
            "working_directory": str(Path.cwd()),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "code": {
            "train_script": str(script_path),
            "train_script_sha256": sha256_file(script_path),
        },
        "checkpoint_sources": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "processor_id": args.processor_id,
            "processor_revision": args.processor_revision,
        },
        "effective_config": json_safe(vars(args)),
        "config_source": config_source,
        "artifacts": {
            "output_dir": str(resolve_from_project(args.output_dir).resolve()),
            "provenance_file": str(
                (
                    resolve_from_project(args.output_dir) / PROVENANCE_FILENAME
                ).resolve()
            ),
        },
        "resume_checkpoint": checkpoint_metadata(args.resume_from_checkpoint),
        "manifests": {
            "train": manifest_metadata(train_manifest, full_train_rows, args.task),
            "validation": manifest_metadata(val_manifest, full_val_rows, args.task),
        },
        "selected_data": {
            "train_rows": len(selected_train_rows),
            "train_rows_sha256": selected_rows_sha256(
                selected_train_rows, args.task
            ),
            "train_labels": {
                label: sum(
                    row[label_field] == label for row in selected_train_rows
                )
                for label in task_labels
            },
            "validation_rows": len(selected_val_rows),
            "validation_rows_sha256": selected_rows_sha256(
                selected_val_rows, args.task
            ),
            "validation_labels": {
                label: sum(
                    row[label_field] == label for row in selected_val_rows
                )
                for label in task_labels
            },
        },
        "packages": package_versions(),
        "gpu": gpu_metadata(torch),
        "cuda_peak_memory": None,
        "trainer_metrics": None,
        "failure": None,
    }


def save_run_provenance(path: Path, provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def peak_cuda_memory(torch: Any) -> dict[str, Any]:
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {
        "max_memory_allocated_bytes": allocated,
        "max_memory_allocated_gib": allocated / 2**30,
        "max_memory_reserved_bytes": reserved,
        "max_memory_reserved_gib": reserved / 2**30,
    }


def freeze_kbit_base_without_dtype_upcast(model: Any) -> dict[str, Any]:
    """Freeze a 4-bit base while preserving frozen Gemma 4 MoE dtypes.

    PEFT's generic ``prepare_model_for_kbit_training`` promotes every BF16 or
    FP16 parameter that is not a bitsandbytes ``Params4bit`` object to FP32.
    Gemma 4 A4B stores its packed expert matrices as ordinary 3-D parameters,
    so that generic promotion would double tens of GiB of frozen weights and
    cannot fit on an 80 GiB A100. TRL installs the LoRA adapters, enables input
    gradients, and applies the non-reentrant gradient-checkpointing settings
    from ``SFTConfig`` later; only the explicit base freeze is needed here.
    """

    if not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("QLoRA preparation requires a model loaded in 4-bit mode")
    if not getattr(model, "supports_gradient_checkpointing", False):
        raise ValueError("model does not support gradient checkpointing")

    frozen_parameters = 0
    params4bit_storage_bytes = 0
    unpacked_expert_storage_bytes = 0
    unpacked_expert_parameters = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        frozen_parameters += parameter.numel()
        storage_bytes = parameter.numel() * parameter.element_size()
        if parameter.__class__.__name__ == "Params4bit":
            params4bit_storage_bytes += storage_bytes
        elif ".experts." in name and parameter.ndim == 3:
            unpacked_expert_storage_bytes += storage_bytes
            unpacked_expert_parameters += parameter.numel()

    return {
        "strategy": "freeze_base_preserve_gemma4_moe_dtypes",
        "frozen_parameters": frozen_parameters,
        "params4bit_storage_bytes": params4bit_storage_bytes,
        "params4bit_storage_gib": params4bit_storage_bytes / 2**30,
        "unpacked_expert_parameters": unpacked_expert_parameters,
        "unpacked_expert_storage_bytes": unpacked_expert_storage_bytes,
        "unpacked_expert_storage_gib": unpacked_expert_storage_bytes / 2**30,
    }


def train(
    args: argparse.Namespace,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
) -> None:
    try:
        import torch
        from PIL import Image
        from peft import LoraConfig
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install ml/requirements-train.txt "
            "on a Linux NVIDIA host."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "26B QLoRA is intentionally blocked on this Mac: the official path requires "
            "CUDA + BF16. Use an NVIDIA cloud GPU, or the E2B fallback config."
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The CUDA GPU must support bfloat16 (for example L4, A100, or H100).")

    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total_gib < 48:
        warning = (
            f"GPU has {total_gib:.1f} GiB. Gemma 4 26B multimodal QLoRA is likely "
            "to OOM below 48 GiB; 80 GiB is preferred."
        )
        if args.strict_hardware and "26b" in args.model_id.lower():
            raise SystemExit(warning)
        print(f"WARNING: {warning}", file=sys.stderr)

    full_train_rows = list(train_rows)
    full_val_rows = list(val_rows)
    train_rows = select_rows_for_run(
        full_train_rows,
        args.max_train_samples,
        seed=args.seed,
        stratified=args.stratified_sampling,
        task=args.task,
    )
    val_rows = select_rows_for_run(
        full_val_rows,
        args.max_eval_samples,
        seed=args.seed + 1,
        stratified=args.stratified_sampling,
        task=args.task,
    )

    train_manifest = resolve_from_project(args.train_manifest)
    val_manifest = resolve_from_project(args.val_manifest)
    output_dir = resolve_from_project(args.output_dir)
    provenance_path = output_dir / PROVENANCE_FILENAME
    provenance = build_run_provenance(
        args,
        train_manifest,
        val_manifest,
        full_train_rows,
        full_val_rows,
        train_rows,
        val_rows,
        torch,
    )
    torch.cuda.reset_peak_memory_stats()
    save_run_provenance(provenance_path, provenance)
    print("RUN_PROVENANCE_BEGIN")
    print(json.dumps(provenance, indent=2, sort_keys=True))

    try:
        dtype = torch.bfloat16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            dtype=dtype,
            device_map="auto",
            quantization_config=quantization,
        )
        processor = AutoProcessor.from_pretrained(
            args.processor_id,
            revision=args.processor_revision,
        )
        processor.tokenizer.padding_side = "right"

        model_vocab_size = getattr(model.config.text_config, "vocab_size", None)
        processor_vocab_size = len(processor.tokenizer)
        if model_vocab_size != processor_vocab_size:
            raise ValueError(
                "processor/model vocabulary mismatch: "
                f"{args.processor_id}@{args.processor_revision} has "
                f"{processor_vocab_size} tokens, while "
                f"{args.model_id}@{args.model_revision} expects {model_vocab_size}"
            )
        for token_name in ("boi_token_id", "image_token_id", "eoi_token_id"):
            model_token_id = getattr(model.config, token_name, None)
            processor_token_id = getattr(processor.tokenizer, token_name, None)
            if model_token_id != processor_token_id:
                raise ValueError(
                    f"processor/model {token_name} mismatch: "
                    f"{processor_token_id} != {model_token_id}"
                )

        provenance["label_token_contract"] = validate_task_label_tokens(
            processor.tokenizer, args.task
        )
        save_run_provenance(provenance_path, provenance)

        model.config.use_cache = False
        provenance["model_preparation"] = freeze_kbit_base_without_dtype_upcast(
            model
        )
        save_run_provenance(provenance_path, provenance)
        print(json.dumps({"model_preparation": provenance["model_preparation"]}))

        peft_kwargs: dict[str, Any] = {}
        if args.lora_target_regex is not None:
            peft_kwargs["target_modules"] = args.lora_target_regex
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_rank,
            bias="none",
            # Official PEFT defaults scope Gemma 4 LoRA to language-model layers.
            task_type="CAUSAL_LM",
            # The instruction checkpoint and processor vocabularies already match,
            # including every multimodal/chat token, so no embedding or LM-head
            # copies are trained. An explicit target regex may broaden the
            # language-attention projections while keeping MoE experts frozen.
            **peft_kwargs,
        )

        interval_strategy = "epoch" if args.max_steps < 0 else "steps"

        sft_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            logging_steps=args.logging_steps,
            save_strategy=interval_strategy,
            save_steps=args.save_steps,
            eval_strategy=interval_strategy,
            eval_steps=args.eval_steps,
            learning_rate=args.learning_rate,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            report_to="tensorboard",
            seed=args.seed,
            data_seed=args.seed,
            max_length=args.max_seq_length,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            push_to_hub=False,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        def collate_fn(examples: list[dict[str, str]]) -> dict[str, Any]:
            texts: list[str] = []
            prompt_texts: list[str] = []
            images: list[list[Image.Image]] = []
            include_empty_channel = use_empty_thought_channel(args)
            for row in examples:
                with Image.open(resolve_from_project(row["image_path"])) as image:
                    rgb = image.convert("RGB")
                texts.append(
                    render_training_text(
                        processor, row, include_empty_channel, task=args.task
                    )
                )
                prompt_texts.append(
                    render_generation_prompt(processor, row, task=args.task)
                )
                images.append([rgb])

            batch = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            prompt_batch = processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            sequence_length = batch["input_ids"].shape[1]
            if sequence_length > args.max_seq_length:
                raise ValueError(
                    f"encoded batch is {sequence_length} tokens, above "
                    f"max_seq_length={args.max_seq_length}; increase the limit "
                    "instead of truncating the supervised answer"
                )
            labels = assistant_only_labels(batch, prompt_batch, torch)
            if args.loss_scope == "decision_token":
                labels = decision_token_only_labels(
                    batch,
                    labels,
                    examples,
                    processor.tokenizer,
                    torch,
                    task=args.task,
                )
            token_ids = [
                processor.tokenizer.pad_token_id,
                getattr(processor.tokenizer, "boi_token_id", None),
                getattr(processor.tokenizer, "image_token_id", None),
                getattr(processor.tokenizer, "eoi_token_id", None),
            ]
            for token_id in token_ids:
                if token_id is not None:
                    labels[labels == token_id] = -100
            batch["labels"] = labels
            return batch

        callbacks = []
        if args.early_stopping_patience:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                )
            )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_rows,
            eval_dataset=val_rows,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=collate_fn,
            callbacks=callbacks,
        )
        trainable = [
            (name, parameter.numel())
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        ]
        trainable_experts = [name for name, _ in trainable if ".experts." in name]
        non_lora_trainable = [name for name, _ in trainable if "lora_" not in name]
        if trainable_experts:
            raise ValueError(
                f"frozen MoE expert parameters became trainable: {trainable_experts[:5]}"
            )
        if non_lora_trainable:
            raise ValueError(
                "unexpected non-LoRA trainable parameters: "
                f"{non_lora_trainable[:5]}"
            )
        provenance["trainable_parameters"] = {
            "count": sum(size for _, size in trainable),
            "tensor_count": len(trainable),
            "names_sha256": hashlib.sha256(
                "\n".join(name for name, _ in trainable).encode("utf-8")
            ).hexdigest(),
        }
        save_run_provenance(provenance_path, provenance)
        trainer.model.print_trainable_parameters()
        provenance["status"] = "training"
        save_run_provenance(provenance_path, provenance)
        if args.resume_from_checkpoint is None:
            train_result = trainer.train()
        else:
            train_result = trainer.train(
                resume_from_checkpoint=str(args.resume_from_checkpoint)
            )
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))
        provenance["trainer_metrics"] = json_safe(train_result.metrics)
        provenance["status"] = "completed"
        print(f"Saved PEFT adapter and processor to {output_dir}")
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        try:
            provenance["cuda_peak_memory"] = peak_cuda_memory(torch)
        except Exception as memory_exc:
            provenance["cuda_peak_memory"] = {
                "collection_error": f"{type(memory_exc).__name__}: {memory_exc}"
            }
        provenance["finished_at_utc"] = utc_now()
        save_run_provenance(provenance_path, provenance)
        print("RUN_PROVENANCE_FINAL")
        print(json.dumps(provenance, indent=2, sort_keys=True))
        print(f"Run provenance saved to {provenance_path}")


def main() -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    known, _ = config_parser.parse_known_args()
    parser = build_parser()
    config_path = resolve_config_path(known.config)
    defaults = validated_config_defaults(
        parser, load_config_defaults(config_path)
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    args.config = config_path
    validate_args(args)
    args.resume_from_checkpoint = validate_resume_checkpoint(
        args.resume_from_checkpoint
    )

    train_rows = read_rows(
        resolve_from_project(args.train_manifest), "train", task=args.task
    )
    val_rows = read_rows(
        resolve_from_project(args.val_manifest), "val", task=args.task
    )
    show_dry_run(args, train_rows, val_rows)
    if args.dry_run:
        return 0
    validate_output_target(args.output_dir, args.resume_from_checkpoint)
    train(args, train_rows, val_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
