#!/usr/bin/env python3
"""Evaluate the pinned Gemma 4 HF checkpoint, with or without a PEFT adapter.

Dry-run mode is dependency-light and never imports CUDA libraries, contacts
Hugging Face, or loads model weights. Real evaluation is CUDA-only and uses the
same manifest sampling, fail-closed schema validation, and safety metrics as
``evaluate_local.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

from evaluate_local import read_manifest, select_rows, summarize
from infer_local import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    extract_object,
    limited_fallback,
    normalize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_MODEL_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
DEFAULT_PROCESSOR = "google/gemma-4-E2B-it"
DEFAULT_PROCESSOR_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def adapter_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    adapter = resolve_project_path(path).resolve()
    if not adapter.is_dir():
        raise ValueError(f"adapter directory does not exist: {adapter}")
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter is missing adapter_config.json: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain one JSON object")
    weight_candidates = (
        adapter / "adapter_model.safetensors",
        adapter / "adapter_model.bin",
    )
    weights = next((candidate for candidate in weight_candidates if candidate.is_file()), None)
    if weights is None:
        raise ValueError(
            "adapter directory has no adapter_model.safetensors or adapter_model.bin"
        )
    provenance_path = adapter / "run_provenance.json"
    if not provenance_path.is_file():
        raise ValueError(
            "adapter is missing run_provenance.json; exact source revisions and "
            f"successful completion cannot be verified: {provenance_path}"
        )
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not isinstance(provenance, dict):
        raise ValueError("run_provenance.json must contain one JSON object")
    if provenance.get("status") != "completed":
        raise ValueError(
            "adapter provenance does not record a completed run: "
            f"{provenance.get('status')!r}"
        )
    checkpoint_sources = provenance.get("checkpoint_sources")
    if not isinstance(checkpoint_sources, dict):
        raise ValueError("adapter provenance is missing checkpoint_sources")
    return {
        "path": str(adapter),
        "config_path": str(config_path),
        "weights_path": str(weights),
        "provenance_path": str(provenance_path),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "checkpoint_sources": checkpoint_sources,
    }


def validate_adapter_base(
    metadata: dict[str, Any] | None,
    expected_model_id: str,
    expected_model_revision: str,
    expected_processor_id: str,
    expected_processor_revision: str,
) -> None:
    if metadata is None:
        return
    recorded = metadata.get("base_model_name_or_path")
    if recorded and recorded != expected_model_id:
        raise ValueError(
            "adapter/base mismatch: adapter records "
            f"{recorded!r}, evaluator expects {expected_model_id!r}"
        )
    expected_sources = {
        "model_id": expected_model_id,
        "model_revision": expected_model_revision,
        "processor_id": expected_processor_id,
        "processor_revision": expected_processor_revision,
    }
    recorded_sources = metadata["checkpoint_sources"]
    mismatches = {
        key: {"adapter": recorded_sources.get(key), "evaluator": value}
        for key, value in expected_sources.items()
        if recorded_sources.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "adapter provenance does not match evaluator sources: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def build_messages(image_path: Path) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image", "image": str(image_path)},
            ],
        },
    ]


def run_one(
    row: dict[str, str],
    *,
    model: Any,
    processor: Any,
    torch: Any,
    max_new_tokens: int,
    include_output: bool,
) -> dict[str, Any]:
    image_path = resolve_project_path(row["image_path"])
    started = time.perf_counter()
    schema_valid = False
    request_succeeded = False
    error: str | None = None
    content: str | None = None
    try:
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        prompt = processor.apply_chat_template(
            build_messages(image_path),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )
        device = model.device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        torch.cuda.synchronize()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        torch.cuda.synchronize()
        input_tokens = inputs["input_ids"].shape[1]
        content = processor.decode(
            generated[0][input_tokens:],
            skip_special_tokens=False,
        )
        # Generation completed successfully even if the model's content later
        # fails the strict schema. Keep transport/runtime success separate from
        # schema validity so this report matches evaluate_local.py.
        request_succeeded = True
        prediction = normalize(extract_object(content))
        schema_valid = True
    except Exception as exc:
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
    if error:
        result["error"] = error
    if include_output:
        result["output"] = {
            key: value for key, value in prediction.items() if not key.startswith("_")
        }
        if content is not None and not schema_valid:
            result["raw_content"] = content
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pinned Gemma 4 HF base or a PEFT adapter on CUDA. "
            "Use identical sampling arguments for a fair base-vs-adapter comparison."
        )
    )
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--processor-id", default=DEFAULT_PROCESSOR)
    parser.add_argument("--processor-revision", default=DEFAULT_PROCESSOR_REVISION)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/val.csv"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sampling",
        choices=("sequential", "random", "stratified"),
        default="sequential",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
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
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if args.progress_every < 0:
        raise ValueError("progress_every cannot be negative")


def load_runtime(args: argparse.Namespace, metadata: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. Run ml/bootstrap_a100.sh first."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("PEFT evaluation requires CUDA; no model was loaded.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("PEFT evaluation requires native CUDA BF16 support.")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization,
    )
    if metadata is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("PEFT is required to load --adapter-dir.") from exc
        model = PeftModel.from_pretrained(
            model,
            metadata["path"],
            is_trainable=False,
        )
    processor = AutoProcessor.from_pretrained(
        args.processor_id,
        revision=args.processor_revision,
    )
    model.eval()
    return model, processor, torch


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        metadata = adapter_metadata(args.adapter_dir)
        validate_adapter_base(
            metadata,
            args.model_id,
            args.model_revision,
            args.processor_id,
            args.processor_revision,
        )
        manifest = resolve_project_path(args.manifest)
        rows = read_manifest(manifest)
        selected = select_rows(
            rows,
            limit=args.limit,
            sampling=args.sampling,
            seed=args.seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    setup = {
        "mode": "peft-adapter" if metadata else "frozen-base",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "processor_id": args.processor_id,
        "processor_revision": args.processor_revision,
        "adapter": metadata,
        "manifest": str(manifest),
        "available_rows": len(rows),
        "selected_rows": len(selected),
        "sampling": args.sampling,
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "setup": setup}, indent=2))
        return 0

    model, processor, torch = load_runtime(args, metadata)
    wall_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        result = run_one(
            row,
            model=model,
            processor=processor,
            torch=torch,
            max_new_tokens=args.max_new_tokens,
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

    summary = summarize(results, time.perf_counter() - wall_started)
    report = {
        "run": {
            **setup,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {output}", file=sys.stderr)
    return 0 if summary["request_success_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
