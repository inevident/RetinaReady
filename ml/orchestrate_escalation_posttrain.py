#!/usr/bin/env python3
"""Fail-closed post-training selection and calibration for RetinaPriority.

The program deliberately has two phases.  First, it scores the completed run
root and every retained epoch checkpoint on the *complete validation manifest*
and freezes a validation-only checkpoint selection.  Only after the immutable
selection artifact has been written does it read or score the calibration and
evaluation partitions.

Existing decision-logit reports are reused only after their manifest, adapter,
training provenance, logits, probabilities, predictions, and aggregate metrics
have all been revalidated.  A partial or stale report is an error; it is never
silently overwritten.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Sequence

from calibrate_escalation_adapter import (
    CalibrationInputError,
    build_report as build_policy_report,
    load_scored_partition,
    verify_integrity as verify_policy_integrity,
)
from calibrate_selective_policy import exact_upper_bound
from evaluate_decision_logits import (
    load_decision_training_contract,
    positive_roc_auc,
)
from evaluate_peft import adapter_metadata, resolve_project_path, validate_adapter_base
from train_qlora import read_rows, select_rows_for_run, selected_rows_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"checkpoint-([0-9]+)")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LORA_TENSOR_RE = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(?P<layer>[0-9]+)\."
    r"self_attn\.(?P<projection>q_proj|k_proj|v_proj|o_proj)\."
    r"lora_(?P<matrix>A|B)\.weight$"
)
PROJECTION_ORDER = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "o_proj": 3}
MATRIX_ORDER = {"A": 0, "B": 1}
REPORT_THRESHOLD = 0.5
ESCALATION_CLASS_TOKEN_IDS = {"ROUTINE": 2073, "PRIORITY": 65324}
DEFAULT_FALSE_ROUTINE_RISK = 0.10
DEFAULT_FALSE_PRIORITY_RISK = 0.10
DEFAULT_DELTA = 0.05
REFERENCE_PRIORITY_CALIBRATION_PATIENTS = 35
REFERENCE_ROUTINE_CALIBRATION_PATIENTS = 40


class PosttrainError(ValueError):
    """Raised when post-training evidence cannot prove the required contract."""


@dataclass(frozen=True)
class Candidate:
    identifier: str
    path: Path
    role: str
    global_step: int | None
    epoch: float | None
    trainer_state_path: Path | None
    adapter_config_path: Path
    weights_path: Path
    provenance_path: Path


@dataclass(frozen=True)
class SafetyCriteria:
    minimum_roc_auc: float = 0.70
    minimum_balanced_accuracy: float = 0.60
    minimum_priority_recall: float = 0.80
    minimum_routine_recall: float = 0.50

    def validate(self) -> None:
        for name, value in (
            ("minimum_roc_auc", self.minimum_roc_auc),
            ("minimum_balanced_accuracy", self.minimum_balanced_accuracy),
            ("minimum_priority_recall", self.minimum_priority_recall),
            ("minimum_routine_recall", self.minimum_routine_recall),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PosttrainError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class RiskProfile:
    false_routine_risk: float = DEFAULT_FALSE_ROUTINE_RISK
    false_priority_risk: float = DEFAULT_FALSE_PRIORITY_RISK
    delta: float = DEFAULT_DELTA

    def validate(self) -> None:
        for name, value in (
            ("false_routine_risk", self.false_routine_risk),
            ("false_priority_risk", self.false_priority_risk),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise PosttrainError(
                    f"{name} must be finite and strictly between zero and one"
                )
        if not math.isfinite(self.delta) or not 0.0 < self.delta < 0.5:
            raise PosttrainError(
                "delta must be finite and strictly between zero and 0.5"
            )

    def provenance(self) -> dict[str, Any]:
        """Describe the non-clinical default and its finite-sample rationale."""

        self.validate()
        priority_zero_error_bound = exact_upper_bound(
            0, REFERENCE_PRIORITY_CALIBRATION_PATIENTS, self.delta
        )
        routine_zero_error_bound = exact_upper_bound(
            0, REFERENCE_ROUTINE_CALIBRATION_PATIENTS, self.delta
        )
        reference_priority_bound = exact_upper_bound(
            0, REFERENCE_PRIORITY_CALIBRATION_PATIENTS, DEFAULT_DELTA
        )
        reference_routine_bound = exact_upper_bound(
            0, REFERENCE_ROUTINE_CALIBRATION_PATIENTS, DEFAULT_DELTA
        )
        return {
            "false_routine_risk": self.false_routine_risk,
            "false_priority_risk": self.false_priority_risk,
            "per_gate_delta": self.delta,
            "profile": (
                "default_research_profile"
                if self
                == RiskProfile(
                    DEFAULT_FALSE_ROUTINE_RISK,
                    DEFAULT_FALSE_PRIORITY_RISK,
                    DEFAULT_DELTA,
                )
                else "explicit_override"
            ),
            "reference_calibration_denominators": {
                "PRIORITY_patients_for_false_ROUTINE_gate": (
                    REFERENCE_PRIORITY_CALIBRATION_PATIENTS
                ),
                "ROUTINE_patients_for_false_PRIORITY_gate": (
                    REFERENCE_ROUTINE_CALIBRATION_PATIENTS
                ),
            },
            "zero_error_one_sided_clopper_pearson_upper_bounds": {
                "delta": self.delta,
                "false_ROUTINE_at_n_35": priority_zero_error_bound,
                "false_PRIORITY_at_n_40": routine_zero_error_bound,
            },
            "five_percent_certifiable_with_zero_errors": {
                "false_ROUTINE_at_n_35": priority_zero_error_bound <= 0.05,
                "false_PRIORITY_at_n_40": routine_zero_error_bound <= 0.05,
            },
            "default_profile_reference": {
                "false_routine_risk": DEFAULT_FALSE_ROUTINE_RISK,
                "false_priority_risk": DEFAULT_FALSE_PRIORITY_RISK,
                "delta": DEFAULT_DELTA,
                "zero_error_upper_bound_at_n_35": reference_priority_bound,
                "zero_error_upper_bound_at_n_40": reference_routine_bound,
            },
            "rationale": (
                "At the default delta=0.05, even zero observed errors among 35 PRIORITY "
                "patients or 40 ROUTINE patients has a one-sided exact upper "
                "bound of about 8.20% or 7.22%, respectively. The 10%/10% "
                "research profile is therefore the tightest simple round-number "
                "default that these denominators can potentially certify; it is "
                "not a clinical safety claim."
            ),
        }


CommandRunner = Callable[[Sequence[str], Path], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PosttrainError(f"{name} must be a regular non-symlink file: {path}")
    return path


def require_regular_directory(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise PosttrainError(
            f"{name} must be a regular non-symlink directory: {path}"
        )
    return path


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PosttrainError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PosttrainError(f"{name} must contain one JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise PosttrainError(f"refusing to replace JSON symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.is_symlink():
        raise PosttrainError(f"refusing to write through temporary symlink: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve(value: str | Path) -> Path:
    return resolve_project_path(value).resolve()


def resolve_existing_directory(value: str | Path, name: str) -> Path:
    """Resolve an existing directory without hiding a symlink at the leaf."""

    unresolved = resolve_project_path(value)
    require_regular_directory(unresolved, name)
    return unresolved.resolve()


def resolve_output_directory(value: str | Path, name: str) -> Path:
    """Resolve a present-or-future output directory without following its leaf."""

    unresolved = resolve_project_path(value)
    if unresolved.is_symlink():
        raise PosttrainError(f"{name} must not be a symlink: {unresolved}")
    if unresolved.exists() and not unresolved.is_dir():
        raise PosttrainError(f"{name} must be a directory: {unresolved}")
    return unresolved.resolve()


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PosttrainError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PosttrainError(f"{name} must be an integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PosttrainError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PosttrainError(f"{name} must be finite")
    return number


def load_full_config(path: Path) -> dict[str, Any]:
    config = load_json(path, "training config")
    required_strings = (
        "model_id",
        "model_revision",
        "processor_id",
        "processor_revision",
        "train_manifest",
        "val_manifest",
        "calibration_manifest",
        "eval_manifest",
        "output_dir",
    )
    for key in required_strings:
        _require_string(config.get(key), f"config.{key}")
    if config.get("task") != "escalation":
        raise PosttrainError("config.task must be 'escalation'")
    if config.get("loss_scope") != "decision_token":
        raise PosttrainError("config.loss_scope must be 'decision_token'")
    if config.get("max_steps") != -1:
        raise PosttrainError("post-training orchestration requires max_steps=-1")
    if config.get("stratified_sampling") is not False:
        raise PosttrainError(
            "full training config must use stratified_sampling=false"
        )
    epochs = _finite(config.get("epochs"), "config.epochs")
    if epochs <= 0:
        raise PosttrainError("config.epochs must be greater than zero")
    for key in ("batch_size", "gradient_accumulation_steps"):
        value = _require_int(config.get(key), f"config.{key}")
        if value <= 0:
            raise PosttrainError(f"config.{key} must be greater than zero")
    return config


def expected_steps_per_epoch(
    config: dict[str, Any], train_rows: Sequence[dict[str, str]]
) -> int:
    examples_per_update = (
        int(config["batch_size"]) * int(config["gradient_accumulation_steps"])
    )
    return math.ceil(len(train_rows) / examples_per_update)


def validate_quality_pass_manifest(path: Path, split: str) -> list[dict[str, str]]:
    rows = read_rows(path, split, task="escalation")
    if not rows:
        raise PosttrainError(f"{split} manifest contains no rows: {path}")
    bad = [row["image_id"] for row in rows if row.get("overall_quality") != "1"]
    if bad:
        raise PosttrainError(
            f"{split} manifest is not quality-pass-only; first failures: {bad[:5]}"
        )
    return rows


def validate_label_token_contract(
    provenance: dict[str, Any], *, name: str
) -> dict[str, Any]:
    contract = provenance.get("label_token_contract")
    if not isinstance(contract, dict) or set(contract) != set(
        ESCALATION_CLASS_TOKEN_IDS
    ):
        raise PosttrainError(f"{name} has no complete escalation label-token contract")
    normalized: dict[str, Any] = {}
    for label, pinned_token_id in ESCALATION_CLASS_TOKEN_IDS.items():
        details = contract.get(label)
        if not isinstance(details, dict):
            raise PosttrainError(f"{name} token contract lacks {label}")
        first_token_id = _require_int(
            details.get("first_token_id"), f"{name}.{label}.first_token_id"
        )
        encoding = details.get("full_encoding")
        if (
            not isinstance(encoding, list)
            or not encoding
            or any(isinstance(item, bool) or not isinstance(item, int) for item in encoding)
        ):
            raise PosttrainError(f"{name}.{label}.full_encoding is invalid")
        if first_token_id != pinned_token_id or encoding[0] != pinned_token_id:
            raise PosttrainError(
                f"{name}.{label} token changed: expected {pinned_token_id}, "
                f"got first_token_id={first_token_id}, encoding={encoding}"
            )
        normalized[label] = {
            "first_token_id": first_token_id,
            "full_encoding": list(encoding),
        }
    return normalized


def validate_completed_full_run(
    config: dict[str, Any], config_path: Path
) -> tuple[Path, dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    run_dir = resolve_existing_directory(config["output_dir"], "completed run")
    provenance_path = run_dir / "run_provenance.json"
    require_regular_file(provenance_path, "root run provenance")
    provenance = load_json(provenance_path, "root run provenance")
    if provenance.get("status") != "completed":
        raise PosttrainError(
            "root run provenance must record status='completed'; got "
            f"{provenance.get('status')!r}"
        )
    if provenance.get("failure") not in (None, {}):
        raise PosttrainError("completed root provenance unexpectedly records a failure")
    effective = provenance.get("effective_config")
    if not isinstance(effective, dict):
        raise PosttrainError("root provenance is missing effective_config")
    expected_effective = {
        "task": "escalation",
        "loss_scope": "decision_token",
        "max_steps": -1,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "processor_id": config["processor_id"],
        "processor_revision": config["processor_revision"],
        "train_manifest": config["train_manifest"],
        "val_manifest": config["val_manifest"],
    }
    mismatches = {
        key: {"expected": expected, "observed": effective.get(key)}
        for key, expected in expected_effective.items()
        if effective.get(key) != expected
    }
    if mismatches:
        raise PosttrainError(
            "root provenance/config mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    source_effective_mismatches = {
        key: {"config_source": expected, "effective": effective.get(key)}
        for key, expected in config.items()
        if effective.get(key) != expected
    }
    if source_effective_mismatches:
        raise PosttrainError(
            "root effective_config differs from its locked training config: "
            + json.dumps(source_effective_mismatches, sort_keys=True)
        )
    validate_label_token_contract(provenance, name="root run provenance")

    train_manifest = resolve(config["train_manifest"])
    val_manifest = resolve(config["val_manifest"])
    train_rows = validate_quality_pass_manifest(train_manifest, "train")
    val_rows = validate_quality_pass_manifest(val_manifest, "val")
    seed = _require_int(effective.get("seed"), "root effective_config.seed")
    if effective.get("stratified_sampling") is not False:
        raise PosttrainError("full run effective_config must use stratified_sampling=false")
    selected_rows_by_role: dict[str, list[dict[str, str]]] = {}
    for key, rows, role in (
        ("max_train_samples", train_rows, "training"),
        ("max_eval_samples", val_rows, "validation"),
    ):
        cap = effective.get(key)
        if cap is None:
            continue
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise PosttrainError(
                f"root provenance effective_config.{key} must be a positive integer "
                "or null"
            )
        if cap != len(rows):
            raise PosttrainError(
                f"full run {role} cap must equal all {len(rows)} rows; got {cap}"
            )
    selected_rows_by_role["train"] = select_rows_for_run(
        train_rows,
        effective.get("max_train_samples"),
        seed=seed,
        stratified=False,
        task="escalation",
    )
    selected_rows_by_role["validation"] = select_rows_for_run(
        val_rows,
        effective.get("max_eval_samples"),
        seed=seed + 1,
        stratified=False,
        task="escalation",
    )
    selected = provenance.get("selected_data")
    manifests = provenance.get("manifests")
    if not isinstance(selected, dict) or not isinstance(manifests, dict):
        raise PosttrainError("root provenance is missing selected_data or manifests")
    for role, path, rows, selected_prefix in (
        ("train", train_manifest, train_rows, "train"),
        ("validation", val_manifest, val_rows, "validation"),
    ):
        binding = manifests.get(role)
        if not isinstance(binding, dict):
            raise PosttrainError(f"root provenance is missing manifests.{role}")
        if binding.get("sha256") != sha256_file(path):
            raise PosttrainError(f"root provenance {role} manifest SHA mismatch")
        if binding.get("rows") != len(rows):
            raise PosttrainError(f"root provenance {role} row count mismatch")
        if selected.get(f"{selected_prefix}_rows") != len(rows):
            raise PosttrainError(
                f"full run did not select every {selected_prefix} row"
            )
        selected_rows = selected_rows_by_role[role]
        if selected.get(f"{selected_prefix}_rows_sha256") != selected_rows_sha256(
            selected_rows, "escalation"
        ):
            raise PosttrainError(
                f"root provenance selected {selected_prefix} rows SHA mismatch"
            )
    source = provenance.get("config_source")
    if not isinstance(source, dict):
        raise PosttrainError("root provenance is missing config_source")
    source_path = _require_string(source.get("path"), "config_source.path")
    values = source.get("values")
    if not isinstance(values, dict) or values != config:
        raise PosttrainError("root provenance config_source.values changed")
    recorded_sha = source.get("sha256")
    if not isinstance(recorded_sha, str) or not SHA256_RE.fullmatch(recorded_sha):
        raise PosttrainError("root provenance config_source.sha256 is invalid")
    if recorded_sha != sha256_file(config_path):
        raise PosttrainError("root provenance training config SHA mismatch")
    # Absolute roots legitimately differ between the A100 host and a verified
    # local mirror.  Content is authoritative; this suffix check catches a
    # misleading filename without rejecting the same config under /workspace.
    try:
        relative_config = config_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        relative_config = None
    normalized_source_path = source_path.replace("\\", "/")
    if relative_config is not None:
        suffix = relative_config.as_posix()
        if normalized_source_path != suffix and not normalized_source_path.endswith(
            "/" + suffix
        ):
            raise PosttrainError("root provenance config_source.path changed")
    elif Path(source_path).name != config_path.name:
        raise PosttrainError("root provenance config_source.path changed")

    # Validate both serialization and the exact training provenance before any
    # pre-existing report could make it possible to skip loading the adapter.
    root_config, root_weights = _adapter_files(run_dir)
    validate_safetensors_adapter(
        root_config,
        root_weights,
        provenance=provenance,
        expected_model_id=config["model_id"],
    )
    metadata = adapter_metadata(run_dir)
    validate_adapter_base(
        metadata,
        config["model_id"],
        config["model_revision"],
        config["processor_id"],
        config["processor_revision"],
    )
    return run_dir, provenance, train_rows, val_rows


def _adapter_files(directory: Path) -> tuple[Path, Path]:
    require_regular_directory(directory, "adapter candidate")
    config_path = directory / "adapter_config.json"
    require_regular_file(config_path, "adapter configuration")
    weights = directory / "adapter_model.safetensors"
    require_regular_file(weights, "safetensors adapter weights")
    legacy_pickle = directory / "adapter_model.bin"
    if legacy_pickle.exists() or legacy_pickle.is_symlink():
        raise PosttrainError(
            f"pickle-capable adapter_model.bin is forbidden: {legacy_pickle}"
        )
    return config_path, weights


def _target_module_matches(module: str, target_modules: Any) -> bool:
    if isinstance(target_modules, str) and target_modules:
        try:
            return re.fullmatch(target_modules, module) is not None
        except re.error as exc:
            raise PosttrainError(
                f"adapter target_modules regex is invalid: {exc}"
            ) from exc
    if (
        isinstance(target_modules, list)
        and target_modules
        and all(isinstance(item, str) and item for item in target_modules)
    ):
        return any(
            module == target or module.endswith("." + target)
            for target in target_modules
        )
    raise PosttrainError(
        "adapter target_modules must be a non-empty regex or list of names"
    )


def _ordered_training_parameter_names(
    parsed_keys: Sequence[tuple[str, re.Match[str]]],
) -> list[str]:
    ordered = sorted(
        parsed_keys,
        key=lambda item: (
            int(item[1].group("layer")),
            PROJECTION_ORDER[item[1].group("projection")],
            MATRIX_ORDER[item[1].group("matrix")],
        ),
    )
    return [
        key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        for key, _match in ordered
    ]


def validate_safetensors_adapter(
    config_path: Path,
    weights_path: Path,
    *,
    provenance: dict[str, Any],
    expected_model_id: str,
) -> dict[str, Any]:
    """Materialize and validate the exact LoRA bundle this trainer emits.

    Merely hashing a file named ``adapter_model.safetensors`` is insufficient:
    a corrupt header, truncated tensor, unsupported PEFT serialization, or
    mismatched rank could otherwise survive report reuse without model loading.
    """

    require_regular_file(config_path, "adapter configuration")
    require_regular_file(weights_path, "safetensors adapter weights")
    adapter_config = load_json(config_path, "adapter configuration")
    if adapter_config.get("peft_type") != "LORA":
        raise PosttrainError("adapter peft_type must be LORA")
    if adapter_config.get("task_type") != "CAUSAL_LM":
        raise PosttrainError("adapter task_type must be CAUSAL_LM")
    if adapter_config.get("base_model_name_or_path") != expected_model_id:
        raise PosttrainError(
            "adapter base_model_name_or_path does not match the locked model"
        )
    rank = _require_int(adapter_config.get("r"), "adapter_config.r")
    if rank <= 0:
        raise PosttrainError("adapter_config.r must be positive")
    alpha = _finite(adapter_config.get("lora_alpha"), "adapter_config.lora_alpha")
    dropout = _finite(
        adapter_config.get("lora_dropout"), "adapter_config.lora_dropout"
    )
    if alpha <= 0:
        raise PosttrainError("adapter_config.lora_alpha must be positive")
    if not 0.0 <= dropout < 1.0:
        raise PosttrainError("adapter_config.lora_dropout must be within [0, 1)")
    if adapter_config.get("bias") != "none":
        raise PosttrainError("only bias='none' LoRA adapters are supported")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise PosttrainError("modules_to_save tensors are unsupported")
    if adapter_config.get("rank_pattern") not in (None, {}):
        raise PosttrainError("rank_pattern adapters are unsupported")
    if adapter_config.get("use_dora") not in (None, False):
        raise PosttrainError("DoRA adapters are unsupported")
    unsupported_semantics = {
        key: adapter_config.get(key)
        for key, allowed in (
            ("use_rslora", (None, False)),
            ("use_qalora", (None, False)),
            ("lora_bias", (None, False)),
            ("fan_in_fan_out", (None, False)),
            ("alpha_pattern", (None, {})),
            ("layer_replication", (None, [])),
            ("target_parameters", (None, [])),
            ("trainable_token_indices", (None, [])),
            ("layers_to_transform", (None, [])),
            ("layers_pattern", (None, [])),
            ("exclude_modules", (None, [])),
            ("alora_invocation_tokens", (None, [])),
        )
        if adapter_config.get(key) not in allowed
    }
    if unsupported_semantics:
        raise PosttrainError(
            "adapter enables unsupported LoRA semantics: "
            + json.dumps(unsupported_semantics, sort_keys=True)
        )
    target_modules = adapter_config.get("target_modules")

    effective = provenance.get("effective_config")
    if not isinstance(effective, dict):
        raise PosttrainError("adapter provenance is missing effective_config")
    expected_effective = {
        "model_id": expected_model_id,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
    }
    mismatches = {
        key: {"expected": expected, "observed": effective.get(key)}
        for key, expected in expected_effective.items()
        if effective.get(key) != expected
    }
    configured_target = effective.get("lora_target_regex")
    if configured_target is not None and configured_target != target_modules:
        mismatches["lora_target_regex"] = {
            "expected": target_modules,
            "observed": configured_target,
        }
    if mismatches:
        raise PosttrainError(
            "adapter configuration/provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise PosttrainError(
            "validating adapter weights requires torch and safetensors"
        ) from exc

    parsed_keys: list[tuple[str, re.Match[str]]] = []
    tensors: dict[str, dict[str, Any]] = {}
    total_numel = 0
    try:
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if not isinstance(metadata, dict) or metadata.get("format") != "pt":
                raise PosttrainError(
                    "adapter safetensors metadata must record format='pt'"
                )
            keys = list(handle.keys())
            if not keys:
                raise PosttrainError("adapter safetensors contains no tensors")
            for key in keys:
                match = LORA_TENSOR_RE.fullmatch(key)
                if match is None:
                    raise PosttrainError(
                        f"unsupported or malformed adapter tensor key: {key}"
                    )
                module = key.rsplit(".lora_", 1)[0]
                if not _target_module_matches(module, target_modules):
                    raise PosttrainError(
                        f"adapter tensor is outside target_modules: {key}"
                    )
                tensor = handle.get_tensor(key)
                if tensor.dtype not in {
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                }:
                    raise PosttrainError(
                        f"adapter tensor {key} has unsupported dtype {tensor.dtype}"
                    )
                if tensor.ndim != 2 or tensor.numel() <= 0:
                    raise PosttrainError(
                        f"adapter tensor {key} must be a non-empty matrix"
                    )
                if not bool(torch.isfinite(tensor).all().item()):
                    raise PosttrainError(
                        f"adapter tensor {key} contains non-finite values"
                    )
                shape = tuple(int(value) for value in tensor.shape)
                tensors[key] = {"shape": shape, "dtype": str(tensor.dtype)}
                total_numel += int(tensor.numel())
                parsed_keys.append((key, match))
    except PosttrainError:
        raise
    except Exception as exc:
        raise PosttrainError(
            f"adapter_model.safetensors is invalid or unreadable: {exc}"
        ) from exc

    by_module: dict[str, dict[str, str]] = {}
    for key, match in parsed_keys:
        module = key.rsplit(".lora_", 1)[0]
        matrix = match.group("matrix")
        if matrix in by_module.setdefault(module, {}):
            raise PosttrainError(f"duplicate LoRA {matrix} tensor for {module}")
        by_module[module][matrix] = key
    for module, pair in sorted(by_module.items()):
        if set(pair) != {"A", "B"}:
            raise PosttrainError(f"unpaired LoRA tensors for {module}")
        a_shape = tensors[pair["A"]]["shape"]
        b_shape = tensors[pair["B"]]["shape"]
        if a_shape[0] != rank or b_shape[1] != rank:
            raise PosttrainError(
                f"LoRA rank mismatch for {module}: A={a_shape}, B={b_shape}, r={rank}"
            )
        if a_shape[1] <= 0 or b_shape[0] <= 0:
            raise PosttrainError(f"invalid LoRA dimensions for {module}")
        if tensors[pair["A"]]["dtype"] != tensors[pair["B"]]["dtype"]:
            raise PosttrainError(f"LoRA A/B dtype mismatch for {module}")

    training_names = _ordered_training_parameter_names(parsed_keys)
    names_sha = hashlib.sha256("\n".join(training_names).encode("utf-8")).hexdigest()
    trainable = provenance.get("trainable_parameters")
    if not isinstance(trainable, dict):
        raise PosttrainError("adapter provenance is missing trainable_parameters")
    expected_trainable = {
        "tensor_count": len(parsed_keys),
        "count": total_numel,
        "names_sha256": names_sha,
    }
    trainable_mismatches = {
        key: {"expected": expected, "observed": trainable.get(key)}
        for key, expected in expected_trainable.items()
        if trainable.get(key) != expected
    }
    if trainable_mismatches:
        raise PosttrainError(
            "adapter tensor inventory/provenance mismatch: "
            + json.dumps(trainable_mismatches, sort_keys=True)
        )
    return {
        "format": "pt",
        "tensor_count": len(parsed_keys),
        "parameter_count": total_numel,
        "training_parameter_names_sha256": names_sha,
        "rank": rank,
        "target_modules": target_modules,
    }


def _checkpoint_provenance(
    root_provenance: dict[str, Any],
    root_provenance_sha: str,
    checkpoint: Path,
    trainer_state: dict[str, Any],
    trainer_state_sha: str,
) -> dict[str, Any]:
    derivative = copy.deepcopy(root_provenance)
    derivative["post_training_candidate"] = {
        "schema_version": 1,
        "role": "retained_epoch_checkpoint",
        "relative_path": checkpoint.name,
        "root_run_provenance_sha256": root_provenance_sha,
        "trainer_state_sha256": trainer_state_sha,
        "global_step": trainer_state["global_step"],
        "epoch": float(trainer_state["epoch"]),
    }
    return derivative


def _validate_epoch_checkpoint_state(
    *,
    checkpoint: Path,
    trainer_state: dict[str, Any],
    named_step: int,
    expected_epochs: float,
    expected_steps_per_epoch: int,
) -> tuple[int, float]:
    global_step = _require_int(
        trainer_state.get("global_step"),
        f"{checkpoint.name}.trainer_state.global_step",
    )
    if global_step != named_step or global_step <= 0:
        raise PosttrainError(
            f"{checkpoint.name} global_step={global_step} does not match its name"
        )
    epoch = _finite(
        trainer_state.get("epoch"), f"{checkpoint.name}.trainer_state.epoch"
    )
    integer_epoch = round(epoch)
    if (
        epoch <= 0
        or not math.isclose(epoch, integer_epoch, rel_tol=0.0, abs_tol=1e-6)
        or epoch > expected_epochs + 1e-6
    ):
        raise PosttrainError(
            f"{checkpoint.name} is not a valid retained epoch checkpoint: "
            f"epoch={epoch}, configured epochs={expected_epochs}"
        )
    expected_step = integer_epoch * expected_steps_per_epoch
    if global_step != expected_step:
        raise PosttrainError(
            f"{checkpoint.name} step={global_step} is not the expected epoch-{integer_epoch} "
            f"boundary step {expected_step}"
        )
    history = trainer_state.get("log_history")
    if not isinstance(history, list):
        raise PosttrainError(f"{checkpoint.name} trainer state lacks log_history")
    matching_eval = [
        event
        for event in history
        if isinstance(event, dict)
        and event.get("step") == global_step
        and isinstance(event.get("eval_loss"), (int, float))
        and not isinstance(event.get("eval_loss"), bool)
        and math.isfinite(float(event["eval_loss"]))
        and isinstance(event.get("epoch"), (int, float))
        and math.isclose(
            float(event["epoch"]), epoch, rel_tol=0.0, abs_tol=1e-6
        )
    ]
    if not matching_eval:
        raise PosttrainError(
            f"{checkpoint.name} has no finite epoch-boundary evaluation event"
        )
    return global_step, float(epoch)


def discover_candidates(
    run_dir: Path,
    root_provenance: dict[str, Any],
    *,
    expected_epochs: float,
    expected_steps_per_epoch: int,
) -> list[Candidate]:
    require_regular_directory(run_dir, "completed run")
    root_config, root_weights = _adapter_files(run_dir)
    root_provenance_path = run_dir / "run_provenance.json"
    require_regular_file(root_provenance_path, "root run provenance")
    expected_model_id = _require_string(
        root_provenance.get("effective_config", {}).get("model_id")
        if isinstance(root_provenance.get("effective_config"), dict)
        else None,
        "root provenance effective_config.model_id",
    )
    validate_safetensors_adapter(
        root_config,
        root_weights,
        provenance=root_provenance,
        expected_model_id=expected_model_id,
    )
    root_sha = sha256_file(root_provenance_path)
    candidates = [
        Candidate(
            identifier="root",
            path=run_dir,
            role="completed_root_best_by_training_eval_loss",
            global_step=None,
            epoch=None,
            trainer_state_path=None,
            adapter_config_path=root_config,
            weights_path=root_weights,
            provenance_path=root_provenance_path,
        )
    ]
    checkpoint_paths: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(child.name)
        if match:
            if child.is_symlink() or not child.is_dir():
                raise PosttrainError(
                    f"checkpoint candidate must be an in-tree non-symlink directory: {child}"
                )
            resolved_child = child.resolve()
            if resolved_child.parent != run_dir.resolve():
                raise PosttrainError(f"checkpoint escaped completed run: {child}")
            checkpoint_paths.append((int(match.group(1)), resolved_child))
    if not checkpoint_paths:
        raise PosttrainError(
            f"completed full run has no retained epoch checkpoints: {run_dir}"
        )
    for named_step, checkpoint in sorted(checkpoint_paths):
        adapter_config, weights = _adapter_files(checkpoint)
        if sha256_file(adapter_config) != sha256_file(root_config):
            raise PosttrainError(
                f"{checkpoint.name} adapter configuration differs from its root run"
            )
        trainer_state_path = checkpoint / "trainer_state.json"
        require_regular_file(trainer_state_path, "checkpoint trainer state")
        trainer_state = load_json(trainer_state_path, "checkpoint trainer state")
        global_step, epoch = _validate_epoch_checkpoint_state(
            checkpoint=checkpoint,
            trainer_state=trainer_state,
            named_step=named_step,
            expected_epochs=expected_epochs,
            expected_steps_per_epoch=expected_steps_per_epoch,
        )
        state_sha = sha256_file(trainer_state_path)
        derived = _checkpoint_provenance(
            root_provenance, root_sha, checkpoint, trainer_state, state_sha
        )
        checkpoint_provenance = checkpoint / "run_provenance.json"
        if checkpoint_provenance.exists() or checkpoint_provenance.is_symlink():
            require_regular_file(
                checkpoint_provenance, "existing checkpoint provenance"
            )
            existing = load_json(checkpoint_provenance, "checkpoint provenance")
            if existing != derived:
                raise PosttrainError(
                    f"existing checkpoint provenance is stale or mismatched: "
                    f"{checkpoint_provenance}"
                )
        else:
            atomic_write_json(checkpoint_provenance, derived)
        require_regular_file(checkpoint_provenance, "checkpoint provenance")
        validate_safetensors_adapter(
            adapter_config,
            weights,
            provenance=derived,
            expected_model_id=expected_model_id,
        )
        adapter_metadata(checkpoint)
        candidates.append(
            Candidate(
                identifier=checkpoint.name,
                path=checkpoint,
                role="retained_epoch_checkpoint",
                global_step=global_step,
                epoch=epoch,
                trainer_state_path=trainer_state_path,
                adapter_config_path=adapter_config,
                weights_path=weights,
                provenance_path=checkpoint_provenance,
            )
        )
    return candidates


def load_external_candidate(
    directory: Path, *, index: int, config: dict[str, Any]
) -> Candidate:
    """Load one explicitly supplied completed adapter without trusting its data paths."""

    resolved = resolve_existing_directory(
        directory, f"external candidate {index}"
    )
    adapter_config, weights = _adapter_files(resolved)
    provenance_path = resolved / "run_provenance.json"
    require_regular_file(provenance_path, "external candidate provenance")
    provenance = load_json(provenance_path, "external candidate provenance")
    if provenance.get("status") != "completed" or provenance.get("failure") not in (
        None,
        {},
    ):
        raise PosttrainError(
            f"external candidate is not a successfully completed run: {resolved}"
        )
    effective = provenance.get("effective_config")
    if not isinstance(effective, dict):
        raise PosttrainError(f"external candidate lacks effective_config: {resolved}")
    required_effective = {
        "task": "escalation",
        "loss_scope": "decision_token",
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "processor_id": config["processor_id"],
        "processor_revision": config["processor_revision"],
    }
    mismatches = {
        key: {"expected": expected, "observed": effective.get(key)}
        for key, expected in required_effective.items()
        if effective.get(key) != expected
    }
    if mismatches:
        raise PosttrainError(
            "external candidate source/training contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    validate_label_token_contract(
        provenance, name=f"external candidate {resolved.name} provenance"
    )
    validate_safetensors_adapter(
        adapter_config,
        weights,
        provenance=provenance,
        expected_model_id=config["model_id"],
    )
    metadata = adapter_metadata(resolved)
    validate_adapter_base(
        metadata,
        config["model_id"],
        config["model_revision"],
        config["processor_id"],
        config["processor_revision"],
    )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved.name).strip("-.")
    if not safe_name:
        safe_name = "adapter"
    return Candidate(
        identifier=f"external-{index:02d}-{safe_name}",
        path=resolved,
        role="explicit_external_completed_adapter",
        global_step=None,
        epoch=None,
        trainer_state_path=None,
        adapter_config_path=adapter_config,
        weights_path=weights,
        provenance_path=provenance_path,
    )


def adapter_fingerprint(candidate: Candidate) -> dict[str, str]:
    """Return the exact content binding used for checkpoint de-duplication."""

    weights_sha = sha256_file(candidate.weights_path)
    config_sha = sha256_file(candidate.adapter_config_path)
    return {
        "adapter_weights_sha256": weights_sha,
        "adapter_config_sha256": config_sha,
        "exact_adapter_bundle_sha256": canonical_sha256(
            {
                "adapter_config_sha256": config_sha,
                "adapter_weights_sha256": weights_sha,
            }
        ),
    }


def deduplicate_candidates(
    candidates: Sequence[Candidate],
) -> tuple[list[Candidate], dict[str, dict[str, Any]]]:
    """Evaluate each exact adapter once while retaining every discovered alias.

    ``adapter_model`` bytes are the primary identity.  Reusing the same weights
    with different adapter configuration would change their interpretation, so
    that ambiguous state fails closed instead of being silently grouped.
    """

    if not candidates:
        raise PosttrainError("no adapter candidates were discovered")
    by_weights: dict[str, list[tuple[Candidate, dict[str, str]]]] = {}
    for candidate in candidates:
        fingerprint = adapter_fingerprint(candidate)
        by_weights.setdefault(
            fingerprint["adapter_weights_sha256"], []
        ).append((candidate, fingerprint))

    unique: list[Candidate] = []
    audit_by_canonical: dict[str, dict[str, Any]] = {}
    for weights_sha, members in by_weights.items():
        config_hashes = {
            fingerprint["adapter_config_sha256"]
            for _candidate, fingerprint in members
        }
        if len(config_hashes) != 1:
            raise PosttrainError(
                "identical adapter weights were discovered with different "
                f"adapter_config.json files (weights SHA {weights_sha}); refusing "
                "an ambiguous de-duplication"
            )
        # Prefer the completed root when it is an alias of the exact same
        # adapter. Otherwise use the earliest deterministic identifier.
        canonical, fingerprint = min(
            members,
            key=lambda item: (
                0 if item[0].identifier == "root" else 1,
                item[0].global_step if item[0].global_step is not None else -1,
                item[0].identifier,
            ),
        )
        aliases = sorted(
            members,
            key=lambda item: (
                0 if item[0].identifier == "root" else 1,
                item[0].global_step if item[0].global_step is not None else -1,
                item[0].identifier,
            ),
        )
        unique.append(canonical)
        audit_by_canonical[canonical.identifier] = {
            **fingerprint,
            "canonical_candidate": canonical.identifier,
            "aliases": [candidate_binding(member) for member, _ in aliases],
            "alias_count": len(aliases),
            "deduplicated_evaluations_saved": len(aliases) - 1,
        }
    unique.sort(
        key=lambda candidate: (
            0 if candidate.identifier == "root" else 1,
            candidate.global_step if candidate.global_step is not None else -1,
            candidate.identifier,
        )
    )
    return unique, audit_by_canonical


def candidate_binding(candidate: Candidate) -> dict[str, Any]:
    fingerprint = adapter_fingerprint(candidate)
    binding: dict[str, Any] = {
        "identifier": candidate.identifier,
        "role": candidate.role,
        "path": str(candidate.path),
        "global_step": candidate.global_step,
        "epoch": candidate.epoch,
        "adapter_config": {
            "path": str(candidate.adapter_config_path),
            "sha256": sha256_file(candidate.adapter_config_path),
        },
        "adapter_weights": {
            "path": str(candidate.weights_path),
            "bytes": candidate.weights_path.stat().st_size,
            "sha256": sha256_file(candidate.weights_path),
        },
        "exact_adapter_fingerprint": fingerprint,
        "training_provenance": {
            "path": str(candidate.provenance_path),
            "sha256": sha256_file(candidate.provenance_path),
        },
    }
    if candidate.trainer_state_path is not None:
        binding["trainer_state"] = {
            "path": str(candidate.trainer_state_path),
            "sha256": sha256_file(candidate.trainer_state_path),
        }
    return binding


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def derive_validation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("PRIORITY", "ROUTINE")
    if not results:
        raise PosttrainError("decision report contains zero results")
    matrix = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for index, result in enumerate(results):
        truth = result.get("truth")
        prediction = result.get("prediction")
        if truth not in labels or prediction not in labels:
            raise PosttrainError(
                f"result {index} has invalid truth/prediction: {truth!r}/{prediction!r}"
            )
        score = _finite(result.get("positive_probability"), f"result {index} score")
        if not 0.0 <= score <= 1.0:
            raise PosttrainError(f"result {index} score is outside [0, 1]")
        expected_prediction = "PRIORITY" if score >= REPORT_THRESHOLD else "ROUTINE"
        if prediction != expected_prediction:
            raise PosttrainError(
                f"result {index} prediction is inconsistent with threshold 0.5"
            )
        matrix[truth][prediction] += 1
    priority_total = sum(matrix["PRIORITY"].values())
    routine_total = sum(matrix["ROUTINE"].values())
    priority_recall = _ratio(matrix["PRIORITY"]["PRIORITY"], priority_total)
    routine_recall = _ratio(matrix["ROUTINE"]["ROUTINE"], routine_total)
    if priority_recall is None or routine_recall is None:
        raise PosttrainError("validation report must contain both classes")
    return {
        "samples": len(results),
        "confusion_matrix": matrix,
        "priority_recall": priority_recall,
        "routine_recall": routine_recall,
        "false_routine_count": matrix["PRIORITY"]["ROUTINE"],
        "false_priority_count": matrix["ROUTINE"]["PRIORITY"],
        "balanced_accuracy": (priority_recall + routine_recall) / 2.0,
        "accuracy": (
            matrix["PRIORITY"]["PRIORITY"] + matrix["ROUTINE"]["ROUTINE"]
        )
        / len(results),
        "roc_auc": positive_roc_auc(results, "PRIORITY"),
    }


def _assert_close(observed: Any, expected: float, name: str) -> None:
    number = _finite(observed, name)
    # Evaluator aggregate values are rounded to six decimal places.
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=5.1e-7):
        raise PosttrainError(
            f"{name} is inconsistent: report={number}, recomputed={expected}"
        )


def validate_decision_report(
    report_path: Path,
    manifest_path: Path,
    expected_split: str,
    candidate: Candidate,
    config: dict[str, Any],
) -> dict[str, Any]:
    require_regular_file(report_path, "decision-logit report")
    require_regular_file(manifest_path, f"{expected_split} manifest")
    try:
        validated = load_scored_partition(
            report_path,
            manifest_path,
            expected_split=expected_split,
            report_name=f"{candidate.identifier}_{expected_split}_report",
        )
    except (CalibrationInputError, FileNotFoundError, OSError, ValueError) as exc:
        raise PosttrainError(str(exc)) from exc
    payload = load_json(report_path, "decision-logit report")
    run = payload.get("run")
    summary = payload.get("summary")
    results = payload.get("results")
    if not isinstance(run, dict) or not isinstance(summary, dict) or not isinstance(
        results, list
    ):
        raise PosttrainError("decision report is missing run, summary, or results")
    expected_sources = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "processor_id": config["processor_id"],
        "processor_revision": config["processor_revision"],
    }
    for key, expected in expected_sources.items():
        if run.get(key) != expected:
            raise PosttrainError(
                f"decision report {key} mismatch: {run.get(key)!r} != {expected!r}"
            )
    adapter = run.get("adapter")
    training = run.get("training_contract")
    if not isinstance(adapter, dict) or not isinstance(training, dict):
        raise PosttrainError("decision report is missing adapter/training binding")
    actual_metadata = adapter_metadata(candidate.path)
    if actual_metadata is None:
        raise PosttrainError("candidate adapter metadata unexpectedly resolved to none")
    expected_adapter_metadata = {
        **actual_metadata,
        "weights_sha256": sha256_file(candidate.weights_path),
    }
    if adapter != expected_adapter_metadata:
        raise PosttrainError(
            "decision report adapter metadata differs from the exact candidate"
        )
    reported_adapter_path = Path(
        _require_string(adapter.get("path"), "report adapter path")
    ).resolve()
    if reported_adapter_path != candidate.path:
        raise PosttrainError("decision report binds a different adapter path")
    reported_config_path = Path(
        _require_string(adapter.get("config_path"), "report adapter config path")
    ).resolve()
    if reported_config_path != candidate.adapter_config_path.resolve():
        raise PosttrainError("decision report binds a different adapter configuration")
    if adapter.get("weights_sha256") != sha256_file(candidate.weights_path):
        raise PosttrainError("decision report adapter weights SHA mismatch")
    if training.get("provenance_sha256") != sha256_file(candidate.provenance_path):
        raise PosttrainError("decision report training provenance SHA mismatch")
    expected_training_contract = load_decision_training_contract(
        actual_metadata, config["model_id"], task="escalation"
    )
    if training != expected_training_contract:
        raise PosttrainError(
            "decision report training contract differs from candidate provenance"
        )
    if run.get("sampling") != "sequential":
        raise PosttrainError("decision report must use sequential full-manifest scoring")
    threshold = _finite(run.get("decision_threshold"), "decision threshold")
    if threshold != REPORT_THRESHOLD:
        raise PosttrainError("decision report threshold must be exactly 0.5")
    if not isinstance(run.get("completed_at"), str):
        raise PosttrainError("decision report has no completed_at timestamp")
    token_contract = run.get("token_contract")
    if not isinstance(token_contract, dict) or set(token_contract) != {
        "PRIORITY",
        "ROUTINE",
    }:
        raise PosttrainError("decision report token contract is incomplete")
    token_ids = []
    training_provenance = load_json(
        candidate.provenance_path, "candidate training provenance"
    )
    trained_tokens = validate_label_token_contract(
        training_provenance, name="candidate training provenance"
    )
    for label in ("PRIORITY", "ROUTINE"):
        details = token_contract.get(label)
        if not isinstance(details, dict):
            raise PosttrainError(f"decision report token contract lacks {label}")
        token_id = _require_int(details.get("token_id"), f"{label} token_id")
        encoding = details.get("full_encoding")
        if encoding != trained_tokens[label]["full_encoding"]:
            raise PosttrainError(
                f"decision report {label} encoding differs from training provenance"
            )
        if token_id != trained_tokens[label]["first_token_id"]:
            raise PosttrainError(
                f"decision report {label} token differs from training provenance"
            )
        token_ids.append(token_id)
    if len(set(token_ids)) != 2:
        raise PosttrainError("decision report class token IDs are not distinct")
    prefix = run.get("prefix_validation")
    if not isinstance(prefix, dict):
        raise PosttrainError("decision report lacks class-prefix validation")
    if _require_int(prefix.get("prefix_tokens"), "prefix_tokens") <= 0:
        raise PosttrainError("decision report prefix_tokens must be positive")

    metrics = derive_validation_metrics(results)
    reported_metrics = summary.get("metrics")
    if summary.get("samples") != metrics["samples"] or not isinstance(
        reported_metrics, dict
    ):
        raise PosttrainError("decision report aggregate sample count is inconsistent")
    if summary.get("confusion_matrix") != metrics["confusion_matrix"]:
        raise PosttrainError("decision report confusion matrix is inconsistent")
    expected_metric_keys = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "positive_recall": metrics["priority_recall"],
        "negative_recall": metrics["routine_recall"],
        "roc_auc_positive": metrics["roc_auc"],
    }
    for key, expected in expected_metric_keys.items():
        if expected is None:
            raise PosttrainError(f"recomputed validation metric {key} is undefined")
        _assert_close(reported_metrics.get(key), expected, f"summary.metrics.{key}")
    if reported_metrics.get("false_negative_count") != metrics["false_routine_count"]:
        raise PosttrainError("decision report false-negative count is inconsistent")
    if reported_metrics.get("false_positive_count") != metrics["false_priority_count"]:
        raise PosttrainError("decision report false-positive count is inconsistent")
    return {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "manifest_binding": validated["manifest_binding"],
        "adapter_binding": validated["adapter_binding"],
        "metrics": metrics,
        "completed_at": run["completed_at"],
    }


def input_content_binding(manifest: Path, expected_split: str) -> dict[str, Any]:
    rows = validate_quality_pass_manifest(manifest, expected_split)
    images: list[dict[str, Any]] = []
    for row in rows:
        image_path = resolve(row["image_path"])
        require_regular_file(image_path, f"{expected_split} input image")
        images.append(
            {
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "path": str(image_path),
                "bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
            }
        )
    return {
        "manifest": {
            "path": str(manifest.resolve()),
            "sha256": sha256_file(manifest),
            "expected_split": expected_split,
            "rows": len(rows),
        },
        "images": images,
        "ordered_image_content_sha256": canonical_sha256(images),
    }


def validate_partition_separation(
    partitions: dict[str, tuple[Path, list[dict[str, str]]]]
) -> dict[str, Any]:
    """Prove train/val/calibration/eval separation by IDs, paths, and bytes."""

    indexed: dict[str, dict[str, Any]] = {}
    for split, (manifest, rows) in partitions.items():
        patient_ids: set[str] = set()
        image_ids: set[str] = set()
        paths: set[str] = set()
        content_hashes: set[str] = set()
        image_bindings: list[dict[str, str]] = []
        class_patients = {"ROUTINE": set(), "PRIORITY": set()}
        for row in rows:
            image_id = row["image_id"]
            if image_id in image_ids:
                raise PosttrainError(f"{split} contains duplicate image_id {image_id}")
            image_ids.add(image_id)
            patient_id = row["patient_id"]
            patient_ids.add(patient_id)
            class_patients[row["escalation_label"]].add(patient_id)
            image_path = resolve(row["image_path"])
            require_regular_file(image_path, f"{split} image")
            resolved_path = str(image_path)
            if resolved_path in paths:
                raise PosttrainError(
                    f"{split} lists the same resolved image path more than once: "
                    f"{resolved_path}"
                )
            paths.add(resolved_path)
            content_sha = sha256_file(image_path)
            content_hashes.add(content_sha)
            image_bindings.append(
                {
                    "image_id": image_id,
                    "path": resolved_path,
                    "sha256": content_sha,
                }
            )
        indexed[split] = {
            "patient_ids": patient_ids,
            "image_ids": image_ids,
            "paths": paths,
            "content_hashes": content_hashes,
            "summary": {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "rows": len(rows),
                "patients": len(patient_ids),
                "ROUTINE_truth_patients": len(class_patients["ROUTINE"]),
                "PRIORITY_truth_patients": len(class_patients["PRIORITY"]),
                "ordered_image_bindings_sha256": canonical_sha256(image_bindings),
            },
        }

    overlap_audit: dict[str, Any] = {}
    names = list(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pair = f"{left}__{right}"
            overlaps = {
                "patient_ids": sorted(
                    indexed[left]["patient_ids"] & indexed[right]["patient_ids"]
                ),
                "image_ids": sorted(
                    indexed[left]["image_ids"] & indexed[right]["image_ids"]
                ),
                "resolved_paths": sorted(
                    indexed[left]["paths"] & indexed[right]["paths"]
                ),
                "image_content_sha256": sorted(
                    indexed[left]["content_hashes"]
                    & indexed[right]["content_hashes"]
                ),
            }
            if any(overlaps.values()):
                first = {
                    key: values[:5] for key, values in overlaps.items() if values
                }
                raise PosttrainError(
                    f"cross-partition leakage between {left} and {right}: "
                    + json.dumps(first, sort_keys=True)
                )
            overlap_audit[pair] = {key: 0 for key in overlaps}

    core = {
        "schema_version": 1,
        "artifact_type": "four_way_escalation_partition_separation",
        "status": "verified",
        "partition_order": names,
        "partitions": {
            name: indexed[name]["summary"] for name in names
        },
        "pairwise_overlap_counts": overlap_audit,
        "comparison_keys": [
            "patient_id",
            "image_id",
            "resolved_image_path",
            "image_content_sha256",
        ],
    }
    return with_integrity(core)


def decision_evidence_path(report_path: Path) -> Path:
    return report_path.with_suffix(report_path.suffix + ".evidence.json")


def build_decision_evidence(
    *,
    report_path: Path,
    candidate: Candidate,
    input_binding: dict[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "artifact_type": "decision_logit_report_evidence_binding",
        "report": {
            "path": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
        },
        "candidate": candidate_binding(candidate),
        "inputs": input_binding,
        "code": {
            "evaluator": {
                "path": str(PROJECT_ROOT / "ml" / "evaluate_decision_logits.py"),
                "sha256": sha256_file(
                    PROJECT_ROOT / "ml" / "evaluate_decision_logits.py"
                ),
            },
            "orchestrator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    return with_integrity(core)


def validate_decision_evidence(
    evidence_path: Path,
    *,
    report_path: Path,
    candidate: Candidate,
    manifest: Path,
    expected_split: str,
) -> dict[str, Any]:
    if not evidence_path.exists():
        raise PosttrainError(
            f"decision report exists without its evidence binding: {evidence_path}"
        )
    require_regular_file(evidence_path, "decision report evidence binding")
    observed = load_json(evidence_path, "decision report evidence binding")
    if not verify_summary_integrity(observed):
        raise PosttrainError("decision report evidence binding failed integrity check")
    expected = build_decision_evidence(
        report_path=report_path,
        candidate=candidate,
        input_binding=input_content_binding(manifest, expected_split),
    )
    if observed != expected:
        raise PosttrainError(
            "decision report evidence binding is stale: adapter configuration, "
            "weights, provenance, manifest, evaluator, or input image bytes changed"
        )
    return {
        "path": str(evidence_path),
        "sha256": sha256_file(evidence_path),
        "ordered_image_content_sha256": expected["inputs"][
            "ordered_image_content_sha256"
        ],
    }


def default_command_runner(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def evaluation_command(
    python_executable: str,
    candidate: Candidate,
    manifest: Path,
    expected_split: str,
    output: Path,
    config: dict[str, Any],
    progress_every: int,
) -> list[str]:
    return [
        python_executable,
        str(PROJECT_ROOT / "ml" / "evaluate_decision_logits.py"),
        "--task",
        "escalation",
        "--adapter-dir",
        str(candidate.path),
        "--model-id",
        config["model_id"],
        "--model-revision",
        config["model_revision"],
        "--processor-id",
        config["processor_id"],
        "--processor-revision",
        config["processor_revision"],
        "--manifest",
        str(manifest),
        "--expected-split",
        expected_split,
        "--sampling",
        "sequential",
        "--decision-threshold",
        str(REPORT_THRESHOLD),
        "--progress-every",
        str(progress_every),
        "--output",
        str(output),
    ]


def reuse_or_evaluate(
    *,
    candidate: Candidate,
    manifest: Path,
    expected_split: str,
    output: Path,
    config: dict[str, Any],
    python_executable: str,
    progress_every: int,
    runner: CommandRunner = default_command_runner,
) -> tuple[dict[str, Any], str]:
    evidence_path = decision_evidence_path(output)
    if output.is_symlink():
        raise PosttrainError(f"decision report path must not be a symlink: {output}")
    if evidence_path.is_symlink():
        raise PosttrainError(
            f"decision evidence path must not be a symlink: {evidence_path}"
        )
    if output.exists():
        validated = validate_decision_report(
            output, manifest, expected_split, candidate, config
        )
        validated["evidence_binding"] = validate_decision_evidence(
            evidence_path,
            report_path=output,
            candidate=candidate,
            manifest=manifest,
            expected_split=expected_split,
        )
        return validated, "reused_after_full_validation"
    if evidence_path.exists():
        raise PosttrainError(
            f"orphaned decision evidence exists without its report: {evidence_path}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    before_inputs = input_content_binding(manifest, expected_split)
    command = evaluation_command(
        python_executable,
        candidate,
        manifest,
        expected_split,
        output,
        config,
        progress_every,
    )
    try:
        runner(command, PROJECT_ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PosttrainError(
            f"decision-logit evaluation failed for {candidate.identifier}/"
            f"{expected_split}: {exc}"
        ) from exc
    if not output.is_file():
        raise PosttrainError(
            f"evaluator returned without writing the required report: {output}"
        )
    require_regular_file(output, "decision-logit report")
    validated = validate_decision_report(
        output, manifest, expected_split, candidate, config
    )
    after_inputs = input_content_binding(manifest, expected_split)
    if before_inputs != after_inputs:
        raise PosttrainError(
            "manifest or input image bytes changed while decision logits were running"
        )
    evidence = build_decision_evidence(
        report_path=output,
        candidate=candidate,
        input_binding=after_inputs,
    )
    atomic_write_json(evidence_path, evidence)
    validated["evidence_binding"] = validate_decision_evidence(
        evidence_path,
        report_path=output,
        candidate=candidate,
        manifest=manifest,
        expected_split=expected_split,
    )
    return validated, "evaluated_now"


def assess_candidate(
    metrics: dict[str, Any], criteria: SafetyCriteria
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for name, observed, minimum in (
        ("roc_auc", metrics["roc_auc"], criteria.minimum_roc_auc),
        (
            "balanced_accuracy",
            metrics["balanced_accuracy"],
            criteria.minimum_balanced_accuracy,
        ),
        (
            "priority_recall",
            metrics["priority_recall"],
            criteria.minimum_priority_recall,
        ),
        (
            "routine_recall",
            metrics["routine_recall"],
            criteria.minimum_routine_recall,
        ),
    ):
        if observed is None or observed < minimum:
            reasons.append(f"{name}={observed!r} below minimum {minimum}")
    return not reasons, reasons


def select_candidate(
    validation_records: list[dict[str, Any]], criteria: SafetyCriteria
) -> dict[str, Any]:
    """Select using validation records only; calibration/eval cannot enter here."""

    criteria.validate()
    for record in validation_records:
        is_eligible, reasons = assess_candidate(record["validation"]["metrics"], criteria)
        record["eligible"] = is_eligible
        record["rejection_reasons"] = reasons
    if not validation_records:
        raise PosttrainError("no validation records were available for selection")

    def rank(record: dict[str, Any]) -> tuple[Any, ...]:
        metrics = record["validation"]["metrics"]
        # The ranking contract is deliberately frozen and validation-only:
        # discrimination first, then the asymmetric miss count/recall, then
        # balanced accuracy. Root wins only an exact metric tie.
        return (
            -metrics["roc_auc"],
            metrics["false_routine_count"],
            -metrics["priority_recall"],
            -metrics["balanced_accuracy"],
            0 if record["candidate"]["identifier"] == "root" else 1,
            record["candidate"]["identifier"],
        )

    # Eligibility floors never reorder candidates. The documented ranking picks
    # one winner across every unique adapter; if that winner fails a floor, the
    # run stops instead of silently substituting a lower-ranked checkpoint.
    selected = min(validation_records, key=rank)
    if not selected["eligible"]:
        raise PosttrainError(
            "the top-ranked validation checkpoint failed the safety floors "
            f"({'; '.join(selected['rejection_reasons'])}); calibration and "
            "evaluation were not touched"
        )
    return selected


def with_integrity(core: dict[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(core)
    report["integrity"] = {
        "algorithm": "SHA-256",
        "canonical_report_without_integrity_sha256": canonical_sha256(core),
        "canonicalization": "UTF-8 JSON with sorted keys; top-level integrity omitted",
    }
    return report


def verify_summary_integrity(report: dict[str, Any]) -> bool:
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


def _selection_semantic_content(report: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(report)
    normalized.pop("integrity", None)
    normalized.pop("generated_at", None)
    candidates = normalized.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate.pop("report_action", None)
    return normalized


def reuse_or_write_selection(
    selection_path: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    if selection_path.exists():
        require_regular_file(selection_path, "checkpoint selection report")
        observed = load_json(selection_path, "checkpoint selection report")
        if not verify_summary_integrity(observed):
            raise PosttrainError("existing checkpoint selection failed integrity check")
        if _selection_semantic_content(observed) != _selection_semantic_content(
            expected
        ):
            raise PosttrainError(
                "existing immutable checkpoint selection does not match the "
                "current adapters, validation reports, ranking, or code"
            )
        return observed, "reused_after_full_validation"
    atomic_write_json(selection_path, expected)
    observed = load_json(selection_path, "checkpoint selection report")
    if not verify_summary_integrity(observed):
        raise PosttrainError("written checkpoint selection failed integrity check")
    return observed, "written_now"


def _completion_semantic_content(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _completion_semantic_content(item)
            for key, item in value.items()
            if key not in {"integrity", "generated_at", "action"}
        }
    if isinstance(value, list):
        return [_completion_semantic_content(item) for item in value]
    return value


def reuse_or_write_completion(
    completion_path: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    if completion_path.exists():
        require_regular_file(completion_path, "posttrain completion report")
        observed = load_json(completion_path, "posttrain completion report")
        if not verify_summary_integrity(observed):
            raise PosttrainError("existing completion report failed integrity check")
        if _completion_semantic_content(observed) != _completion_semantic_content(
            expected
        ):
            raise PosttrainError(
                "existing immutable completion report does not match current evidence"
            )
        return observed, "reused_after_full_validation"
    atomic_write_json(completion_path, expected)
    observed = load_json(completion_path, "posttrain completion report")
    if not verify_summary_integrity(observed):
        raise PosttrainError("written completion report failed integrity check")
    return observed, "written_now"


def build_selection_summary(
    *,
    config_path: Path,
    run_dir: Path,
    root_provenance: dict[str, Any],
    val_manifest: Path,
    validation_records: list[dict[str, Any]],
    selected: dict[str, Any],
    criteria: SafetyCriteria,
    deduplication: dict[str, dict[str, Any]],
    risk_profile: RiskProfile,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    core = {
        "schema_version": 1,
        "artifact_type": "gemma_escalation_validation_checkpoint_selection",
        "status": "checkpoint_selected_and_frozen",
        "generated_at": utc_now(),
        "selection_boundary": {
            "selection_partition": "val",
            "validation_manifest_owner": "primary --config only",
            "external_candidate_training_or_evaluation_manifests_used": False,
            "calibration_manifest_read_before_selection": False,
            "evaluation_manifest_read_before_selection": False,
            "calibration_or_evaluation_metrics_used": False,
            "selection_frozen_before_calibration_or_evaluation_scoring": True,
        },
        "post_selection_research_risk_profile": risk_profile.provenance(),
        "code": {
            "orchestrator": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "evaluator": {
                "path": str(PROJECT_ROOT / "ml" / "evaluate_decision_logits.py"),
                "sha256": sha256_file(PROJECT_ROOT / "ml" / "evaluate_decision_logits.py"),
            },
            "calibrator": {
                "path": str(PROJECT_ROOT / "ml" / "calibrate_escalation_adapter.py"),
                "sha256": sha256_file(PROJECT_ROOT / "ml" / "calibrate_escalation_adapter.py"),
            },
        },
        "inputs": {
            "training_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "completed_run": {
                "path": str(run_dir),
                "root_provenance_sha256": canonical_sha256(root_provenance),
                "root_provenance_file_sha256": sha256_file(
                    run_dir / "run_provenance.json"
                ),
            },
            "validation_manifest": {
                "path": str(val_manifest),
                "sha256": sha256_file(val_manifest),
                "rows": len(read_rows(val_manifest, "val", task="escalation")),
            },
        },
        "safety_criteria": {
            "eligibility_floors": {
                "roc_auc": criteria.minimum_roc_auc,
                "balanced_accuracy": criteria.minimum_balanced_accuracy,
                "priority_recall": criteria.minimum_priority_recall,
                "routine_recall": criteria.minimum_routine_recall,
            },
            "lexicographic_selection_order": [
                "highest PRIORITY ROC-AUC",
                "fewest false ROUTINE decisions among PRIORITY truths",
                "highest PRIORITY recall (paired safety tie-breaker)",
                "highest balanced accuracy",
                "completed root on an exact metric tie",
                "candidate identifier for deterministic final tie break",
            ],
            "threshold_used_for_checkpoint_comparison": REPORT_THRESHOLD,
        },
        "deduplication": {
            "identity": (
                "SHA-256 of adapter weights, with identical adapter-config "
                "SHA-256 required; exact aliases are scored once"
            ),
            "unique_adapter_count": len(deduplication),
            "discovered_candidate_count": sum(
                group["alias_count"] for group in deduplication.values()
            ),
            "groups_by_canonical_candidate": deduplication,
        },
        "candidates": validation_records,
        "selected": {
            "candidate": selected["candidate"],
            "validation": selected["validation"],
            "reason": (
                "passed every eligibility floor and won the documented "
                "validation-only ROC-AUC-first lexicographic ranking"
            ),
        },
    }
    return with_integrity(core)


def calibrator_command(
    python_executable: str,
    calibration_report: Path,
    evaluation_report: Path,
    calibration_manifest: Path,
    evaluation_manifest: Path,
    output: Path,
    false_routine_risk: float,
    false_priority_risk: float,
    delta: float,
) -> list[str]:
    return [
        python_executable,
        str(PROJECT_ROOT / "ml" / "calibrate_escalation_adapter.py"),
        "--calibration-report",
        str(calibration_report),
        "--evaluation-report",
        str(evaluation_report),
        "--calibration-manifest",
        str(calibration_manifest),
        "--evaluation-manifest",
        str(evaluation_manifest),
        "--false-routine-risk",
        str(false_routine_risk),
        "--false-priority-risk",
        str(false_priority_risk),
        "--delta",
        str(delta),
        "--output",
        str(output),
    ]


def validate_policy_report(
    policy_path: Path,
    calibration_report: Path,
    evaluation_report: Path,
    calibration_manifest: Path,
    evaluation_manifest: Path,
    candidate: Candidate,
    risk_profile: RiskProfile,
) -> dict[str, Any]:
    require_regular_file(policy_path, "selective policy report")
    report = load_json(policy_path, "selective policy report")
    if not verify_policy_integrity(report):
        raise PosttrainError("selective policy report failed its integrity check")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise PosttrainError("selective policy report is missing inputs")
    expected_hashes = {
        "calibration_decision_report": sha256_file(calibration_report),
        "evaluation_decision_report": sha256_file(evaluation_report),
        "calibration_manifest": sha256_file(calibration_manifest),
        "evaluation_manifest": sha256_file(evaluation_manifest),
    }
    for key, expected in expected_hashes.items():
        binding = inputs.get(key)
        if not isinstance(binding, dict) or binding.get("sha256") != expected:
            raise PosttrainError(f"selective policy report {key} binding mismatch")
    adapter = inputs.get("adapter")
    if not isinstance(adapter, dict):
        raise PosttrainError("selective policy report lacks adapter binding")
    if adapter.get("adapter_weights_sha256") != sha256_file(candidate.weights_path):
        raise PosttrainError("selective policy report binds different adapter weights")
    if adapter.get("training_provenance_sha256") != sha256_file(
        candidate.provenance_path
    ):
        raise PosttrainError("selective policy report binds different provenance")
    policy = report.get("policy")
    if not isinstance(policy, dict):
        raise PosttrainError("selective policy report lacks its frozen policy")
    false_routine = policy.get("false_routine")
    false_priority = policy.get("false_priority")
    if not isinstance(false_routine, dict) or not isinstance(false_priority, dict):
        raise PosttrainError("selective policy report lacks its risk-gate metadata")
    expected_profile = {
        "policy.false_routine.risk_limit": risk_profile.false_routine_risk,
        "policy.false_priority.risk_limit": risk_profile.false_priority_risk,
        "policy.per_gate_delta": risk_profile.delta,
    }
    observed_profile = {
        "policy.false_routine.risk_limit": false_routine.get("risk_limit"),
        "policy.false_priority.risk_limit": false_priority.get("risk_limit"),
        "policy.per_gate_delta": policy.get("per_gate_delta"),
    }
    for name, expected in expected_profile.items():
        observed = observed_profile[name]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise PosttrainError(f"selective policy report {name} is not numeric")
        if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
            raise PosttrainError(
                f"selective policy report {name}={observed!r} does not match "
                f"requested value {expected!r}; refusing stale policy reuse"
            )
    try:
        expected_report = build_policy_report(
            calibration_report=calibration_report,
            evaluation_report=evaluation_report,
            calibration_manifest=calibration_manifest,
            evaluation_manifest=evaluation_manifest,
            false_routine_risk=risk_profile.false_routine_risk,
            false_priority_risk=risk_profile.false_priority_risk,
            delta=risk_profile.delta,
        )
    except (CalibrationInputError, FileNotFoundError, OSError, ValueError) as exc:
        raise PosttrainError(
            f"cannot reconstruct the expected frozen selective policy: {exc}"
        ) from exc

    def semantic_content(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(payload)
        normalized.pop("integrity", None)
        normalized.pop("generated_at", None)
        return normalized

    if semantic_content(report) != semantic_content(expected_report):
        raise PosttrainError(
            "selective policy report differs from a fresh deterministic rebuild; "
            "refusing threshold or metric tampering"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a completed RetinaPriority checkpoint on full validation, "
            "then score disjoint calibration/evaluation and freeze a policy."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/configs/gemma4_26b_escalation_quality_pass_full.json"),
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional completed escalation adapter directory to compare on "
            "the config's locked validation manifest; repeatable"
        ),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--min-val-roc-auc", type=float, default=0.70)
    parser.add_argument("--min-val-balanced-accuracy", type=float, default=0.60)
    parser.add_argument("--min-val-priority-recall", type=float, default=0.80)
    parser.add_argument("--min-val-routine-recall", type=float, default=0.50)
    parser.add_argument(
        "--false-routine-risk",
        type=float,
        default=DEFAULT_FALSE_ROUTINE_RISK,
        help=(
            "patient-level false-ROUTINE risk limit for post-selection "
            "calibration (default: 0.10 research profile)"
        ),
    )
    parser.add_argument(
        "--false-priority-risk",
        type=float,
        default=DEFAULT_FALSE_PRIORITY_RISK,
        help=(
            "patient-level false-PRIORITY risk limit for post-selection "
            "calibration (default: 0.10 research profile)"
        ),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help=(
            "one-sided error probability per gate (default: 0.05; with zero "
            "errors, n=35/n=40 yield 8.20%%/7.22%% upper bounds)"
        ),
    )
    return parser


def run_orchestration(
    args: argparse.Namespace, runner: CommandRunner = default_command_runner
) -> dict[str, Any]:
    if args.progress_every < 0:
        raise PosttrainError("progress_every cannot be negative")
    risk_profile = RiskProfile(
        false_routine_risk=args.false_routine_risk,
        false_priority_risk=args.false_priority_risk,
        delta=args.delta,
    )
    # Validate this before any evaluator could allocate the GPU.
    risk_profile.validate()
    unresolved_config_path = resolve_project_path(args.config)
    require_regular_file(unresolved_config_path, "training config")
    config_path = unresolved_config_path.resolve()
    config = load_full_config(config_path)
    run_dir, root_provenance, train_rows, val_rows = validate_completed_full_run(
        config, config_path
    )
    discovered_candidates = discover_candidates(
        run_dir,
        root_provenance,
        expected_epochs=float(config["epochs"]),
        expected_steps_per_epoch=expected_steps_per_epoch(config, train_rows),
    )
    for index, directory in enumerate(args.candidate_dir, start=1):
        discovered_candidates.append(
            load_external_candidate(directory, index=index, config=config)
        )
    candidates, deduplication = deduplicate_candidates(discovered_candidates)
    work_dir = resolve_output_directory(
        args.work_dir if args.work_dir else run_dir / "posttrain",
        "post-training work directory",
    )
    val_manifest = resolve(config["val_manifest"])
    criteria = SafetyCriteria(
        minimum_roc_auc=args.min_val_roc_auc,
        minimum_balanced_accuracy=args.min_val_balanced_accuracy,
        minimum_priority_recall=args.min_val_priority_recall,
        minimum_routine_recall=args.min_val_routine_recall,
    )
    # Reject malformed floors before any report evaluation can allocate CUDA.
    criteria.validate()

    # Phase 1: only the validation manifest is opened/scored before selection.
    validation_records: list[dict[str, Any]] = []
    for candidate in candidates:
        report_path = work_dir / "decision-reports" / candidate.identifier / "val.json"
        validation, action = reuse_or_evaluate(
            candidate=candidate,
            manifest=val_manifest,
            expected_split="val",
            output=report_path,
            config=config,
            python_executable=args.python_executable,
            progress_every=args.progress_every,
            runner=runner,
        )
        validation_records.append(
            {
                "candidate": candidate_binding(candidate),
                "exact_adapter_aliases": deduplication[candidate.identifier],
                "validation": validation,
                "report_action": action,
            }
        )
    selected_record = select_candidate(validation_records, criteria)
    expected_selection_report = build_selection_summary(
        config_path=config_path,
        run_dir=run_dir,
        root_provenance=root_provenance,
        val_manifest=val_manifest,
        validation_records=validation_records,
        selected=selected_record,
        criteria=criteria,
        deduplication=deduplication,
        risk_profile=risk_profile,
    )
    selection_path = work_dir / "checkpoint-selection.json"
    selection_report, selection_action = reuse_or_write_selection(
        selection_path, expected_selection_report
    )

    # Phase 2 starts only after the validation-only selection is durable.
    selected_id = selection_report["selected"]["candidate"]["identifier"]
    selected_candidate = next(
        candidate for candidate in candidates if candidate.identifier == selected_id
    )
    calibration_manifest = resolve(config["calibration_manifest"])
    evaluation_manifest = resolve(config["eval_manifest"])
    calibration_rows = validate_quality_pass_manifest(
        calibration_manifest, "calibration"
    )
    evaluation_rows = validate_quality_pass_manifest(evaluation_manifest, "eval")
    separation = validate_partition_separation(
        {
            "train": (resolve(config["train_manifest"]), train_rows),
            "val": (val_manifest, val_rows),
            "calibration": (calibration_manifest, calibration_rows),
            "eval": (evaluation_manifest, evaluation_rows),
        }
    )
    separation_path = work_dir / "four-way-partition-separation.json"
    if separation_path.exists():
        require_regular_file(separation_path, "partition separation report")
        existing_separation = load_json(
            separation_path, "partition separation report"
        )
        if (
            not verify_summary_integrity(existing_separation)
            or existing_separation != separation
        ):
            raise PosttrainError(
                "existing partition separation evidence is stale or corrupted"
            )
        separation_action = "reused_after_full_validation"
    else:
        atomic_write_json(separation_path, separation)
        separation_action = "written_now"
    downstream_dir = work_dir / "decision-reports" / selected_id
    calibration_path = downstream_dir / "calibration.json"
    evaluation_path = downstream_dir / "eval.json"
    calibration, calibration_action = reuse_or_evaluate(
        candidate=selected_candidate,
        manifest=calibration_manifest,
        expected_split="calibration",
        output=calibration_path,
        config=config,
        python_executable=args.python_executable,
        progress_every=args.progress_every,
        runner=runner,
    )
    evaluation, evaluation_action = reuse_or_evaluate(
        candidate=selected_candidate,
        manifest=evaluation_manifest,
        expected_split="eval",
        output=evaluation_path,
        config=config,
        python_executable=args.python_executable,
        progress_every=args.progress_every,
        runner=runner,
    )
    policy_path = work_dir / "selective-policy-evaluation.json"
    if policy_path.is_symlink():
        raise PosttrainError(
            f"selective policy output path must not be a symlink: {policy_path}"
        )
    if policy_path.exists():
        policy_action = "reused_after_full_validation"
        policy = validate_policy_report(
            policy_path,
            calibration_path,
            evaluation_path,
            calibration_manifest,
            evaluation_manifest,
            selected_candidate,
            risk_profile,
        )
    else:
        command = calibrator_command(
            args.python_executable,
            calibration_path,
            evaluation_path,
            calibration_manifest,
            evaluation_manifest,
            policy_path,
            args.false_routine_risk,
            args.false_priority_risk,
            args.delta,
        )
        try:
            runner(command, PROJECT_ROOT)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PosttrainError(f"escalation calibration failed: {exc}") from exc
        if not policy_path.is_file():
            raise PosttrainError("calibrator returned without writing its policy report")
        policy = validate_policy_report(
            policy_path,
            calibration_path,
            evaluation_path,
            calibration_manifest,
            evaluation_manifest,
            selected_candidate,
            risk_profile,
        )
        policy_action = "calibrated_now"

    completion_risk_profile = risk_profile.provenance()
    calibration_partition = separation["partitions"]["calibration"]
    completion_risk_profile["actual_calibration_denominators"] = {
        "PRIORITY_patients_for_false_ROUTINE_gate": calibration_partition[
            "PRIORITY_truth_patients"
        ],
        "ROUTINE_patients_for_false_PRIORITY_gate": calibration_partition[
            "ROUTINE_truth_patients"
        ],
        "reference_35_40_match": (
            calibration_partition["PRIORITY_truth_patients"]
            == REFERENCE_PRIORITY_CALIBRATION_PATIENTS
            and calibration_partition["ROUTINE_truth_patients"]
            == REFERENCE_ROUTINE_CALIBRATION_PATIENTS
        ),
    }
    completion = with_integrity(
        {
            "schema_version": 1,
            "artifact_type": "gemma_escalation_posttraining_completion",
            "status": "completed_research_evaluation_not_runtime_promotion",
            "generated_at": utc_now(),
            "selection": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
                "selected_candidate": selected_id,
                "selection_remained_frozen": True,
                "action": selection_action,
            },
            "research_risk_profile": completion_risk_profile,
            "four_way_partition_separation": {
                "path": str(separation_path),
                "sha256": sha256_file(separation_path),
                "integrity_verified": verify_summary_integrity(separation),
                "action": separation_action,
            },
            "downstream": {
                "calibration_decision_report": {
                    **calibration,
                    "action": calibration_action,
                },
                "evaluation_decision_report": {
                    **evaluation,
                    "action": evaluation_action,
                },
                "selective_policy_report": {
                    "path": str(policy_path),
                    "sha256": sha256_file(policy_path),
                    "integrity_verified": verify_policy_integrity(policy),
                    "action": policy_action,
                },
            },
            "recommendation": {
                "runtime_promotion_authorized": False,
                "reason": (
                    "checkpoint selection and retrospective selective-policy "
                    "measurement do not constitute clinical validation"
                ),
            },
        }
    )
    completion_path = work_dir / "posttrain-completion.json"
    _completion_report, completion_action = reuse_or_write_completion(
        completion_path, completion
    )
    return {
        "selection_report": str(selection_path),
        "completion_report": str(completion_path),
        "selected_candidate": selected_id,
        "policy_report": str(policy_path),
        "completion_action": completion_action,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_orchestration(args)
    except (PosttrainError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
