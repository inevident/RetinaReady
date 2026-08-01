#!/usr/bin/env python3
"""Read-only readiness checks for RetinaReady Gemma 4 QLoRA.

This program never imports or launches ``train_qlora.py``. Hugging Face checks
request repository metadata and small JSON configuration files only; model
weights are never downloaded.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = Path(__file__).with_name("requirements-train.txt")
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gemma4_26b_smoke.json"
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
IMPORT_NAMES = {
    "pillow": "PIL",
    "protobuf": "google.protobuf",
    "huggingface-hub": "huggingface_hub",
}
WEIGHT_SUFFIXES = {
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".safetensors",
}


@dataclass
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] | None = None


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        name: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.checks.append(Check(name, status, message, details))

    @property
    def failures(self) -> int:
        return sum(check.status == "FAIL" for check in self.checks)

    @property
    def warnings(self) -> int:
        return sum(check.status == "WARN" for check in self.checks)

    def payload(self) -> dict[str, Any]:
        return {
            "ready": self.failures == 0,
            "summary": {
                "failures": self.failures,
                "warnings": self.warnings,
                "checks": len(self.checks),
            },
            "checks": [asdict(check) for check in self.checks],
        }

    def print_human(self) -> None:
        for check in self.checks:
            print(f"[{check.status:4}] {check.name}: {check.message}")
        state = "READY" if self.failures == 0 else "NOT READY"
        print(
            f"\nA100 preflight: {state} "
            f"({self.failures} blocker(s), {self.warnings} warning(s))"
        )
        if self.failures:
            print("No training was started. Resolve every FAIL before training.")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(report: Report, path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("top-level value is not an object")
        task = config.get("task", "quality")
        if task not in {"quality", "escalation"}:
            raise ValueError(f"unsupported task {task!r}")
        for key in (
            "model_id",
            "model_revision",
            "processor_id",
            "processor_revision",
            "train_manifest",
            "val_manifest",
        ):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"missing non-empty string {key!r}")
        for key in ("calibration_manifest", "eval_manifest"):
            if key in config and (
                not isinstance(config[key], str) or not config[key].strip()
            ):
                raise ValueError(f"{key!r} must be a non-empty string when set")
        if task == "quality" and any(
            key in config for key in ("calibration_manifest", "eval_manifest")
        ):
            raise ValueError(
                "calibration_manifest and eval_manifest are escalation-only"
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("config.training", "FAIL", f"{path}: {exc}")
        return None
    report.add(
        "config.training",
        "PASS",
        f"loaded {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}",
        {
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "processor_id": config["processor_id"],
            "processor_revision": config["processor_revision"],
            "task": config.get("task", "quality"),
        },
    )
    return config


def manifest_paths_for_config(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve every manifest required by the configured task.

    Escalation calibration/evaluation paths are overridable so a quality-pass
    cohort cannot accidentally be validated against the legacy mixed-quality
    defaults.  Existing configs retain their current defaults.
    """

    task = config.get("task", "quality")
    if task not in {"quality", "escalation"}:
        raise ValueError(f"unsupported task {task!r}")
    paths = {
        "train": resolve_project_path(config["train_manifest"]),
        "val": resolve_project_path(config["val_manifest"]),
    }
    if task == "quality":
        paths["test"] = PROJECT_ROOT / "data" / "manifests" / "test.csv"
        return paths

    paths["calibration"] = resolve_project_path(
        config.get(
            "calibration_manifest",
            "data/escalation-manifests/calibration.csv",
        )
    )
    paths["eval"] = resolve_project_path(
        config.get("eval_manifest", "data/escalation-manifests/eval.csv")
    )
    return paths


def check_system(report: Report, minimum_disk_gib: float) -> None:
    system = platform.system()
    machine = platform.machine()
    if system == "Linux":
        report.add("system.os", "PASS", f"Linux {platform.release()} ({machine})")
    else:
        report.add(
            "system.os",
            "FAIL",
            f"{system} {platform.release()} ({machine}); training requires Linux",
        )

    version = platform.python_version()
    if sys.version_info < (3, 10):
        report.add("system.python", "FAIL", f"Python {version}; require >=3.10")
    elif sys.version_info >= (3, 13):
        report.add(
            "system.python",
            "WARN",
            f"Python {version}; Python 3.11 is the conservative CUDA-stack choice",
        )
    else:
        report.add("system.python", "PASS", f"Python {version}")

    disk = shutil.disk_usage(PROJECT_ROOT)
    free_gib = disk.free / 2**30
    status = "PASS" if free_gib >= minimum_disk_gib else "FAIL"
    report.add(
        "system.disk",
        status,
        f"{free_gib:.1f} GiB free; require >= {minimum_disk_gib:.0f} GiB",
        {"free_gib": round(free_gib, 2), "path": str(PROJECT_ROOT)},
    )


def check_c_compiler(report: Report) -> None:
    """Require the host compiler used by Triton's first-run CUDA helpers."""
    configured = os.getenv("CC", "").strip()
    candidates: list[str] = []
    if configured:
        try:
            candidates.append(shlex.split(configured)[0])
        except ValueError:
            report.add(
                "system.c_compiler",
                "FAIL",
                f"CC is not a valid shell command: {configured!r}",
            )
            return
    candidates.extend(("cc", "gcc", "clang"))

    compiler_path = next(
        (resolved for candidate in candidates if (resolved := shutil.which(candidate))),
        None,
    )
    if compiler_path is None:
        report.add(
            "system.c_compiler",
            "FAIL",
            "no C compiler found; Triton requires CC, cc, gcc, or clang at runtime",
        )
        return

    version = command_output([compiler_path, "--version"])
    first_line = version.splitlines()[0] if version else "version unavailable"
    report.add(
        "system.c_compiler",
        "PASS",
        f"{compiler_path}: {first_line}",
        {"path": compiler_path, "configured_cc": configured or None},
    )


def command_output(command: list[str]) -> str | None:
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def check_gpu(report: Report, minimum_vram_gib: float) -> Any | None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            torch = importlib.import_module("torch")
    except Exception as exc:  # dependency loaders can raise more than ImportError
        report.add("hardware.cuda", "FAIL", f"PyTorch import failed: {exc}")
        report.add("hardware.gpu", "FAIL", "cannot inspect a CUDA GPU")
        report.add("hardware.bf16", "FAIL", "cannot inspect BF16 support")
        return None

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not torch.cuda.is_available():
        mps = bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        )
        suffix = "; MPS is present but unsupported for this trainer" if mps else ""
        report.add(
            "hardware.cuda",
            "FAIL",
            f"torch.cuda.is_available() is false (torch CUDA={cuda_version}){suffix}",
        )
        report.add("hardware.gpu", "FAIL", "no CUDA devices detected")
        report.add("hardware.bf16", "FAIL", "CUDA BF16 support unavailable")
        return torch

    device_count = torch.cuda.device_count()
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "vram_gib": round(properties.total_memory / 2**30, 2),
                "compute_capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(index)
                ),
            }
        )
    report.add(
        "hardware.cuda",
        "PASS" if cuda_version else "FAIL",
        f"CUDA available through PyTorch {torch.__version__}; runtime {cuda_version}",
        {"device_count": device_count},
    )

    primary = devices[0]
    vram = float(primary["vram_gib"])
    gpu_status = "PASS" if vram >= minimum_vram_gib else "FAIL"
    report.add(
        "hardware.gpu",
        gpu_status,
        f"{primary['name']}, {vram:.1f} GiB, compute {primary['compute_capability']}; "
        f"require >= {minimum_vram_gib:.0f} GiB",
        {"devices": devices, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
    )
    if "A100" in primary["name"].upper():
        report.add("hardware.target", "PASS", "NVIDIA A100 detected")
    else:
        report.add(
            "hardware.target",
            "WARN",
            f"{primary['name']} is not an A100; capability checks are authoritative",
        )

    bf16 = bool(torch.cuda.is_bf16_supported())
    report.add(
        "hardware.bf16",
        "PASS" if bf16 else "FAIL",
        "native CUDA BF16 is supported" if bf16 else "native CUDA BF16 is not supported",
    )

    smi = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    report.add(
        "hardware.nvidia_smi",
        "PASS" if smi else "WARN",
        smi.replace("\n", " | ") if smi else "nvidia-smi unavailable or returned an error",
    )
    return torch


def parse_requirements() -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            from packaging.requirements import Requirement

            requirement = Requirement(line)
            requirements.append((requirement.name, str(requirement.specifier)))
        except ImportError:
            name = line.split("=", 1)[0].split("<", 1)[0].split(">", 1)[0]
            requirements.append((name.strip(), ""))
    return requirements


def check_dependencies(report: Report, torch: Any | None) -> None:
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:
        SpecifierSet = None
        Version = None
        report.add(
            "dependencies.packaging",
            "WARN",
            "packaging is missing; minimum versions cannot be evaluated",
        )

    imported: dict[str, Any] = {}
    for distribution, specifier in parse_requirements():
        normalized = distribution.lower().replace("_", "-")
        module_name = IMPORT_NAMES.get(normalized, normalized.replace("-", "_"))
        try:
            version = importlib.metadata.version(distribution)
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                imported[module_name] = importlib.import_module(module_name)
            matches = True
            if specifier and SpecifierSet is not None and Version is not None:
                matches = Version(version) in SpecifierSet(specifier)
            if not matches:
                report.add(
                    f"dependency.{normalized}",
                    "FAIL",
                    f"{version} does not satisfy {specifier}",
                )
            else:
                report.add(
                    f"dependency.{normalized}",
                    "PASS",
                    f"{version} imported successfully"
                    + (f" ({specifier})" if specifier else ""),
                )
        except importlib.metadata.PackageNotFoundError:
            report.add(f"dependency.{normalized}", "FAIL", "not installed")
        except Exception as exc:
            report.add(
                f"dependency.{normalized}",
                "FAIL",
                f"installed but import failed: {type(exc).__name__}: {exc}",
            )

    api_requirements = {
        "transformers": (
            "AutoModelForMultimodalLM",
            "AutoProcessor",
            "BitsAndBytesConfig",
        ),
        "peft": ("LoraConfig", "prepare_model_for_kbit_training"),
        "trl": ("SFTConfig", "SFTTrainer"),
    }
    missing_apis: list[str] = []
    for module_name, names in api_requirements.items():
        module = imported.get(module_name)
        for name in names:
            if module is None or not hasattr(module, name):
                missing_apis.append(f"{module_name}.{name}")
    report.add(
        "dependencies.training_api",
        "PASS" if not missing_apis else "FAIL",
        "required trainer APIs are available"
        if not missing_apis
        else "missing: " + ", ".join(missing_apis),
    )

    bnb = imported.get("bitsandbytes")
    if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
        compiled = getattr(
            getattr(getattr(bnb, "cextension", None), "lib", None),
            "compiled_with_cuda",
            None,
        )
        if compiled is False:
            report.add(
                "dependencies.bitsandbytes_cuda",
                "FAIL",
                "bitsandbytes was not compiled with CUDA support",
            )
        elif compiled is None:
            report.add(
                "dependencies.bitsandbytes_cuda",
                "WARN",
                "could not confirm bitsandbytes CUDA compilation",
            )
        else:
            report.add(
                "dependencies.bitsandbytes_cuda",
                "PASS",
                "bitsandbytes CUDA backend detected",
            )


def read_manifest(
    report: Report,
    path: Path,
    expected_split: str,
    task: str = "quality",
) -> list[dict[str, str]]:
    try:
        if task == "quality":
            required_columns = QUALITY_REQUIRED_COLUMNS
        elif task == "escalation":
            required_columns = ESCALATION_REQUIRED_COLUMNS
        else:
            raise ValueError(f"unsupported task {task!r}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required_columns - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"missing columns: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError("manifest is empty")
        for row in rows:
            if row["split"] != expected_split:
                raise ValueError(
                    f"{row.get('image_id')}: split {row['split']!r} != {expected_split!r}"
                )
            if row["overall_quality"] not in {"0", "1"}:
                raise ValueError(
                    f"{row.get('image_id')}: invalid overall_quality "
                    f"{row['overall_quality']!r}"
                )
            if task == "quality":
                if row["quality_label"] not in {"READY", "RETAKE"}:
                    raise ValueError(
                        f"{row.get('image_id')}: invalid label "
                        f"{row['quality_label']!r}"
                    )
                expected_label = (
                    "READY" if row["overall_quality"] == "1" else "RETAKE"
                )
                if row["quality_label"] != expected_label:
                    raise ValueError(f"{row.get('image_id')}: label mapping mismatch")
            else:
                if row["escalation_label"] not in {"ROUTINE", "PRIORITY"}:
                    raise ValueError(
                        f"{row.get('image_id')}: invalid escalation label "
                        f"{row['escalation_label']!r}"
                    )
                try:
                    grade = int(row["dr_grade"])
                except ValueError as exc:
                    raise ValueError(
                        f"{row.get('image_id')}: invalid dr_grade "
                        f"{row['dr_grade']!r}"
                    ) from exc
                if grade not in {0, 1, 2, 3, 4}:
                    raise ValueError(
                        f"{row.get('image_id')}: invalid dr_grade {grade!r}"
                    )
                expected_label = "PRIORITY" if grade >= 2 else "ROUTINE"
                if row["escalation_label"] != expected_label:
                    raise ValueError(
                        f"{row.get('image_id')}: escalation label mapping mismatch"
                    )
    except (OSError, ValueError) as exc:
        report.add(f"dataset.manifest.{expected_split}", "FAIL", f"{path}: {exc}")
        return []
    report.add(
        f"dataset.manifest.{expected_split}",
        "PASS",
        f"{len(rows)} rows / {len({row['patient_id'] for row in rows})} patients",
    )
    return rows


def check_dataset(
    report: Report,
    config: dict[str, Any],
    verify_image_headers: bool,
) -> None:
    task = config.get("task", "quality")
    manifest_paths = manifest_paths_for_config(config)
    rows_by_split = {
        split: read_manifest(report, path, split, task=task)
        for split, path in manifest_paths.items()
    }
    if not all(rows_by_split.values()):
        report.add(
            "dataset.integrity",
            "FAIL",
            "one or more required manifests could not be validated",
        )
        return

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    ids = [row["image_id"] for row in all_rows]
    duplicate_ids = len(ids) - len(set(ids))
    missing: list[str] = []
    empty: list[str] = []
    image_paths: list[Path] = []
    for row in all_rows:
        path = resolve_project_path(row["image_path"])
        image_paths.append(path)
        if not path.is_file():
            missing.append(str(path))
        elif path.stat().st_size == 0:
            empty.append(str(path))

    overlaps: dict[str, list[str]] = {}
    split_names = list(rows_by_split)
    for index, left in enumerate(split_names):
        left_patients = {row["patient_id"] for row in rows_by_split[left]}
        for right in split_names[index + 1 :]:
            right_patients = {row["patient_id"] for row in rows_by_split[right]}
            overlap = sorted(left_patients & right_patients)
            if overlap:
                overlaps[f"{left}_{right}"] = overlap[:10]

    if duplicate_ids or missing or empty or overlaps:
        report.add(
            "dataset.integrity",
            "FAIL",
            f"{duplicate_ids} duplicate IDs, {len(missing)} missing, "
            f"{len(empty)} empty, {sum(map(len, overlaps.values()))} patient overlaps",
            {
                "duplicate_image_ids": duplicate_ids,
                "missing_examples": missing[:5],
                "empty_examples": empty[:5],
                "patient_overlap_examples": overlaps,
            },
        )
        return

    report.add(
        "dataset.integrity",
        "PASS",
        f"{len(all_rows)} unique images exist; "
        f"{'/'.join(manifest_paths)} patients are disjoint",
    )

    if not verify_image_headers:
        report.add(
            "dataset.image_headers",
            "SKIP",
            "disabled with --no-verify-image-headers",
        )
        return
    try:
        from PIL import Image
    except ImportError:
        report.add("dataset.image_headers", "FAIL", "Pillow is not importable")
        return
    corrupt: list[str] = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt.append(str(path))
    report.add(
        "dataset.image_headers",
        "PASS" if not corrupt else "FAIL",
        f"verified {len(image_paths)} image headers"
        if not corrupt
        else f"{len(corrupt)} corrupt/unreadable images",
        None if not corrupt else {"examples": corrupt[:5]},
    )


def safe_config_filename(info: Any, preferred: tuple[str, ...]) -> str | None:
    filenames = {sibling.rfilename for sibling in getattr(info, "siblings", ())}
    for filename in preferred:
        if filename in filenames and Path(filename).suffix.lower() not in WEIGHT_SUFFIXES:
            return filename
    return None


def check_hugging_face(
    report: Report,
    model_id: str,
    model_revision: str,
    processor_id: str,
    processor_revision: str,
    timeout: float,
) -> None:
    try:
        from huggingface_hub import HfApi, get_token, hf_hub_download
    except ImportError:
        report.add(
            "huggingface.access",
            "FAIL",
            "huggingface_hub is not installed",
        )
        return

    token = get_token()
    if token:
        try:
            identity = HfApi(token=token).whoami(token=token)
            username = identity.get("name") or identity.get("fullname") or "authenticated"
            report.add(
                "huggingface.token",
                "PASS",
                f"cached/environment token is valid for {username}",
            )
        except Exception as exc:
            report.add(
                "huggingface.token",
                "FAIL",
                f"a token was found but authentication failed: {type(exc).__name__}",
            )
    else:
        report.add(
            "huggingface.token",
            "WARN",
            "no token found; current public repositories will be tested anonymously "
            "(a token is optional unless access policy changes)",
        )

    api = HfApi(token=token)
    targets = (
        (model_id, model_revision, ("config.json",)),
        (
            processor_id,
            processor_revision,
            ("processor_config.json", "config.json"),
        ),
    )
    for repo_id, revision, preferred in targets:
        try:
            info = api.model_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=False,
                timeout=timeout,
                token=token,
            )
            if info.sha != revision:
                raise ValueError(
                    f"resolved revision {info.sha!r} does not match pin {revision!r}"
                )
            filename = safe_config_filename(info, preferred)
            if filename is None:
                raise FileNotFoundError(
                    f"none of {preferred!r} appears in repository metadata"
                )
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
                etag_timeout=timeout,
            )
            if Path(config_path).suffix.lower() in WEIGHT_SUFFIXES:
                raise RuntimeError("refusing unexpected weight artifact")
            with Path(config_path).open(encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"{filename} is not a JSON object")
            report.add(
                f"huggingface.repo.{repo_id}",
                "PASS",
                f"metadata and {filename} accessible at pinned revision {info.sha}",
                {
                    "expected_revision": revision,
                    "gated": getattr(info, "gated", None),
                    "private": getattr(info, "private", None),
                    "config_keys": sorted(value)[:20],
                    "weights_downloaded": False,
                },
            )
        except Exception as exc:
            report.add(
                f"huggingface.repo.{repo_id}",
                "FAIL",
                f"metadata/config access failed: {type(exc).__name__}: {exc}",
                {"weights_downloaded": False},
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only A100 preflight. This command never launches training or "
            "downloads model weights."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--min-vram-gib", type=float, default=48.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=150.0)
    parser.add_argument("--hf-timeout", type=float, default=15.0)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument(
        "--verify-image-headers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = Report()
    config_path = resolve_project_path(args.config)
    config = load_config(report, config_path)
    check_system(report, args.min_free_disk_gib)
    check_c_compiler(report)
    torch = check_gpu(report, args.min_vram_gib)
    check_dependencies(report, torch)
    if config is not None:
        check_dataset(report, config, args.verify_image_headers)
        if args.skip_hf:
            report.add("huggingface.access", "SKIP", "disabled with --skip-hf")
        else:
            check_hugging_face(
                report,
                config["model_id"],
                config["model_revision"],
                config["processor_id"],
                config["processor_revision"],
                args.hf_timeout,
            )

    payload = report.payload()
    if args.json_output:
        output = resolve_project_path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json_only:
        print(json.dumps(payload, indent=2))
    else:
        report.print_human()
        if args.json_output:
            print(f"JSON report: {resolve_project_path(args.json_output)}")
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
