#!/usr/bin/env python3
"""Incrementally mirror and verify the small A100 training artifacts.

The collector deliberately uses an allowlist.  It never copies model-base
weights, Hugging Face caches, optimizer/rng/scheduler state, TensorBoard event
files, or arbitrary files from the remote host.  Files are stored under
``<output>/mirror`` using their path relative to the remote repository root.

The remote tree is hashed before and after rsync.  A collection is accepted
only when the two remote snapshots agree and a locally generated manifest is
byte-for-byte identical to that stable remote manifest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_HOST = "vast-retinapriority"
DEFAULT_REMOTE_ROOT = "/workspace/retina-ready"
DEFAULT_OUTPUT_ROOT = Path("outputs/a100-retinapriority-20260801")

SMOKE_RUN = "ml/runs/gemma4-26b-retinapriority-decision-smoke"
FULL_RUN = "ml/runs/gemma4-26b-retinapriority-quality-pass-decision-full-v1"
CHALLENGER_RUN = (
    "ml/runs/gemma4-26b-retinapriority-quality-pass-qv-challenger-v1"
)
POSTTRAIN_RUN = "ml/runs/gemma4-26b-retinapriority-cross-run-posttrain-v1"
POSTTRAIN_LOG = "ml/runs/logs/retinapriority-cross-run-posttrain-v1.log"

RUN_ROOT_EXACT = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "processor_config.json",
    "run_provenance.json",
    "tokenizer_config.json",
    "training_args.bin",
)

# checkpoint-74 was explicitly evaluated.  Root weights preserve the final
# checkpoint for both finished runs, so other checkpoint copies are redundant.
CHECKPOINT_FILES = {
    f"{FULL_RUN}/checkpoint-74": (
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
    ),
}

CONFIG_NAMES = (
    "gemma4_26b_decision_full.json",
    "gemma4_26b_decision_smoke.json",
    "gemma4_26b_escalation_full.json",
    "gemma4_26b_escalation_quality_pass_full.json",
    "gemma4_26b_escalation_quality_pass_qv_challenger.json",
    "gemma4_26b_escalation_smoke.json",
    "gemma4_26b_vram_probe.json",
)


def profile_rules(profile: str) -> dict[str, Any]:
    """Return the JSON-serializable allowlist for a collection profile."""

    if profile == "finished-v1":
        runs = (SMOKE_RUN, FULL_RUN)
        excluded_log_fragments = ("quality-pass-qv-challenger",)
        evidence_trees: tuple[dict[str, Any], ...] = ()
        required_exact_files: tuple[str, ...] = ()
        include_standard_artifacts = True
        immutable = False
    elif profile == "with-challenger-v1":
        runs = (SMOKE_RUN, FULL_RUN, CHALLENGER_RUN)
        excluded_log_fragments = ()
        evidence_trees = ()
        required_exact_files = ()
        include_standard_artifacts = True
        immutable = False
    elif profile == "cross-run-posttrain-v1":
        runs = ()
        excluded_log_fragments = ()
        include_standard_artifacts = False
        immutable = True
        required_exact_files = (POSTTRAIN_LOG,)
        evidence_trees = (
            {
                "root": POSTTRAIN_RUN,
                "completion_file": "posttrain-completion.json",
                "required_status": (
                    "completed_research_evaluation_not_runtime_promotion"
                ),
                "verify_completion_integrity": True,
                "excluded_directory_names": [
                    ".cache",
                    ".git",
                    "__pycache__",
                    "cache",
                    "checkpoints",
                    "runs",
                    "wandb",
                ],
                "excluded_file_names": [
                    "optimizer.pt",
                    "rng_state.pth",
                    "scheduler.pt",
                    "tokenizer.json",
                ],
                "excluded_suffixes": [
                    ".bin",
                    ".gguf",
                    ".pt",
                    ".pth",
                    ".safetensors",
                    ".tfevents",
                ],
            },
        )
    else:  # Defensive even though argparse also constrains this.
        raise ValueError(f"unsupported profile: {profile}")

    rules = {
        "profile": profile,
        "runs": list(runs),
        "run_root_exact": list(RUN_ROOT_EXACT),
        "run_root_json": True,
        "run_root_json_exclude": ["tokenizer.json"],
        "required_run_status": "completed",
        "checkpoint_files": {
            path: list(names)
            for path, names in CHECKPOINT_FILES.items()
            if any(path.startswith(f"{run}/") for run in runs)
        },
        "config_names": list(CONFIG_NAMES) if include_standard_artifacts else [],
        "log_root": "ml/runs/logs",
        "log_prefix": "retinapriority-",
        "log_suffix": ".log",
        "excluded_log_fragments": list(excluded_log_fragments),
        "gguf_root": "ml/gguf/retinapriority-gemma4-26b",
        "gguf_name_fragment": "lora",
        "gguf_suffix": ".gguf",
        # A LoRA is tens of MB.  This cap makes accidental base-model copying
        # impossible even if a base GGUF is placed in the LoRA directory.
        "maximum_file_bytes": 256 * 1024 * 1024,
    }
    if profile == "cross-run-posttrain-v1":
        rules.update(
            {
                "evidence_trees": list(evidence_trees),
                "required_exact_files": list(required_exact_files),
                "immutable": immutable,
                "include_standard_logs": include_standard_artifacts,
                "include_gguf": include_standard_artifacts,
            }
        )
    return rules


REMOTE_COLLECTOR = r"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

root = Path(sys.argv[1]).resolve()
rules = json.load(sys.stdin)
selected = set()

def safe_add(relative):
    rel = PurePosixPath(str(relative))
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError("unsafe relative path: %s" % rel)
    path = root.joinpath(*rel.parts)
    if path.is_symlink():
        raise RuntimeError("refusing symlink: %s" % rel)
    if not path.is_file():
        return
    if path.stat().st_size > int(rules["maximum_file_bytes"]):
        raise RuntimeError("allowlisted file exceeds size cap: %s" % rel)
    selected.add(rel.as_posix())

for run in rules["runs"]:
    provenance = root / run / "run_provenance.json"
    if not provenance.is_file() or provenance.is_symlink():
        raise RuntimeError("missing provenance for required run: %s" % run)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    if payload.get("status") != rules["required_run_status"]:
        raise RuntimeError(
            "run is not completed: %s (status=%r)" % (run, payload.get("status"))
        )
    run_dir = root / run
    for path in run_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in rules["run_root_exact"]:
            safe_add(PurePosixPath(run) / path.name)
        elif (
            rules["run_root_json"]
            and path.suffix == ".json"
            and path.name not in rules["run_root_json_exclude"]
        ):
            safe_add(PurePosixPath(run) / path.name)

for tree in rules.get("evidence_trees", []):
    tree_relative = PurePosixPath(tree["root"])
    tree_root = root.joinpath(*tree_relative.parts)
    if tree_root.is_symlink() or not tree_root.is_dir():
        raise RuntimeError("missing evidence tree: %s" % tree_relative)
    completion_relative = tree_relative / tree["completion_file"]
    completion = root.joinpath(*completion_relative.parts)
    if completion.is_symlink() or not completion.is_file():
        raise RuntimeError(
            "posttrain evidence is incomplete; missing completion report: %s"
            % completion_relative
        )
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    observed_status = completion_payload.get("status")
    if observed_status != tree["required_status"]:
        raise RuntimeError(
            "posttrain evidence is incomplete: %s (status=%r)"
            % (tree_relative, observed_status)
        )
    if tree["verify_completion_integrity"]:
        integrity = completion_payload.get("integrity")
        if not isinstance(integrity, dict):
            raise RuntimeError("completion report lacks integrity binding")
        expected = integrity.get("canonical_report_without_integrity_sha256")
        core = {
            key: value
            for key, value in completion_payload.items()
            if key != "integrity"
        }
        encoded = json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        observed = hashlib.sha256(encoded).hexdigest()
        if expected != observed:
            raise RuntimeError("completion report failed its integrity binding")

    excluded_directories = set(tree["excluded_directory_names"])
    excluded_files = set(tree["excluded_file_names"])
    excluded_suffixes = tuple(tree["excluded_suffixes"])
    for path in tree_root.rglob("*"):
        relative_to_tree = path.relative_to(tree_root)
        if path.is_symlink():
            raise RuntimeError(
                "refusing symlink inside evidence tree: %s"
                % (tree_relative / relative_to_tree)
            )
        if any(part in excluded_directories for part in relative_to_tree.parts):
            continue
        if not path.is_file():
            continue
        if path.name in excluded_files or path.name.endswith(excluded_suffixes):
            continue
        safe_add(tree_relative / relative_to_tree)

for relative in rules.get("required_exact_files", []):
    rel = PurePosixPath(relative)
    path = root.joinpath(*rel.parts)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("missing required exact artifact: %s" % rel)
    safe_add(rel)

for checkpoint, names in rules["checkpoint_files"].items():
    for name in names:
        safe_add(PurePosixPath(checkpoint) / name)

for name in rules["config_names"]:
    safe_add(PurePosixPath("ml/configs") / name)

log_root = root / rules["log_root"]
if rules.get("include_standard_logs", True) and log_root.is_dir():
    for path in log_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative_to_logs = path.relative_to(log_root)
        is_conversion_log = (
            relative_to_logs.parts
            and relative_to_logs.parts[0] == "conversion"
            and path.suffix == rules["log_suffix"]
        )
        is_training_log = (
            path.name.startswith(rules["log_prefix"])
            and path.name.endswith(rules["log_suffix"])
        )
        excluded = any(
            fragment in path.name for fragment in rules["excluded_log_fragments"]
        )
        is_dedicated_posttrain_log = (
            path.name == "retinapriority-cross-run-posttrain-v1.log"
        )
        if (
            (is_conversion_log or is_training_log)
            and not excluded
            and not is_dedicated_posttrain_log
        ):
            safe_add(PurePosixPath(rules["log_root"]) / relative_to_logs)

gguf_root = root / rules["gguf_root"]
if rules.get("include_gguf", True) and gguf_root.is_dir():
    for path in gguf_root.rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and rules["gguf_name_fragment"] in path.name.lower()
            and path.name.endswith(rules["gguf_suffix"])
        ):
            safe_add(
                PurePosixPath(rules["gguf_root"]) / path.relative_to(gguf_root)
            )

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

files = []
for relative in sorted(selected):
    path = root.joinpath(*PurePosixPath(relative).parts)
    files.append(
        {
            "path": relative,
            "sha256": digest(path),
            "size_bytes": path.stat().st_size,
        }
    )

rules_bytes = json.dumps(
    rules, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
manifest = {
    "schema_version": 1,
    "profile": rules["profile"],
    "selection_rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
    "files": files,
}
print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
"""


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"remote returned unsafe path: {value!r}")
    return path


def validate_manifest(manifest: Any, rules: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise RuntimeError("remote manifest is not an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("remote manifest has an unsupported schema")
    if manifest.get("profile") != rules["profile"]:
        raise RuntimeError("remote manifest profile does not match request")

    expected_rules_sha = sha256_bytes(
        json.dumps(
            rules, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )
    if manifest.get("selection_rules_sha256") != expected_rules_sha:
        raise RuntimeError("remote manifest rules hash does not match request")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("remote manifest contains no files")
    previous = ""
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise RuntimeError("remote manifest has a malformed file entry")
        relative = validate_relative_path(str(entry.get("path", ""))).as_posix()
        if relative in seen or (previous and relative <= previous):
            raise RuntimeError("remote manifest paths are not unique and sorted")
        seen.add(relative)
        previous = relative
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RuntimeError(f"invalid SHA-256 for {relative}")
        size = entry.get("size_bytes")
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"invalid size for {relative}")
        if size > int(rules["maximum_file_bytes"]):
            raise RuntimeError(f"remote file exceeds local size cap: {relative}")
    return manifest


def parse_remote_json(stdout: str) -> Any:
    """Parse the final JSON line, tolerating provider login banners."""

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("remote collector returned no JSON manifest")


def remote_manifest(
    host: str, remote_root: str, rules: dict[str, Any]
) -> dict[str, Any]:
    encoded_source = base64.b64encode(REMOTE_COLLECTOR.encode("utf-8")).decode(
        "ascii"
    )
    launcher = (
        "import base64;"
        f"exec(base64.b64decode({encoded_source!r}).decode('utf-8'))"
    )
    command = " ".join(
        (
            "python3",
            "-c",
            shlex.quote(launcher),
            shlex.quote(remote_root),
        )
    )
    result = subprocess.run(
        ["ssh", host, command],
        input=json.dumps(rules, sort_keys=True),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"remote snapshot failed: {detail}")
    return validate_manifest(parse_remote_json(result.stdout), rules)


def rsync_manifest_files(
    host: str,
    remote_root: str,
    mirror_root: Path,
    manifest: dict[str, Any],
) -> None:
    mirror_root.mkdir(parents=True, exist_ok=True)
    paths = [entry["path"] for entry in manifest["files"]]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        for relative in paths:
            validate_relative_path(relative)
            handle.write(relative)
            handle.write("\n")
        file_list = Path(handle.name)
    try:
        result = subprocess.run(
            [
                "rsync",
                "--archive",
                "--checksum",
                "--relative",
                f"--files-from={file_list}",
                f"{host}:{remote_root.rstrip('/')}/",
                f"{mirror_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        file_list.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"rsync failed: {detail}")


def local_manifest(
    mirror_root: Path,
    remote: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for remote_entry in remote["files"]:
        relative = validate_relative_path(remote_entry["path"])
        local_path = mirror_root.joinpath(*relative.parts)
        if not local_path.is_file() or local_path.is_symlink():
            missing.append(relative.as_posix())
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(local_path),
                "size_bytes": local_path.stat().st_size,
            }
        )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "profile": remote["profile"],
            "selection_rules_sha256": remote["selection_rules_sha256"],
            "files": files,
        },
        missing,
    )


def comparison(
    remote: dict[str, Any], local: dict[str, Any], missing: Iterable[str]
) -> dict[str, Any]:
    remote_by_path = {entry["path"]: entry for entry in remote["files"]}
    local_by_path = {entry["path"]: entry for entry in local["files"]}
    hash_mismatches = sorted(
        path
        for path in remote_by_path.keys() & local_by_path.keys()
        if remote_by_path[path]["sha256"] != local_by_path[path]["sha256"]
    )
    size_mismatches = sorted(
        path
        for path in remote_by_path.keys() & local_by_path.keys()
        if remote_by_path[path]["size_bytes"] != local_by_path[path]["size_bytes"]
    )
    unexpected = sorted(local_by_path.keys() - remote_by_path.keys())
    missing_sorted = sorted(set(missing) | (remote_by_path.keys() - local_by_path.keys()))
    remote_bytes = canonical_bytes(remote)
    local_bytes = canonical_bytes(local)
    matched = not (missing_sorted or unexpected or hash_mismatches or size_mismatches)
    matched = matched and remote_bytes == local_bytes
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": remote["profile"],
        "matched": matched,
        "file_count": len(remote_by_path),
        "total_size_bytes": sum(
            int(entry["size_bytes"]) for entry in remote_by_path.values()
        ),
        "remote_manifest_sha256": sha256_bytes(remote_bytes),
        "local_manifest_sha256": sha256_bytes(local_bytes),
        "missing": missing_sorted,
        "unexpected": unexpected,
        "hash_mismatches": hash_mismatches,
        "size_mismatches": size_mismatches,
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_evidence(
    integrity_root: Path,
    profile: str,
    remote: dict[str, Any],
    local: dict[str, Any],
    result: dict[str, Any],
) -> None:
    atomic_write(
        integrity_root / f"{profile}.remote-manifest.json", canonical_bytes(remote)
    )
    atomic_write(
        integrity_root / f"{profile}.local-manifest.json", canonical_bytes(local)
    )
    atomic_write(
        integrity_root / f"{profile}.comparison.json", canonical_bytes(result)
    )


def enforce_immutable_snapshot(
    integrity_root: Path,
    profile: str,
    remote: dict[str, Any],
    immutable: bool,
) -> None:
    """Refuse to repoint an already-recorded immutable profile."""

    if not immutable:
        return
    recorded_path = integrity_root / f"{profile}.remote-manifest.json"
    if not recorded_path.exists():
        return
    if recorded_path.is_symlink() or not recorded_path.is_file():
        raise RuntimeError(
            f"immutable manifest path is not a regular file: {recorded_path}"
        )
    try:
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot validate existing immutable manifest {recorded_path}: {exc}"
        ) from exc
    if canonical_bytes(recorded) != canonical_bytes(remote):
        raise RuntimeError(
            f"immutable profile {profile!r} differs from its recorded remote manifest"
        )


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rules = profile_rules(args.profile)
    output_root = args.output_root.resolve()
    mirror_root = output_root / "mirror"
    integrity_root = output_root / "integrity"

    if args.verify_only:
        stable_remote = remote_manifest(args.ssh_host, args.remote_root, rules)
        enforce_immutable_snapshot(
            integrity_root,
            args.profile,
            stable_remote,
            bool(rules.get("immutable", False)),
        )
        local, missing = local_manifest(mirror_root, stable_remote)
        result = comparison(stable_remote, local, missing)
        write_evidence(integrity_root, args.profile, stable_remote, local, result)
        return result

    for attempt in range(1, args.max_attempts + 1):
        before = remote_manifest(args.ssh_host, args.remote_root, rules)
        enforce_immutable_snapshot(
            integrity_root, args.profile, before, bool(rules.get("immutable", False))
        )
        rsync_manifest_files(
            args.ssh_host, args.remote_root, mirror_root, before
        )
        after = remote_manifest(args.ssh_host, args.remote_root, rules)
        enforce_immutable_snapshot(
            integrity_root, args.profile, after, bool(rules.get("immutable", False))
        )
        if canonical_bytes(before) != canonical_bytes(after):
            if attempt == args.max_attempts:
                raise RuntimeError(
                    "remote allowlisted artifacts changed during every collection attempt"
                )
            continue
        local, missing = local_manifest(mirror_root, after)
        result = comparison(after, local, missing)
        write_evidence(integrity_root, args.profile, after, local, result)
        if result["matched"]:
            return result
        if attempt == args.max_attempts:
            return result
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--profile",
        choices=(
            "finished-v1",
            "with-challenger-v1",
            "cross-run-posttrain-v1",
        ),
        default="finished-v1",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    try:
        result = collect(args)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"artifact collection failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
