#!/usr/bin/env python3
"""Fail-closed RETFound-Green licence and frozen-input audit.

This script intentionally does not download model weights or train a model.
RETFound-Green's current licence withdraws permission from an
``Industry-Involved Project`` absent prior written permission.  The Gemma NYC
event is prize-funded by a commercial entity, so the checked-in project context
must remain blocked unless documented written permission is supplied.

The audit also pins the exact escalation manifests and DenseNet control report
that a future, permitted experiment would have to use.  This is an engineering
compatibility check, not legal advice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import ssl
import sys
from typing import Any
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("ml/configs/retfound_green_license_audit.json")
DEFAULT_REPORT = Path("outputs/retfound-green-escalation/license-blocker.json")


def resolve(path: Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else project_root / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(payload: bytes) -> str:
    return re.sub(r"\s+", " ", payload.decode("utf-8")).strip()


def fetch_license(url: str, *, timeout: float) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "RetinaReady-RETFound-Green-license-audit/1.0"},
    )
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urlopen(  # noqa: S310 - pinned HTTPS URL and verified TLS context
        request, timeout=timeout, context=context
    ) as response:
        return response.read()


def verify_license(payload: bytes, upstream: dict[str, Any]) -> dict[str, Any]:
    observed_sha256 = sha256_bytes(payload)
    expected_sha256 = upstream["license_sha256"]
    normalized = normalized_text(payload)
    missing_signatures = [
        signature
        for signature in upstream["required_license_signatures"]
        if signature not in normalized
    ]
    return {
        "passed": observed_sha256 == expected_sha256 and not missing_signatures,
        "source_revision": upstream["repository_revision"],
        "source_url": upstream["license_raw_url"],
        "license_name": upstream["license_name"],
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "required_signatures": upstream["required_license_signatures"],
        "missing_signatures": missing_signatures,
    }


def verify_frozen_file(
    item: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    path = resolve(Path(item["path"]), project_root=project_root)
    if not path.is_file():
        return {
            "path": item["path"],
            "exists": False,
            "expected_sha256": item["sha256"],
            "observed_sha256": None,
            "passed": False,
        }
    observed_sha256 = sha256_file(path)
    return {
        "path": item["path"],
        "exists": True,
        "expected_sha256": item["sha256"],
        "observed_sha256": observed_sha256,
        "passed": observed_sha256 == item["sha256"],
    }


def verify_frozen_inputs(
    frozen_inputs: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    manifests = {
        name: verify_frozen_file(item, project_root=project_root)
        for name, item in frozen_inputs["manifests"].items()
    }
    control = verify_frozen_file(
        frozen_inputs["control_report"], project_root=project_root
    )
    return {
        "passed": all(item["passed"] for item in manifests.values())
        and control["passed"],
        "manifests": manifests,
        "control_report": control,
    }


def verify_permission(
    permission: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    if not permission.get("obtained", False):
        return {
            "obtained": False,
            "evidence_present": False,
            "evidence_path": None,
            "evidence_sha256": None,
            "passed": False,
        }
    raw_path = permission.get("evidence_path")
    expected_sha256 = permission.get("evidence_sha256")
    if not raw_path or not expected_sha256:
        return {
            "obtained": True,
            "evidence_present": False,
            "evidence_path": raw_path,
            "evidence_sha256": None,
            "passed": False,
        }
    path = resolve(Path(raw_path), project_root=project_root)
    if not path.is_file():
        return {
            "obtained": True,
            "evidence_present": False,
            "evidence_path": raw_path,
            "evidence_sha256": None,
            "passed": False,
        }
    observed_sha256 = sha256_file(path)
    return {
        "obtained": True,
        "evidence_present": True,
        "evidence_path": raw_path,
        "expected_sha256": expected_sha256,
        "evidence_sha256": observed_sha256,
        "passed": observed_sha256 == expected_sha256,
    }


def build_report(
    config: dict[str, Any],
    *,
    license_payload: bytes,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    license_audit = verify_license(license_payload, config["upstream"])
    frozen_audit = verify_frozen_inputs(
        config["frozen_inputs"], project_root=project_root
    )
    permission_audit = verify_permission(
        config["project_context"]["prior_written_permission"],
        project_root=project_root,
    )
    industry_involved = bool(
        config["project_context"]["commercial_entity_involvement"]
    )

    audit_passed = license_audit["passed"] and frozen_audit["passed"]
    if not audit_passed:
        status = "AUDIT_FAILED_FAIL_CLOSED"
        permitted = False
        reason = "Licence or frozen-input verification failed; do not download or train."
    elif industry_involved and not permission_audit["passed"]:
        status = "BLOCKED_LICENSE_INCOMPATIBLE"
        permitted = False
        reason = (
            "The current project is industry-involved under the upstream licence "
            "and no verifiable prior written permission is present."
        )
    else:
        status = "READY_FOR_EXPERIMENT_SETUP"
        permitted = True
        reason = (
            "The pinned inputs passed and the current project context does not "
            "trigger an unresolved upstream licence bar."
        )

    weight_path = resolve(
        Path(config["local_weight_path"]), project_root=project_root
    )
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": status,
        "permitted_to_download_or_train": permitted,
        "reason": reason,
        "engineering_note": "Compatibility audit only; not legal advice.",
        "upstream": {
            "project_name": config["upstream"]["project_name"],
            "project_repository": config["upstream"]["project_repository"],
            "repository_revision": config["upstream"]["repository_revision"],
            "paper_url": config["upstream"]["paper_url"],
            "weights_release": config["upstream"]["weights_release"],
            "weights_url": config["upstream"]["weights_url"],
            "architecture": config["upstream"]["architecture"],
            "normalization": config["upstream"]["normalization"],
        },
        "license_audit": license_audit,
        "project_context": {
            "event_name": config["project_context"]["event_name"],
            "event_url": config["project_context"]["event_url"],
            "commercial_entity_involvement": industry_involved,
            "evidence": config["project_context"]["evidence"],
            "prior_written_permission": permission_audit,
        },
        "frozen_input_audit": frozen_audit,
        "execution": {
            "weight_download_attempted_by_this_audit": False,
            "training_started_by_this_audit": False,
            "runtime_or_ui_modified_by_this_audit": False,
            "configured_local_weight_path": config["local_weight_path"],
            "configured_local_weight_present": weight_path.is_file(),
        },
        "unblock_condition": (
            "Obtain prior written permission from the RETFound-Green licensor "
            "that explicitly covers this Google-funded public hackathon submission, "
            "record the permission artifact and SHA-256 in the audit config, then rerun."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--license-file",
        type=Path,
        help="Optional offline copy of the pinned upstream LICENSE for verification.",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve(args.config)
    report_path = resolve(args.json_output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        if args.license_file:
            license_payload = resolve(args.license_file).read_bytes()
        else:
            license_payload = fetch_license(
                config["upstream"]["license_raw_url"], timeout=args.timeout
            )
        report = build_report(config, license_payload=license_payload)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "experiment": config.get("experiment"),
            "status": "AUDIT_FAILED_FAIL_CLOSED",
            "permitted_to_download_or_train": False,
            "reason": f"Audit could not complete: {type(exc).__name__}: {exc}",
            "execution": {
                "weight_download_attempted_by_this_audit": False,
                "training_started_by_this_audit": False,
                "runtime_or_ui_modified_by_this_audit": False,
            },
        }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    if report["status"] == "READY_FOR_EXPERIMENT_SETUP":
        return 0
    if report["status"] == "BLOCKED_LICENSE_INCOMPATIBLE":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
