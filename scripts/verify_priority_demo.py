#!/usr/bin/env python3
"""Exercise the four fixed quality-first demo paths through the live API."""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener


EXPECTED = {
    "ROUTINE": {
        "display": "ROUTINE_REVIEW",
        "quality": "READY",
        "priority": "ROUTINE_REVIEW",
        "executed": True,
        "release": True,
        "states": ["COMPLETED", "COMPLETED", "RELEASED"],
    },
    "READY": {
        "display": "PRIORITY_REVIEW",
        "quality": "READY",
        "priority": "PRIORITY_REVIEW",
        "executed": True,
        "release": True,
        "states": ["COMPLETED", "COMPLETED", "RELEASED"],
    },
    "LIMITED": {
        "display": "LIMITED",
        "quality": "LIMITED",
        "priority": "UNCERTAIN",
        "executed": False,
        "release": False,
        "states": ["COMPLETED", "BLOCKED", "ABSTAINED"],
    },
    "RETAKE": {
        "display": "RETAKE",
        "quality": "RETAKE",
        "priority": "UNCERTAIN",
        "executed": False,
        "release": False,
        "states": ["COMPLETED", "BLOCKED", "ABSTAINED"],
    },
}


def fetch_json(opener: Any, request: Request, *, timeout: float) -> Any:
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    if base_url != "http://127.0.0.1:8000" and not base_url.startswith(
        "http://127.0.0.1:"
    ):
        raise SystemExit("Acceptance is restricted to a numeric loopback URL.")

    opener = build_opener(ProxyHandler({}))
    health = fetch_json(
        opener, Request(f"{base_url}/api/health"), timeout=args.timeout
    )
    escalation = health.get("escalation", {})
    required_health = {
        "status": "ready",
        "mode": "specialist-local",
        "model_verified": True,
        "specialist_verified": True,
        "privacy": "local-only",
        "network_required": False,
    }
    for key, expected in required_health.items():
        if health.get(key) != expected:
            raise SystemExit(f"Health mismatch for {key}: {health.get(key)!r}")
    required_escalation = {
        "status": "ready",
        "profile": "gemma-lora-free-generation-uncalibrated-experimental",
        "model_verified": True,
        "lora_verified": True,
        "adapter_hash_verified": True,
        "release_enabled": True,
        "input_scope": "fixed-deepdrid-quality-pass-demo-samples",
        "clinical_use": False,
    }
    for key, expected in required_escalation.items():
        if escalation.get(key) != expected:
            raise SystemExit(
                f"Escalation health mismatch for {key}: {escalation.get(key)!r}"
            )

    summary: list[dict[str, object]] = []
    for scenario, expected in EXPECTED.items():
        with opener.open(
            f"{base_url}/api/demo-samples/{scenario}", timeout=args.timeout
        ) as response:
            image_bytes = response.read()
        request = Request(
            f"{base_url}/api/workflow",
            data=image_bytes,
            headers={
                "Content-Type": "image/jpeg",
                "X-Filename": f"{scenario.lower()}.jpg",
                "X-Demo-Scenario": scenario,
                "X-Product-Mode": "COMBINED",
            },
            method="POST",
        )
        payload = fetch_json(opener, request, timeout=args.timeout)
        observed = {
            "display": payload["display"]["status"],
            "quality": payload["quality_assessment"]["status"],
            "priority": payload["escalation_assessment"]["decision"],
            "executed": payload["escalation_assessment"]["executed"],
            "release": payload["escalation_assessment"]["release_allowed"],
            "states": [item["state"] for item in payload["workflow_trace"]],
        }
        if observed != expected:
            raise SystemExit(
                f"{scenario} mismatch:\nexpected={expected}\nobserved={observed}"
            )
        summary.append(
            {
                "scenario": scenario,
                **observed,
                "latency_ms": payload["display"].get("meta", {}).get("latency_ms"),
            }
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "health": {
                    "quality": health["profile"],
                    "escalation": escalation["profile"],
                    "local_only": health["privacy"] == "local-only",
                },
                "cases": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
