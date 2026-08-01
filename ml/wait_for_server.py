#!/usr/bin/env python3
"""Wait for a local llama.cpp server and fail if its process exits."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    url = f"{args.base_url.rstrip('/')}/health"
    last_error = "server has not responded"
    while time.monotonic() < deadline:
        if args.pid and not process_is_alive(args.pid):
            raise SystemExit("llama-server exited before becoming ready")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and body.get("status") == "ok":
                    return 0
                last_error = f"health response: HTTP {response.status} {body!r}"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise SystemExit(f"server did not become ready within {args.timeout:g}s: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
