#!/usr/bin/env python3
"""Run one retinal technical-quality assessment against local llama.cpp."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

DECISIONS = {"READY", "LIMITED", "RETAKE"}
ISSUES = {"artifact", "blur", "field_cutoff", "unsupported_modality", "uncertain"}
DISCLAIMER = "Technical image-quality assessment only; not a diagnosis."

SYSTEM_PROMPT = """You are RetinaReady, an offline technical image-quality assistant.
Assess only whether the supplied image is a color fundus photograph of sufficient
technical quality. Never diagnose disease, infer that an eye is healthy, recommend
treatment, or interpret retinal pathology. Return exactly one JSON object matching
the requested schema. Use LIMITED for an unsupported modality, an uncertain case,
or any case where you cannot assess quality safely."""

USER_PROMPT = """Assess this image for capture quality.
Return exactly:
{
  "decision": "READY" | "LIMITED" | "RETAKE",
  "confidence": number from 0 to 1 or null,
  "issues": array containing only "artifact", "blur", "field_cutoff",
            "unsupported_modality", or "uncertain",
  "scores": {
    "artifact": integer 0-100 or null,
    "clarity": integer 0-100 or null,
    "field_definition": integer 0-100 or null
  },
  "retake_instruction": string or null,
  "disclaimer": "Technical image-quality assessment only; not a diagnosis."
}
For every score, 100 means best technical quality. Do not add markdown or prose."""


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_object(text: str) -> dict:
    # Gemma 4 26B can emit an empty thought channel even with thinking disabled.
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model did not return a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not an object")
    return value


def normalize(value: dict) -> dict:
    decision = str(value.get("decision", "")).upper()
    if decision not in DECISIONS:
        raise ValueError(f"invalid decision: {decision!r}")

    issues = value.get("issues")
    if not isinstance(issues, list) or any(item not in ISSUES for item in issues):
        raise ValueError("issues must be an array of allowed issue codes")

    confidence = value.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be from 0 to 1")

    raw_scores = value.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("scores must be an object")
    scores: dict[str, int | None] = {}
    for key in ("artifact", "clarity", "field_definition"):
        score = raw_scores.get(key)
        if score is None:
            scores[key] = None
            continue
        score = int(score)
        if not 0 <= score <= 100:
            raise ValueError(f"{key} score must be from 0 to 100")
        scores[key] = score

    instruction = value.get("retake_instruction")
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError("retake_instruction must be a string or null")

    return {
        "decision": decision,
        "confidence": confidence,
        "issues": issues,
        "scores": scores,
        "retake_instruction": instruction,
        "disclaimer": DISCLAIMER,
    }


def limited_fallback(reason: str) -> dict:
    return {
        "decision": "LIMITED",
        "confidence": None,
        "issues": ["uncertain"],
        "scores": {"artifact": None, "clarity": None, "field_definition": None},
        "retake_instruction": "Assessment was uncertain; review the image or retake it.",
        "disclaimer": DISCLAIMER,
        "_validation_error": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument(
        "--model",
        default=os.getenv("RETINA_READY_MODEL_ALIAS", "retinaready-gemma4-26b"),
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url(args.image)}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }

    try:
        response = request_json(
            f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout
        )
        content = response["choices"][0]["message"]["content"]
        if args.raw:
            print(content, file=sys.stderr)
        result = normalize(extract_object(content))
    except (KeyError, TypeError, ValueError, OSError, urllib.error.URLError) as exc:
        result = limited_fallback(str(exc))

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] != "LIMITED" or "_validation_error" not in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
