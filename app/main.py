"""FastAPI shell for the fully local RetinaReady demo."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from analyzer import AnalyzerError, SPECIALIST_DEMO_IMAGE_SHA256, build_analyzer
from workflow import (
    ProductMode,
    WorkflowOrchestrator,
    WorkflowResponse,
    build_escalation_adapter,
)


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
PROJECT_ROOT = APP_DIR.parent
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ACCEPTED_IMAGE_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
analysis_engine = build_analyzer()
escalation_engine = build_escalation_adapter()

DATASET_DEMO_SAMPLES = {
    "ROUTINE": PROJECT_ROOT
    / "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-training/Images/146/146_l2.jpg",
    "READY": PROJECT_ROOT
    / "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-validation/Images/296/296_l2.jpg",
    "LIMITED": PROJECT_ROOT
    / "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-validation/Images/265/265_l2.jpg",
    "RETAKE": PROJECT_ROOT
    / "data/raw/deepdrid-v1.1/regular_fundus_images/regular-fundus-validation/Images/431/431_l2.jpg",
}


def _dataset_demo_samples_available() -> bool:
    if not all(path.is_file() for path in DATASET_DEMO_SAMPLES.values()):
        return False
    if analysis_engine.mode != "specialist-local":
        return True
    try:
        observed = frozenset(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in DATASET_DEMO_SAMPLES.values()
        )
    except OSError:
        return False
    return observed == SPECIALIST_DEMO_IMAGE_SHA256

app = FastAPI(
    title="RetinaReady",
    description="Offline retinal capture-quality and review-priority research demo.",
    version="0.1.0",
)


@app.get("/api/health")
def health() -> dict[str, object]:
    """Describe and, for local inference, verify the configured runtime."""

    runtime = analysis_engine.runtime_status()
    escalation_runtime = escalation_engine.runtime_status()
    return {
        "status": runtime["status"],
        "mode": analysis_engine.mode,
        "profile": runtime["profile"],
        "model_verified": runtime["model_verified"],
        "lora_verified": runtime.get("lora_verified", False),
        "specialist_verified": runtime.get("specialist_verified", False),
        "input_scope": runtime.get("input_scope"),
        "privacy": "local-only",
        "model": analysis_engine.model_label,
        "network_required": False,
        "product_modes": [mode.value for mode in ProductMode],
        "escalation": escalation_runtime,
        "dataset_samples_available": _dataset_demo_samples_available(),
    }


@app.get("/api/demo-samples/{scenario}", include_in_schema=False)
def dataset_demo_sample(scenario: str) -> FileResponse:
    """Serve four fixed DeepDRiD dataset examples to the local demo only."""

    path = DATASET_DEMO_SAMPLES.get(scenario.upper())
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset demo sample unavailable.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=f"deepdrid-{scenario.lower()}-sample.jpg",
    )


@app.post("/api/analyze")
async def analyze(
    request: Request,
    x_filename: str = Header(default="retinal-capture.jpg"),
    x_demo_scenario: str | None = Header(default=None),
) -> dict[str, object]:
    """Accept raw image bytes and return a local quality assessment.

    Raw request bodies keep the shell dependency-light. Multipart requests are
    rejected explicitly so their envelope is never mistaken for image bytes.
    The configured local Gemma runtime or deterministic presentation engine
    sits behind the same browser contract.
    """

    started = perf_counter()
    request_content_type = request.headers.get(
        "content-type", "application/octet-stream"
    )
    content_type = request_content_type.partition(";")[0].strip().lower()
    if content_type == "multipart/form-data":
        raise HTTPException(
            status_code=415,
            detail=(
                "Multipart uploads are not supported. Send the image as the raw "
                "request body with its image Content-Type."
            ),
        )
    if content_type not in ACCEPTED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Choose a JPEG, PNG, or WEBP image.",
        )
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 16 MB or smaller.")

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Choose an image before analyzing.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 16 MB or smaller.")

    filename = unquote(x_filename)[:240]
    try:
        result = await analysis_engine.analyze(
            image_bytes,
            filename=filename,
            content_type=content_type,
            scenario=x_demo_scenario,
        )
    except AnalyzerError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    elapsed_ms = round((perf_counter() - started) * 1000, 1)
    result["meta"] = {
        "latency_ms": elapsed_ms,
        "latency_label": (
            "Local demo pipeline"
            if analysis_engine.mode == "demo"
            else (
                "Local Gemma + frozen quality specialist"
                if analysis_engine.mode == "hybrid-local"
                else (
                    "Local frozen quality specialist"
                    if analysis_engine.mode == "specialist-local"
                    else "Local Gemma inference"
                )
            )
        ),
        "model": analysis_engine.model_label,
        "privacy": "Processed in memory on this device",
        "retained": False,
    }
    return result


@app.post("/api/workflow", response_model=WorkflowResponse)
async def workflow_analyze(
    request: Request,
    x_product_mode: ProductMode = Header(default=ProductMode.QUALITY_ONLY),
    x_filename: str = Header(default="retinal-capture.jpg"),
    x_demo_scenario: str | None = Header(default=None),
) -> WorkflowResponse:
    """Run an explicit product mode without changing the quality-only API.

    Combined mode always executes the existing quality gate first. Only a
    READY result is allowed to reach the review-priority adapter. The local
    research-demo adapter is disabled unless its explicit opt-in and exact-hash
    promotion checks pass; otherwise it returns a typed UNCERTAIN result and
    the safety policy releases no priority decision.
    """

    started = perf_counter()
    request_content_type = request.headers.get(
        "content-type", "application/octet-stream"
    )
    content_type = request_content_type.partition(";")[0].strip().lower()
    if content_type == "multipart/form-data":
        raise HTTPException(
            status_code=415,
            detail=(
                "Multipart uploads are not supported. Send the image as the raw "
                "request body with its image Content-Type."
            ),
        )
    if content_type not in ACCEPTED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Choose a JPEG, PNG, or WEBP image.",
        )
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 16 MB or smaller.")

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Choose an image before analyzing.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 16 MB or smaller.")

    filename = unquote(x_filename)[:240]
    orchestrator = WorkflowOrchestrator(
        quality=analysis_engine,
        escalation=escalation_engine,
    )
    result = await orchestrator.run(
        x_product_mode,
        image_bytes,
        filename=filename,
        content_type=content_type,
        scenario=x_demo_scenario,
    )
    elapsed_ms = round((perf_counter() - started) * 1000, 1)
    result.display["meta"] = {
        "latency_ms": elapsed_ms,
        "latency_label": "Local fail-closed workflow",
        "model": (
            analysis_engine.model_label
            if x_product_mode is ProductMode.QUALITY_ONLY
            else (
                escalation_engine.model_label
                if x_product_mode is ProductMode.ESCALATION_ONLY
                else f"{analysis_engine.model_label} + {escalation_engine.model_label}"
            )
        ),
        "privacy": "Processed in memory on this device",
        "retained": False,
    }
    return result


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
