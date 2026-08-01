#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_HOST="${RETINA_READY_HOST:-127.0.0.1}"
MODEL_PORT="${RETINA_READY_PORT:-8081}"
APP_HOST="${RETINA_READY_APP_HOST:-127.0.0.1}"
APP_PORT="${RETINA_READY_APP_PORT:-8000}"
MODEL_ID="${RETINA_READY_MODEL_ALIAS:-retinaready-gemma4-26b}"
TUNED_MODEL_DIR="${PROJECT_ROOT}/models/retinaready-gemma4-26b-tuned"
SPECIALIST_DIR="${PROJECT_ROOT}/models/retinaready-quality-specialist"

if [[ ( -s "${TUNED_MODEL_DIR}/retinaready-gemma4-26b-a4b-q4_0.gguf" ||
        -s "${TUNED_MODEL_DIR}/retinaready-gemma4-26b-a4b-q4_k_m.gguf" ) &&
      -s "${TUNED_MODEL_DIR}/retinaready-gemma4-26b-a4b-mmproj-bf16.gguf" &&
      -s "${TUNED_MODEL_DIR}/retinaready-gemma4-26b-a4b-retina-decision-lora-f32.gguf" ]]; then
  MODEL_PROFILE="tuned-lora"
  MODEL_LABEL="Gemma 4 26B · Tuned LoRA · Local"
else
  MODEL_PROFILE="untuned-qat-fallback"
  MODEL_LABEL="Gemma 4 26B · Untuned QAT fallback · Local"
fi

if [[ -s "${SPECIALIST_DIR}/densenet121-a639ec97.pth" &&
      -s "${SPECIALIST_DIR}/decision-head.pt" &&
      -s "${SPECIALIST_DIR}/factor-head.pt" ]] &&
   python3 -c "import numpy, PIL, torch, torchvision" >/dev/null 2>&1; then
  HYBRID_ENABLED=1
else
  HYBRID_ENABLED=0
fi

model_launcher_pid=""
app_pid=""

cleanup() {
  if [[ -n "${app_pid}" ]]; then
    kill "${app_pid}" 2>/dev/null || true
  fi
  if [[ -n "${model_launcher_pid}" ]]; then
    kill "${model_launcher_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if ! python3 -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "App dependencies are missing." >&2
  echo "Run: python3 -m pip install -r ${PROJECT_ROOT}/app/requirements.txt" >&2
  exit 1
fi

if curl -fsS "http://${MODEL_HOST}:${MODEL_PORT}/health" >/dev/null 2>&1; then
  if ! python3 -c 'import json, sys, urllib.request; payload = json.load(urllib.request.urlopen(sys.argv[1], timeout=5)); ids = {item.get("id") for item in payload.get("data", [])}; raise SystemExit(0 if sys.argv[2] in ids else 1)' \
    "http://${MODEL_HOST}:${MODEL_PORT}/v1/models" "${MODEL_ID}"; then
    echo "A healthy but different model server is already using ${MODEL_HOST}:${MODEL_PORT}." >&2
    echo "Stop it or choose another RETINA_READY_PORT; refusing to silently reuse the wrong model." >&2
    exit 1
  fi
  echo "Using the verified ${MODEL_ID} server on ${MODEL_HOST}:${MODEL_PORT}."
else
  RETINA_READY_CORS_ORIGINS="${RETINA_READY_CORS_ORIGINS:-http://${APP_HOST}:${APP_PORT}}" \
    "${PROJECT_ROOT}/ml/serve_local.sh" &
  model_launcher_pid=$!
  python3 "${PROJECT_ROOT}/ml/wait_for_server.py" \
    --base-url "http://${MODEL_HOST}:${MODEL_PORT}" \
    --timeout 900 \
    --pid "${model_launcher_pid}"
fi

cd "${PROJECT_ROOT}/app"
RETINA_ANALYZER=local \
GEMMA_API_URL="http://${MODEL_HOST}:${MODEL_PORT}" \
MODEL_ID="${MODEL_ID}" \
RETINA_MODEL_PROFILE="${RETINA_MODEL_PROFILE:-${MODEL_PROFILE}}" \
RETINA_MODEL_LABEL="${RETINA_MODEL_LABEL:-${MODEL_LABEL}}" \
RETINA_HYBRID="${RETINA_HYBRID:-${HYBRID_ENABLED}}" \
RETINA_SPECIALIST_DIR="${RETINA_SPECIALIST_DIR:-${SPECIALIST_DIR}}" \
RETINA_SPECIALIST_DEVICE="${RETINA_SPECIALIST_DEVICE:-cpu}" \
GEMMA_TIMEOUT_SECONDS="${GEMMA_TIMEOUT_SECONDS:-30}" \
python3 -m uvicorn main:app --host "${APP_HOST}" --port "${APP_PORT}" &
app_pid=$!

echo "RetinaReady is available at http://${APP_HOST}:${APP_PORT}"
echo "Press Ctrl-C to stop the app and the model process started by this launcher."
wait "${app_pid}"
