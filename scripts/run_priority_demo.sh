#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_HOST="${RETINA_PRIORITY_HOST:-127.0.0.1}"
MODEL_PORT="${RETINA_PRIORITY_PORT:-8082}"
APP_HOST="${RETINA_PRIORITY_APP_HOST:-127.0.0.1}"
APP_PORT="${RETINA_PRIORITY_APP_PORT:-8000}"
MODEL_ID="${RETINA_PRIORITY_MODEL_ALIAS:-retinapriority-gemma4-26b}"
MODEL_SLEEP_IDLE_SECONDS="${RETINA_READY_SLEEP_IDLE_SECONDS:-5}"
if [[ ! "${MODEL_SLEEP_IDLE_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "RETINA_READY_SLEEP_IDLE_SECONDS must be a non-negative integer, got: ${MODEL_SLEEP_IDLE_SECONDS}" >&2
  exit 1
fi
MODEL_SLEEP_IDLE_SECONDS="$((10#${MODEL_SLEEP_IDLE_SECONDS}))"

BASE_DIR="${PROJECT_ROOT}/models/retinaready-gemma4-26b-tuned"
PRIORITY_DIR="${PROJECT_ROOT}/models/retinapriority-gemma4-26b"
SPECIALIST_DIR="${PROJECT_ROOT}/models/retinaready-quality-specialist"
MODEL_FILE="${BASE_DIR}/retinaready-gemma4-26b-a4b-q4_0.gguf"
MMPROJ_FILE="${BASE_DIR}/retinaready-gemma4-26b-a4b-mmproj-bf16.gguf"
LORA_FILE="${PRIORITY_DIR}/retinapriority-gemma4-26b-a4b-lora-f32.gguf"
LORA_SHA_FILE="${PRIORITY_DIR}/retinapriority-gemma4-26b-a4b-lora-f32.gguf.sha256"

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

for required_file in \
  "${MODEL_FILE}" \
  "${MMPROJ_FILE}" \
  "${LORA_FILE}" \
  "${LORA_SHA_FILE}" \
  "${SPECIALIST_DIR}/densenet121-a639ec97.pth" \
  "${SPECIALIST_DIR}/decision-head.pt" \
  "${SPECIALIST_DIR}/factor-head.pt"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "Required local demo artifact is missing or empty: ${required_file}" >&2
    exit 1
  fi
done

if ! python3 -c "import fastapi, numpy, PIL, torch, torchvision, uvicorn" >/dev/null 2>&1; then
  echo "Local app or specialist dependencies are missing." >&2
  echo "Run: python3 -m pip install -r ${PROJECT_ROOT}/app/requirements.txt" >&2
  exit 1
fi

EXPECTED_LORA_SHA="$(awk 'NF { print $1; exit }' "${LORA_SHA_FILE}")"
if [[ ! "${EXPECTED_LORA_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "The RetinaPriority LoRA checksum pin is invalid." >&2
  exit 1
fi
OBSERVED_LORA_SHA="$(shasum -a 256 "${LORA_FILE}" | awk '{print $1}')"
if [[ "${OBSERVED_LORA_SHA}" != "${EXPECTED_LORA_SHA}" ]]; then
  echo "The RetinaPriority LoRA checksum does not match its local pin." >&2
  exit 1
fi

verify_runtime_identity() {
  python3 - \
    "http://${MODEL_HOST}:${MODEL_PORT}" \
    "${MODEL_ID}" \
    "${LORA_FILE}" <<'PY'
import json
from pathlib import Path
import sys
from urllib.request import ProxyHandler, build_opener

base_url, expected_model, expected_lora = sys.argv[1:]
opener = build_opener(ProxyHandler({}))
with opener.open(f"{base_url}/v1/models", timeout=5) as response:
    models = json.load(response)
model_ids = [item.get("id") for item in models.get("data", [])]
if expected_model not in model_ids:
    raise SystemExit("The exact RetinaPriority model alias is not active.")
with opener.open(f"{base_url}/lora-adapters", timeout=5) as response:
    adapters = json.load(response)
expected_path = str(Path(expected_lora).resolve())
if (
    not isinstance(adapters, list)
    or len(adapters) != 1
    or adapters[0].get("path") != expected_path
    or float(adapters[0].get("scale", 0)) != 1.0
):
    raise SystemExit("The exact RetinaPriority LoRA is not active at scale 1.")
PY
}

if curl -fsS "http://${MODEL_HOST}:${MODEL_PORT}/health" >/dev/null 2>&1; then
  if ! verify_runtime_identity; then
    echo "A healthy but incompatible model server is using ${MODEL_HOST}:${MODEL_PORT}." >&2
    exit 1
  fi
  echo "Using the verified ${MODEL_ID} server on ${MODEL_HOST}:${MODEL_PORT}."
else
  RETINA_READY_MODEL_FILE="${MODEL_FILE}" \
  RETINA_READY_MMPROJ_FILE="${MMPROJ_FILE}" \
  RETINA_READY_LORA_FILE="${LORA_FILE}" \
  RETINA_READY_MODEL_ALIAS="${MODEL_ID}" \
  RETINA_READY_HOST="${MODEL_HOST}" \
  RETINA_READY_PORT="${MODEL_PORT}" \
  RETINA_READY_CORS_ORIGINS="http://${APP_HOST}:${APP_PORT}" \
  RETINA_READY_SLEEP_IDLE_SECONDS="${MODEL_SLEEP_IDLE_SECONDS}" \
    "${PROJECT_ROOT}/ml/serve_local.sh" &
  model_launcher_pid=$!
  python3 "${PROJECT_ROOT}/ml/wait_for_server.py" \
    --base-url "http://${MODEL_HOST}:${MODEL_PORT}" \
    --timeout 900 \
    --pid "${model_launcher_pid}"
  verify_runtime_identity
fi

cd "${PROJECT_ROOT}/app"
RETINA_ANALYZER=specialist \
RETINA_SPECIALIST_DIR="${SPECIALIST_DIR}" \
RETINA_SPECIALIST_DEVICE=cpu \
RETINA_ENABLE_ESCALATION_RESEARCH_DEMO=1 \
RETINA_ENABLE_VIDEO_CANDIDATE_WORKFLOW=1 \
RETINA_ESCALATION_ENGINE=gemma \
RETINA_ESCALATION_GEMMA_API_URL="http://${MODEL_HOST}:${MODEL_PORT}" \
RETINA_ESCALATION_GEMMA_MODEL_ID="${MODEL_ID}" \
RETINA_ESCALATION_GEMMA_LORA_PATH="${LORA_FILE}" \
RETINA_ESCALATION_GEMMA_LORA_SHA256="${EXPECTED_LORA_SHA}" \
RETINA_ESCALATION_GEMMA_TIMEOUT_SECONDS="${RETINA_ESCALATION_GEMMA_TIMEOUT_SECONDS:-90}" \
python3 -m uvicorn main:app --host "${APP_HOST}" --port "${APP_PORT}" &
app_pid=$!

app_is_ready() {
  curl -fsS --max-time 15 "http://${APP_HOST}:${APP_PORT}/api/health" |
    python3 -c '
import json
import sys

health = json.load(sys.stdin)
ready = (
    health.get("status") == "ready"
    and health.get("video_candidate_workflow_enabled") is True
    and isinstance(health.get("escalation"), dict)
    and health["escalation"].get("release_enabled") is True
)
raise SystemExit(0 if ready else 1)
'
}

for _ in {1..120}; do
  if app_is_ready >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${app_pid}" 2>/dev/null; then
    wait "${app_pid}"
    exit 1
  fi
  sleep 0.25
done
if ! app_is_ready >/dev/null 2>&1; then
  echo "RetinaPriority app and exact local adapter did not become healthy in time." >&2
  exit 1
fi

echo "RetinaReady + RetinaPriority is available at http://${APP_HOST}:${APP_PORT}"
echo "Quality runs first; only READY images reach the local Gemma escalation LoRA."
if [[ -z "${model_launcher_pid}" ]]; then
  echo "A verified existing Gemma server was reused; its idle-sleep policy was not changed by this launcher."
elif (( MODEL_SLEEP_IDLE_SECONDS > 0 )); then
  echo "Gemma releases its model allocation after ${MODEL_SLEEP_IDLE_SECONDS}s idle to leave headroom for browser video."
else
  echo "Gemma idle sleeping is disabled by RETINA_READY_SLEEP_IDLE_SECONDS=0."
fi
echo "Press Ctrl-C to stop the processes started by this launcher."
wait "${app_pid}"
