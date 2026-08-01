#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TUNED_MODEL_DIR="${PROJECT_ROOT}/models/retinaready-gemma4-26b-tuned"
FALLBACK_MODEL_DIR="${PROJECT_ROOT}/models/gemma-4-26b-q4"
TUNED_Q4_0_NAME="retinaready-gemma4-26b-a4b-q4_0.gguf"
TUNED_Q4_K_M_NAME="retinaready-gemma4-26b-a4b-q4_k_m.gguf"
TUNED_MMPROJ_NAME="retinaready-gemma4-26b-a4b-mmproj-bf16.gguf"
TUNED_LORA_NAME="retinaready-gemma4-26b-a4b-retina-decision-lora-f32.gguf"

if [[ -n "${RETINA_READY_MODEL_DIR:-}" ]]; then
  MODEL_DIR="${RETINA_READY_MODEL_DIR}"
elif [[ ( -s "${TUNED_MODEL_DIR}/${TUNED_Q4_0_NAME}" || -s "${TUNED_MODEL_DIR}/${TUNED_Q4_K_M_NAME}" ) &&
        -s "${TUNED_MODEL_DIR}/${TUNED_MMPROJ_NAME}" &&
        -s "${TUNED_MODEL_DIR}/${TUNED_LORA_NAME}" ]]; then
  MODEL_DIR="${TUNED_MODEL_DIR}"
else
  MODEL_DIR="${FALLBACK_MODEL_DIR}"
fi

if [[ ( -s "${MODEL_DIR}/${TUNED_Q4_0_NAME}" || -s "${MODEL_DIR}/${TUNED_Q4_K_M_NAME}" ) &&
      -s "${MODEL_DIR}/${TUNED_MMPROJ_NAME}" &&
      -s "${MODEL_DIR}/${TUNED_LORA_NAME}" ]]; then
  MODEL_PROFILE="tuned"
  if [[ -s "${MODEL_DIR}/${TUNED_Q4_0_NAME}" ]]; then
    DEFAULT_MODEL_FILE="${MODEL_DIR}/${TUNED_Q4_0_NAME}"
  else
    DEFAULT_MODEL_FILE="${MODEL_DIR}/${TUNED_Q4_K_M_NAME}"
  fi
  DEFAULT_MMPROJ_FILE="${MODEL_DIR}/${TUNED_MMPROJ_NAME}"
  DEFAULT_LORA_FILE="${MODEL_DIR}/${TUNED_LORA_NAME}"
else
  MODEL_PROFILE="qat-fallback"
  DEFAULT_MODEL_FILE="${MODEL_DIR}/gemma-4-26B_q4_0-it.gguf"
  DEFAULT_MMPROJ_FILE="${MODEL_DIR}/gemma-4-26B-it-mmproj.gguf"
  DEFAULT_LORA_FILE=""
fi

MODEL_FILE="${RETINA_READY_MODEL_FILE:-${DEFAULT_MODEL_FILE}}"
MMPROJ_FILE="${RETINA_READY_MMPROJ_FILE:-${DEFAULT_MMPROJ_FILE}}"
LORA_FILE="${RETINA_READY_LORA_FILE:-${DEFAULT_LORA_FILE}}"
MODEL_ALIAS="${RETINA_READY_MODEL_ALIAS:-retinaready-gemma4-26b}"
if [[ -n "${LORA_FILE}" && "${MODEL_PROFILE}" != "tuned" ]]; then
  MODEL_PROFILE="tuned-custom-paths"
fi

HOST="${RETINA_READY_HOST:-127.0.0.1}"
PORT="${RETINA_READY_PORT:-8081}"
CTX_SIZE="${RETINA_READY_CTX_SIZE:-2048}"
BATCH_SIZE="${RETINA_READY_BATCH_SIZE:-512}"
UBATCH_SIZE="${RETINA_READY_UBATCH_SIZE:-512}"
MTMD_BATCH_MAX_TOKENS="${RETINA_READY_MTMD_BATCH_MAX_TOKENS:-512}"
GPU_LAYERS="${RETINA_READY_GPU_LAYERS:-999}"
if [[ -n "${RETINA_READY_MMPROJ_OFFLOAD:-}" ]]; then
  MMPROJ_OFFLOAD="${RETINA_READY_MMPROJ_OFFLOAD}"
elif [[ "$(basename "${MODEL_FILE}")" == "${TUNED_Q4_0_NAME}" ]]; then
  MMPROJ_OFFLOAD="on"
else
  MMPROJ_OFFLOAD="off"
fi
STARTUP_TIMEOUT="${RETINA_READY_STARTUP_TIMEOUT:-900}"
CORS_ORIGINS="${RETINA_READY_CORS_ORIGINS:-http://127.0.0.1:8000}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server is missing. On macOS: brew install llama.cpp" >&2
  exit 1
fi
if [[ ! -s "${MODEL_FILE}" || ! -s "${MMPROJ_FILE}" ]]; then
  echo "Model is incomplete at ${MODEL_DIR}." >&2
  if [[ "${MODEL_DIR}" == "${FALLBACK_MODEL_DIR}" ]]; then
    echo "Run: ${SCRIPT_DIR}/download_official_q4.sh" >&2
  else
    echo "Provide a complete tuned triad or set RETINA_READY_MODEL_FILE and RETINA_READY_MMPROJ_FILE explicitly." >&2
  fi
  exit 1
fi
if [[ -n "${LORA_FILE}" && ! -s "${LORA_FILE}" ]]; then
  echo "LoRA adapter is missing or empty: ${LORA_FILE}" >&2
  exit 1
fi

echo "Starting Gemma 4 26B A4B on ${HOST}:${PORT}"
echo "Profile: ${MODEL_PROFILE}"
echo "Server model alias: ${MODEL_ALIAS}"
echo "Model: ${MODEL_FILE}"
echo "Vision projector: ${MMPROJ_FILE}"
if [[ -n "${LORA_FILE}" ]]; then
  echo "RetinaReady LoRA: ${LORA_FILE}"
else
  echo "RetinaReady LoRA: disabled (official QAT fallback)"
fi
echo "24-GB Mac safety profile: context=${CTX_SIZE}, batch=${BATCH_SIZE}, ubatch=${UBATCH_SIZE}, vision batch=${MTMD_BATCH_MAX_TOKENS}, parallel=1, GPU layers=${GPU_LAYERS}, projector offload=${MMPROJ_OFFLOAD}."

server_args=(
  --model "${MODEL_FILE}"
  --alias "${MODEL_ALIAS}"
  --mmproj "${MMPROJ_FILE}"
  --host "${HOST}"
  --port "${PORT}"
  --cors-origins "${CORS_ORIGINS}"
  --ctx-size "${CTX_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --ubatch-size "${UBATCH_SIZE}"
  --mtmd-batch-max-tokens "${MTMD_BATCH_MAX_TOKENS}"
  --parallel 1
  --n-gpu-layers "${GPU_LAYERS}"
  --flash-attn auto
  --reasoning off
  --reasoning-budget 0
  --jinja
)
if [[ "${MMPROJ_OFFLOAD}" == "off" ]]; then
  server_args+=(--no-mmproj-offload)
elif [[ "${MMPROJ_OFFLOAD}" != "on" ]]; then
  echo "RETINA_READY_MMPROJ_OFFLOAD must be 'on' or 'off', got: ${MMPROJ_OFFLOAD}" >&2
  exit 1
fi
if [[ -n "${LORA_FILE}" ]]; then
  server_args+=(--lora "${LORA_FILE}")
fi

llama-server "${server_args[@]}" &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 "${SCRIPT_DIR}/wait_for_server.py" \
  --base-url "http://${HOST}:${PORT}" \
  --timeout "${STARTUP_TIMEOUT}" \
  --pid "${server_pid}"

echo "RetinaReady model server is ready at http://${HOST}:${PORT}/v1"
wait "${server_pid}"
