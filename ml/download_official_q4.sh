#!/usr/bin/env bash
set -euo pipefail

# Official Google QAT Q4_0 release. This is an inference artifact, not the
# bitsandbytes/NF4 checkpoint used by train_qlora.py.
REPO_ID="google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
REVISION="d1c082be9cf3c8a514acf63b8761f4b41935842e"
MODEL_FILE="gemma-4-26B_q4_0-it.gguf"
MMPROJ_FILE="gemma-4-26B-it-mmproj.gguf"
MODEL_BYTES=14439363584
MMPROJ_BYTES=1194828160

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${RETINA_READY_QAT_MODEL_DIR:-${PROJECT_ROOT}/models/gemma-4-26b-q4}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install it with: python3 -m pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"

required_kib=$(( (MODEL_BYTES + MMPROJ_BYTES + 3 * 1024 * 1024 * 1024 + 1023) / 1024 ))
available_kib="$(df -Pk "${MODEL_DIR}" | awk 'NR==2 {print $4}')"
if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || (( available_kib < required_kib )); then
  echo "Need at least 18.6 GB free for the two official files plus download headroom." >&2
  exit 1
fi

echo "Downloading official Gemma 4 26B A4B QAT Q4_0 files to ${MODEL_DIR}"
echo "Repository is public; HF_TOKEN is optional unless Hugging Face rate-limits the request."
hf download "${REPO_ID}" \
  "${MODEL_FILE}" \
  "${MMPROJ_FILE}" \
  "README.md" \
  ".gitattributes" \
  --revision "${REVISION}" \
  --local-dir "${MODEL_DIR}"

actual_model_bytes="$(wc -c < "${MODEL_DIR}/${MODEL_FILE}" | tr -d '[:space:]')"
actual_mmproj_bytes="$(wc -c < "${MODEL_DIR}/${MMPROJ_FILE}" | tr -d '[:space:]')"
if (( actual_model_bytes != MODEL_BYTES || actual_mmproj_bytes != MMPROJ_BYTES )); then
  echo "Downloaded file size did not match the pinned official revision." >&2
  exit 1
fi

echo "Download complete:"
du -h "${MODEL_DIR}/${MODEL_FILE}" "${MODEL_DIR}/${MMPROJ_FILE}"
