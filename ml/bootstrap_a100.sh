#!/usr/bin/env bash
# Prepare and verify an A100 environment. This script never starts training.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${RETINA_READY_A100_VENV:-${PROJECT_ROOT}/.venv-a100}"
CONFIG_PATH="${RETINA_READY_TRAIN_CONFIG:-${PROJECT_ROOT}/ml/configs/gemma4_26b_smoke.json}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "A100 bootstrap requires Linux; detected $(uname -s)." >&2
  echo "Nothing was installed and no training was started." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
  echo "Use Python 3.10, 3.11, or 3.12 for the CUDA training environment." >&2
  echo "Nothing was installed and no training was started." >&2
  exit 1
fi

echo "Creating/reusing virtual environment: ${VENV_PATH}"
python3 -m venv "${VENV_PATH}"
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PROJECT_ROOT}/ml/requirements-train.txt"

echo
echo "Dependencies installed. Running read-only A100 preflight."
echo "This script will not start training or download model weights."
cd "${PROJECT_ROOT}"
python ml/preflight_a100.py \
  --config "${CONFIG_PATH}" \
  --json-output ml/runs/a100-preflight.json
