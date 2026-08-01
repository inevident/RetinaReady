#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${PROJECT_DIR}/data/raw"
ARCHIVE="${RAW_DIR}/DeepDRiD-v1.1.zip"
EXTRACT_DIR="${RAW_DIR}/deepdrid-v1.1"

ZENODO_URL="https://zenodo.org/api/records/8248825/files/deepdrdoc/DeepDRiD-v1.1.zip/content"
EXPECTED_BYTES="1373472897"
EXPECTED_MD5="3379e2fd7a2dd398545a67148420a5d3"

mkdir -p "${RAW_DIR}"

if [[ ! -f "${ARCHIVE}" ]] || [[ "$(wc -c < "${ARCHIVE}" | tr -d ' ')" != "${EXPECTED_BYTES}" ]]; then
  echo "Downloading DeepDRiD v1.1 from the pinned Zenodo record..."
  echo "Destination: ${ARCHIVE}"
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --continue-at - \
    --output "${ARCHIVE}" \
    "${ZENODO_URL}"
fi

ACTUAL_BYTES="$(wc -c < "${ARCHIVE}" | tr -d ' ')"
if [[ "${ACTUAL_BYTES}" != "${EXPECTED_BYTES}" ]]; then
  echo "ERROR: archive has ${ACTUAL_BYTES} bytes; expected ${EXPECTED_BYTES}." >&2
  echo "Re-run this script to resume the download." >&2
  exit 1
fi

if command -v md5 >/dev/null 2>&1; then
  ACTUAL_MD5="$(md5 -q "${ARCHIVE}")"
elif command -v md5sum >/dev/null 2>&1; then
  ACTUAL_MD5="$(md5sum "${ARCHIVE}" | awk '{print $1}')"
else
  echo "ERROR: neither md5 nor md5sum is installed." >&2
  exit 1
fi

if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
  echo "ERROR: MD5 mismatch for ${ARCHIVE}." >&2
  echo "Expected: ${EXPECTED_MD5}" >&2
  echo "Actual:   ${ACTUAL_MD5}" >&2
  echo "Move the invalid archive aside and re-run this script." >&2
  exit 1
fi

echo "Verified ${ACTUAL_BYTES} bytes and MD5 ${ACTUAL_MD5}."

EXPECTED_LABELS="${EXTRACT_DIR}/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv"
if [[ ! -f "${EXPECTED_LABELS}" ]]; then
  mkdir -p "${EXTRACT_DIR}"
  echo "Extracting into ${EXTRACT_DIR}..."
  bsdtar -xf "${ARCHIVE}" -C "${EXTRACT_DIR}" --strip-components 1
fi

if [[ ! -f "${EXPECTED_LABELS}" ]]; then
  echo "ERROR: extraction finished without the expected training labels:" >&2
  echo "${EXPECTED_LABELS}" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/prepare_deepdrid.py" \
  --dataset-root "${EXTRACT_DIR}" \
  --output-dir "${PROJECT_DIR}/data/manifests"

echo
echo "DeepDRiD is ready. Raw data remains ignored by version control."
echo "Manifests: ${PROJECT_DIR}/data/manifests"
