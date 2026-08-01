#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

host=""
remote_dir=""
execute=0
data_mode="none"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/sync_to_a100.sh --host USER@HOST --remote-dir /absolute/remote/path [options]

Required:
  --host USER@HOST       SSH host or configured SSH alias.
  --remote-dir PATH      Existing absolute directory for RetinaReady on the host.

Transfer mode:
  --include-archive      Include only data/raw/DeepDRiD-v1.1.zip (~1.3 GiB).
  --include-raw          Include the archive and extracted data/raw tree (~2.6 GiB).
  --execute              Perform the transfer. Without this flag, rsync is a dry run.

Other:
  -h, --help             Show this help.

The local GGUF/model directory, outputs, checkpoints, virtual environments,
package directories, and caches are always excluded. The sync never removes
files from the remote destination.

Examples:
  # Preview code + manifests only (still opens an SSH connection):
  ./scripts/sync_to_a100.sh \
    --host gpu-a100 \
    --remote-dir /home/ubuntu/retina-ready

  # Transfer code + manifests + the verified dataset archive:
  ./scripts/sync_to_a100.sh \
    --host ubuntu@gpu.example \
    --remote-dir /home/ubuntu/retina-ready \
    --include-archive \
    --execute
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --host)
      (($# >= 2)) || die "--host requires a value"
      host="$2"
      shift 2
      ;;
    --remote-dir)
      (($# >= 2)) || die "--remote-dir requires a value"
      remote_dir="$2"
      shift 2
      ;;
    --include-archive)
      [[ "${data_mode}" == "none" ]] ||
        die "--include-archive and --include-raw are mutually exclusive"
      data_mode="archive"
      shift
      ;;
    --include-raw)
      [[ "${data_mode}" == "none" ]] ||
        die "--include-archive and --include-raw are mutually exclusive"
      data_mode="raw"
      shift
      ;;
    --execute)
      execute=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -n "${host}" ]] || die "--host is required"
[[ -n "${remote_dir}" ]] || die "--remote-dir is required"
command -v rsync >/dev/null 2>&1 || die "rsync is not installed"

# Keep both values out of a remote shell's metacharacter grammar. SSH aliases,
# hostnames, IPv4 addresses, and user@host targets are supported. Put ports,
# ProxyJump, and identity files in ~/.ssh/config.
[[ "${host}" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] ||
  die "--host must be USER@HOST or a simple SSH alias; use ~/.ssh/config for advanced options"
[[ "${remote_dir}" == /* ]] ||
  die "--remote-dir must be an absolute POSIX path"
[[ "${remote_dir}" != "/" ]] ||
  die "--remote-dir cannot be /"
[[ "${remote_dir}" =~ ^/[A-Za-z0-9._+/-]+$ ]] ||
  die "--remote-dir may contain only letters, digits, '.', '_', '+', '-', and '/'"
[[ "/${remote_dir#/}/" != *"/../"* ]] ||
  die "--remote-dir cannot contain '..' path segments"

archive="${PROJECT_ROOT}/data/raw/DeepDRiD-v1.1.zip"
raw_dir="${PROJECT_ROOT}/data/raw/deepdrid-v1.1"
if [[ "${data_mode}" != "none" && ! -f "${archive}" ]]; then
  die "requested archive is missing: ${archive}"
fi
if [[ "${data_mode}" != "none" ]]; then
  expected_bytes="1373472897"
  expected_md5="3379e2fd7a2dd398545a67148420a5d3"
  actual_bytes="$(wc -c < "${archive}" | tr -d ' ')"
  [[ "${actual_bytes}" == "${expected_bytes}" ]] ||
    die "DeepDRiD archive size is ${actual_bytes}; expected ${expected_bytes}"
  if command -v md5 >/dev/null 2>&1; then
    actual_md5="$(md5 -q "${archive}")"
  elif command -v md5sum >/dev/null 2>&1; then
    actual_md5="$(md5sum "${archive}" | awk '{print $1}')"
  else
    die "neither md5 nor md5sum is installed; cannot verify the archive"
  fi
  [[ "${actual_md5}" == "${expected_md5}" ]] ||
    die "DeepDRiD archive checksum does not match the pinned v1.1 release"
fi
if [[ "${data_mode}" == "raw" && ! -d "${raw_dir}" ]]; then
  die "requested extracted dataset is missing: ${raw_dir}"
fi

rsync_args=(
  --archive
  --human-readable
  --itemize-changes
  --partial
  --prune-empty-dirs
  --exclude=/.git/
  --exclude=/.DS_Store
  --exclude=/.playwright-cli/
  --exclude=/models/
  --exclude=/outputs/
  --exclude=/output/
  --exclude=/checkpoints/
  --exclude=/ml/runs/
  --exclude=/data/cache/
  --exclude=/data/external/
  --exclude=/data/extracted/
  --exclude=/ml/cache/
  --exclude=.venv*/
  --exclude=venv*/
  --exclude=env/
  --exclude=__pycache__/
  --exclude=.pytest_cache/
  --exclude=.mypy_cache/
  --exclude=.ruff_cache/
  --exclude=.cache/
  --exclude=node_modules/
  --exclude='*.gguf'
  --exclude='*.safetensors'
  --exclude='*.pyc'
)

case "${data_mode}" in
  none)
    rsync_args+=(--exclude=/data/raw/)
    ;;
  archive)
    # Filter ordering matters: permit the directory and verified archive, then
    # reject every other raw-data entry.
    rsync_args+=(
      --include=/data/raw/
      --include=/data/raw/DeepDRiD-v1.1.zip
      '--exclude=/data/raw/**'
    )
    ;;
  raw)
    ;;
esac

if ((execute == 0)); then
  rsync_args+=(--dry-run)
fi

destination="${host}:${remote_dir%/}/"

echo "RetinaReady source: ${PROJECT_ROOT}/"
echo "Remote destination: ${destination}"
echo "Dataset transfer: ${data_mode}"
if ((execute == 0)); then
  echo "Mode: DRY RUN (no files will be written; SSH connectivity is still required)"
else
  echo "Mode: EXECUTE"
fi
echo

# A trailing slash transfers the project contents into the explicitly supplied
# remote directory. No remote-deletion option is used.
rsync "${rsync_args[@]}" "${PROJECT_ROOT}/" "${destination}"

if ((execute == 0)); then
  echo
  echo "Dry run complete. Review the list, then repeat with --execute."
else
  echo
  echo "Transfer complete. Local-only models and generated outputs were not sent."
fi
