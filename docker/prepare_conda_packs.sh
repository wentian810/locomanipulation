#!/usr/bin/env bash
set -euo pipefail

# Package the two environments that are known to work on the server. This
# does not modify either environment; it only creates relocatable archives.
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/.codex_conda_packs}"
N_THREADS="${N_THREADS:-8}"

if ! command -v conda-pack >/dev/null 2>&1; then
  echo "conda-pack is required; install it in a temporary/base environment first:" >&2
  echo "  python -m pip install conda-pack" >&2
  exit 1
fi

for env_name in locomotion phc; do
  env_path="${CONDA_BASE}/envs/${env_name}"
  archive="${OUTPUT_DIR}/${env_name}.tar.gz"
  if [ ! -d "${env_path}" ]; then
    echo "missing conda environment: ${env_path}" >&2
    exit 1
  fi
  if [ -e "${archive}" ]; then
    echo "refusing to overwrite existing archive: ${archive}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
conda-pack -p "${CONDA_BASE}/envs/locomotion" \
  -o "${OUTPUT_DIR}/locomotion.tar.gz" \
  --ignore-editable-packages --n-threads "${N_THREADS}"
conda-pack -p "${CONDA_BASE}/envs/phc" \
  -o "${OUTPUT_DIR}/phc.tar.gz" \
  --ignore-editable-packages --n-threads "${N_THREADS}"

sha256sum "${OUTPUT_DIR}/locomotion.tar.gz" "${OUTPUT_DIR}/phc.tar.gz"
du -h "${OUTPUT_DIR}/locomotion.tar.gz" "${OUTPUT_DIR}/phc.tar.gz"
