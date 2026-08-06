#!/usr/bin/env bash
set -euo pipefail

ROOT="${PIPELINE_ROOT:-/workspace/locomotion}"
PY_LOCO="${PY_LOCO:-${CONDA_BASE:-/opt/conda}/envs/locomotion/bin/python}"
export PIPELINE_ROOT CONDA_BASE="${CONDA_BASE:-/opt/conda}" PY_LOCO
export PY_GVHMR="${PY_GVHMR:-$PY_LOCO}"
export PY_PHC="${PY_PHC:-${CONDA_BASE}/envs/phc/bin/python}"
export PY_GMR="${PY_GMR:-$PY_LOCO}"
export PIPELINE_PYTHON="${PIPELINE_PYTHON:-$PY_LOCO}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}:${ROOT}/GVHMR-hand/GVHMR-main:${ROOT}/locomotion_pipeline-main:${ROOT}/GMR-master:${ROOT}/phc-dev-felix-pipeline:${PYTHONPATH:-}"

cd "$ROOT"

case "${1:-check}" in
  check)
    shift || true
    exec bash "$ROOT/docker/preflight.sh" "$@"
    ;;
  run|human)
    shift || true
    exec "$PY_LOCO" "$ROOT/scripts/run_pipeline_from_config.py" \
      --config_dir "$ROOT/configs/pipelines/human_sharpa.yaml" \
      --stage all "$@"
    ;;
  shell|bash)
    shift || true
    exec /bin/bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
