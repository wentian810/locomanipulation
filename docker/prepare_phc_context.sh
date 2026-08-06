#!/usr/bin/env bash
set -euo pipefail

# Prepare a small named Docker context from the server-pipeline tree.  Hard
# links avoid duplicating the multi-GB PHC runtime on the same filesystem;
# this script never removes or modifies the source tree.
SRC="${1:-/home/jixingyu/Loco-manipulation-human-only-release}"
DST="${2:-${HOME}/.codex_phc_context_20260806}"

for required in \
  "$SRC/phc-dev-felix-pipeline/phc" \
  "$SRC/phc-dev-felix-pipeline/poselib" \
  "$SRC/phc-dev-felix-pipeline/isaacgym" \
  "$SRC/phc-dev-felix-pipeline/data" \
  "$SRC/phc-dev-felix-pipeline/assets" \
  "$SRC/phc-dev-felix-pipeline/scripts" \
  "$SRC/phc-dev-felix-pipeline/sample_data" \
  "$SRC/phc-dev-felix-pipeline/output/HumanoidIm" \
  "$SRC/phc-deps"; do
  test -e "$required" || { echo "missing PHC source: $required" >&2; exit 2; }
done

if [ -e "$DST" ]; then
  echo "refusing to overwrite existing context: $DST" >&2
  exit 3
fi

mkdir -p "$DST/phc-dev-felix-pipeline/output"
cp -al "$SRC/phc-dev-felix-pipeline/phc" "$DST/phc-dev-felix-pipeline/"
cp -al "$SRC/phc-dev-felix-pipeline/poselib" "$DST/phc-dev-felix-pipeline/"
cp -al "$SRC/phc-dev-felix-pipeline/isaacgym" "$DST/phc-dev-felix-pipeline/"
cp -al "$SRC/phc-dev-felix-pipeline/data" "$DST/phc-dev-felix-pipeline/"
cp -al "$SRC/phc-dev-felix-pipeline/assets" "$DST/phc-dev-felix-pipeline/"
cp -al "$SRC/phc-dev-felix-pipeline/scripts" "$DST/phc-dev-felix-pipeline/"
cp -al "$SRC/phc-dev-felix-pipeline/sample_data" "$DST/phc-dev-felix-pipeline/"
cp -al "$SRC/phc-dev-felix-pipeline/output/HumanoidIm" "$DST/phc-dev-felix-pipeline/output/"
cp -al "$SRC/phc-deps" "$DST/"

du -sh "$DST"
echo "PHC Docker context ready: $DST"
