#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/jixingyu/Loco-manipulation-human-only-release
CLIP='Ways_to_Jump_+_Sit_+_Fall_Dramatic_clip1'
PKG="$REPO/output_dir/sit_contact_pilot_v2/$CLIP/scene_reconstruction"
export PYTHONNOUSERSITE=1
PYTHON=/home/jixingyu/miniconda3/envs/vm1recon/bin/python
MUJOCO_PYTHON=/home/jixingyu/miniconda3/envs/locomotion/bin/python

"$PYTHON" "$REPO/scripts/scene/validate_scene_package.py" --scene-root "$PKG"
"$MUJOCO_PYTHON" - "$PKG/scene/scene_mujoco.xml" "$PKG/scene/simulation_frame.json" "$PKG/scene/primitives_mujoco.json" <<'PY'
import json
import sys

import mujoco

model = mujoco.MjModel.from_xml_path(sys.argv[1])
print(f"MUJOCO_OK nmesh={model.nmesh} ngeom={model.ngeom}")
report = json.load(open(sys.argv[2], encoding="utf-8"))
primitive = json.load(open(sys.argv[3], encoding="utf-8"))["primitives"][0]
print(
    "SIM_FRAME", report["source_frame"], "->", report["target_frame"],
    "vertices", report["vertex_count"], "primitives", report["primitive_count"],
)
print("PRIMITIVE", primitive["center"], primitive["extents"])
PY

du -sh "$PKG"
find "$PKG/scene" -maxdepth 1 -type f -printf '%f %s\n' | sort
