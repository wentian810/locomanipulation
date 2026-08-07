#!/usr/bin/env bash
# Publish a coordinate-corrected portable scene package without deleting the
# previous package.  This script is deliberately single-use and fails closed.
set -Eeuo pipefail

REPO=/home/jixingyu/Loco-manipulation-human-only-release
CLIP='Ways_to_Jump_+_Sit_+_Fall_Dramatic_clip1'
HUMAN="$REPO/output_dir/sit_contact_pilot_v2/$CLIP"
CURRENT="$HUMAN/scene_reconstruction"
BACKUP="$HUMAN/scene_reconstruction_pre_sim_frame_v2"
SOURCE="$REPO/scene_work/sit_contact_pilot_v2/$CLIP/videomimic_package_mujoco_asset_v2/videomimic/output_calib_mesh"
WORK="$REPO/scene_work/sit_contact_pilot_v2/$CLIP/videomimic_package_sim_frame_v3"
CALIBRATED="$WORK/videomimic/output_calib_mesh"
PYTHON=/home/jixingyu/miniconda3/envs/vm1recon/bin/python

[[ -d "$CURRENT" ]] || { echo "missing current package: $CURRENT" >&2; exit 1; }
[[ ! -e "$BACKUP" ]] || { echo "refusing to overwrite backup: $BACKUP" >&2; exit 1; }
[[ ! -e "$WORK" ]] || { echo "refusing to reuse work directory: $WORK" >&2; exit 1; }
[[ -f "$SOURCE/gravity_calibrated_megahunter.h5" ]] || exit 1
[[ -f "$SOURCE/background_mesh_nksr_first_round.obj" ]] || exit 1

mkdir -p "$CALIBRATED"
ln "$SOURCE/gravity_calibrated_megahunter.h5" "$CALIBRATED/gravity_calibrated_megahunter.h5"
ln "$SOURCE/background_mesh_nksr_first_round.obj" "$CALIBRATED/background_mesh_nksr_first_round.obj"

restore_current() {
  local status=$?
  if [[ $status -ne 0 && ! -e "$CURRENT" && -e "$BACKUP" ]]; then
    mv "$BACKUP" "$CURRENT"
    echo "[RESTORED] previous package after failure" >&2
  fi
  exit "$status"
}
trap restore_current ERR

mv "$CURRENT" "$BACKUP"
export PYTHONNOUSERSITE=1
"$PYTHON" "$REPO/scripts/scene/package_videomimic_scene.py" \
  --project-root "$REPO" \
  --human-dir "$HUMAN" \
  --work-dir "$WORK" \
  --output-root "$CURRENT" \
  --model-root "$REPO/external/VideoMimic/real2sim/assets/body_models" \
  --motion-npz "$HUMAN/001_smoothed.npz" \
  --camera-npz "$HUMAN/gvhmr_camera.npz" \
  --gender male \
  --camera-mode static_hard

trap - ERR
echo '[PASS] coordinate-corrected scene package published'
