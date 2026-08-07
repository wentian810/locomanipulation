#!/usr/bin/env bash
# Batch Stage-4 contact-residual sweep. No model/checkpoint is trained or replaced.
set -euo pipefail

repo=/home/jixingyu/Loco-manipulation-human-only-release
stage_python=/home/jixingyu/miniconda3/envs/vm1rs/bin/python
audit_python=/home/jixingyu/miniconda3/envs/locomotion/bin/python
stage_script=$repo/external/VideoMimic/real2sim/stage4_retargeting/robot_motion_retargeting.py
audit_script=$repo/scripts/scene/audit_h5_gmr_contact_alignment.py
baseline_root=$repo/scene_work/videomimic_policy_stage2_v1
gmr_root=$repo/output_dir/videomimic_calibrated_semantic_batch_v6
output_root=$repo/scene_work/gmr_original_weights_contact_sweep_v1
cudnn_lib=/home/jixingyu/miniconda3/envs/vm1rs/lib/python3.12/site-packages/nvidia/cudnn/lib

if [[ -e "$output_root" ]]; then
  echo "refusing to mix with an existing sweep: $output_root" >&2
  exit 2
fi
for path in "$stage_python" "$audit_python" "$stage_script" "$audit_script" "$cudnn_lib"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
mkdir -p "$output_root"

clips=(
  Ways_to_Jump_+_Sit_+_Fall_Buying_a_Chair_clip1
  Ways_to_Jump_+_Sit_+_Fall_Dramatic_clip1
)
contact_weights=(30 100)

contact_dir_for() {
  case "$1" in
    Ways_to_Jump_+_Sit_+_Fall_Buying_a_Chair_clip1)
      printf '%s\n' "$repo/scene_work/videomimic_real_scene_regression_v2_repackage/$1/videomimic/input_contacts/$1/cam01"
      ;;
    Ways_to_Jump_+_Sit_+_Fall_Dramatic_clip1)
      printf '%s\n' "$repo/scene_work/videomimic_real_scene_regression_v2_stage4/$1/bstro_contacts"
      ;;
    *)
      echo "unknown clip: $1" >&2
      return 2
      ;;
  esac
}

copy_stage4_inputs() {
  local source_dir=$1
  local target_dir=$2
  mkdir -p "$target_dir"
  local name
  for name in evidence_manifest.json gravity_calibrated_megahunter.h5 gravity_calibrated_keypoints.h5 background_mesh.obj background_mesh_contact_recovered.obj; do
    [[ -e "$source_dir/$name" ]] || { echo "missing Stage-4 input: $source_dir/$name" >&2; return 2; }
    cp -a "$source_dir/$name" "$target_dir/$name"
  done
}

for clip in "${clips[@]}"; do
  reference_dir=$baseline_root/$clip/stage4_reference
  case "$clip" in
    Ways_to_Jump_+_Sit_+_Fall_Buying_a_Chair_clip1)
      source_input=$reference_dir/retarget_poses_g1_input_frame_0_221_subsample_1
      stage_input_name=retarget_poses_g1_input_frame_0_221_subsample_1
      ;;
    Ways_to_Jump_+_Sit_+_Fall_Dramatic_clip1)
      source_input=$reference_dir/retarget_poses_g1_input_frame_0_154_subsample_1
      stage_input_name=retarget_poses_g1_input_frame_0_154_subsample_1
      ;;
  esac
  contact_dir=$(contact_dir_for "$clip")
  baseline_motion=$gmr_root/$clip/robot_motion.pkl
  [[ -d "$source_input" && -d "$contact_dir" && -f "$baseline_motion" ]] || {
    echo "missing immutable input for $clip" >&2
    exit 2
  }

  for contact_weight in "${contact_weights[@]}"; do
    candidate=$output_root/foot_skating_${contact_weight}/$clip
    # VideoMimic derives the subsample factor from this directory name.
    stage_input=$candidate/$stage_input_name
    mkdir -p "$candidate"
    copy_stage4_inputs "$source_input" "$stage_input"
    sha256sum "$reference_dir/retarget_poses_g1.h5" "$baseline_motion" "$stage_script" > "$candidate/input_sha256.txt"
    printf '%s\n' "contact_dir=$contact_dir" >> "$candidate/input_sha256.txt"
    printf '%s\n' "foot_skating_cost_weight=$contact_weight" >> "$candidate/input_sha256.txt"
    printf '%s\n' 'ground_contact_cost_weight=1.0' >> "$candidate/input_sha256.txt"

    (
      export PYTHONNOUSERSITE=1
      # Prevent JAX from reserving the entire GPU before the optimizer starts.
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      export LD_LIBRARY_PATH="$cudnn_lib:$($stage_python -c 'import sys; print(sys.prefix)')/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      cd "$repo/external/VideoMimic/real2sim"
      "$stage_python" "$stage_script" \
        --src-dir "$stage_input" \
        --contact-dir "$contact_dir" \
        --foot-skating-cost-weight "$contact_weight" \
        --ground-contact-cost-weight 1.0
    ) > "$candidate/stage4.log" 2>&1

    [[ -s "$stage_input/retarget_poses_g1.h5" ]] || {
      echo "Stage-4 did not create an H5 for $clip at contact weight $contact_weight" >&2
      exit 1
    }
    cp -a "$stage_input/retarget_poses_g1.h5" "$candidate/retarget_poses_g1.h5"
    "$audit_python" "$audit_script" \
      --contact-h5 "$candidate/retarget_poses_g1.h5" \
      --robot-motion "$baseline_motion" \
      --report "$candidate/h5_to_gmr_contact_transfer_audit.json" \
      > "$candidate/contact_audit.log" 2>&1
    sha256sum "$candidate/retarget_poses_g1.h5" >> "$candidate/input_sha256.txt"
  done
done

"$audit_python" - <<'PY'
import json
from pathlib import Path

root = Path('/home/jixingyu/Loco-manipulation-human-only-release/scene_work/gmr_original_weights_contact_sweep_v1')
summary = []
for report_path in sorted(root.glob('foot_skating_*/*/h5_to_gmr_contact_transfer_audit.json')):
    report = json.loads(report_path.read_text())
    summary.append({
        'candidate': str(report_path.parent),
        'accepted_for_contact_transfer': report['accepted_for_contact_transfer'],
        'rejection_reasons': report['rejection_reasons'],
        'best_temporal_alignment': report['joint_contract']['best'],
        'h5_contact_speed_p95_m_s': {
            side: details['horizontal_speed_during_h5_contact_m_s']['p95']
            for side, details in report['h5_contacts'].items()
        },
    })
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
