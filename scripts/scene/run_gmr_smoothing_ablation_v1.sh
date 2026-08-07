#!/usr/bin/env bash
# Compare original GMR retargeting with only the documented qpos smoothing changed.
set -euo pipefail

repo=/home/jixingyu/Loco-manipulation-human-only-release
show_root=$repo/output_dir/videomimic_calibrated_semantic_batch_v6
output_root=$repo/scene_work/gmr_original_weight_smoothing_ablation_v1
wrapper=$repo/GMR-master/run_show_gmr_batch.sh
audit_python=/home/jixingyu/miniconda3/envs/locomotion/bin/python
audit_script=$repo/scripts/scene/audit_gmr_foot_support.py
robot_xml=$repo/GMR-master/assets/unitree_g1/g1_mocap_29dof.xml
clips=Ways_to_Jump_+_Sit_+_Fall_Buying_a_Chair_clip1,Ways_to_Jump_+_Sit_+_Fall_Dramatic_clip1

[[ ! -e "$output_root" ]] || { echo "refusing to mix with existing ablation: $output_root" >&2; exit 2; }
for path in "$show_root" "$wrapper" "$audit_python" "$audit_script" "$robot_xml"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 2; }
done
mkdir -p "$output_root"

for smooth_window in 0 5; do
  variant=$output_root/smooth_${smooth_window}
  mkdir -p "$variant"
  sha256sum \
    "$show_root/Ways_to_Jump_+_Sit_+_Fall_Buying_a_Chair_clip1/001_smoothed.npz" \
    "$show_root/Ways_to_Jump_+_Sit_+_Fall_Dramatic_clip1/001_smoothed.npz" \
    "$repo/GMR-master/scripts/smpl_npz_to_robot_headless.py" > "$variant/input_sha256.txt"
  (
    export SHOW_ROOT="$show_root"
    export OUT_ROOT="$variant"
    export GMR_SOURCE=smoothed
    export GMR_SELECTED_CLIPS="$clips"
    export GMR_SMOOTH_WINDOW="$smooth_window"
    # Pin every baseline GMR contract; only smooth_window may differ.
    export GMR_ROBOT=unitree_g1
    export GMR_HEIGHT_ADJUST_MODE=support_aware_foot_geom
    export GMR_HAND_RETARGET_MODE=off
    export GMR_AUTO_HAND_NPZ=0
    export GMR_HAND_WRIST_ORIENTATION_MODE=diagnostic
    export GMR_PALM_ROLL_MODE=off
    export GMR_WRIST_PITCH_YAW_STABILIZE=soft_limit
    export GMR_RELAX_ORIENTATION_BODIES=
    export GMR_HUMAN_YAW_OFFSET_DEG=0.0
    export GMR_OVERRIDE=1
    export GMR_RENDER=0
    export GMR_COMPOSITE=0
    export PYTHONNOUSERSITE=1
    bash "$wrapper"
  ) > "$variant/gmr_retarget.log" 2>&1

  for clip in ${clips//,/ }; do
    motion=$variant/$clip/robot_motion.pkl
    [[ -s "$motion" ]] || { echo "missing candidate motion: $motion" >&2; exit 1; }
    "$audit_python" "$audit_script" \
      --robot-motion "$motion" \
      --robot-xml "$robot_xml" \
      --report "$variant/$clip/gmr_foot_support_audit.json" \
      > "$variant/$clip/gmr_foot_support_audit.log" 2>&1
  done
done

"$audit_python" - <<'PY'
import json
from pathlib import Path

root = Path('/home/jixingyu/Loco-manipulation-human-only-release/scene_work/gmr_original_weight_smoothing_ablation_v1')
summary = []
for report_path in sorted(root.glob('smooth_*/*/gmr_foot_support_audit.json')):
    report = json.loads(report_path.read_text())
    summary.append({
        'candidate': str(report_path.parent),
        'accepted': report['accepted'],
        'rejection_reasons': report['rejection_reasons'],
        'feet': {
            side: {
                'stable_support_frame_count': detail['stable_support_frame_count'],
                'ground_skate_frame_count': detail['ground_skate_frame_count'],
                'near_ground_motion_frame_count': detail['near_ground_motion_frame_count'],
                'hover_frame_count': detail['hover_frame_count'],
            }
            for side, detail in report['feet'].items()
        },
    })
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
