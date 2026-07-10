# GMR 手部与腕部调整说明

日期：2026-06-24

这份文档简要记录当前针对 GMR / GVHMR-hand / do-as-i-do 思路所做的调整。目标是让 Unitree G1 的手指、掌心翻转和腕部表现更稳定，尤其避免手腕和前臂形成很夸张的角度。

## 当前稳定方案

目前 G1 不直接使用完整的 MANO 手腕朝向。完整 wrist orientation 会把掌心翻转、手腕 roll、pitch、yaw 都塞进 G1 的手腕关节里，容易造成 wrist pitch/yaw 到极限，视觉上就是手腕折得很怪。

现在采用的是更保守的方案：

1. GVHMR-hand / HaMeR 输出 `001_smplx_hands.npz` 手部 sidecar。
2. GMR 读取 sidecar 里的左右手 pose 和有效帧 mask。
3. 手指关节直接由 MANO hand pose 映射到 G1 手指，并做平滑、限速、死区和关节范围 clamp。
4. 掌心翻面只提取为绕前臂方向的 roll 信号。
5. 这个 roll 只写入 G1 的 `left_wrist_roll_joint` / `right_wrist_roll_joint`。
6. G1 的 wrist pitch / yaw 固定到 neutral，避免手腕脱离前臂方向。
7. 渲染 2x2 时使用 GVHMR 导出的相机，但固定第一帧相机状态，避免画面抖动。

## 和 do-as-i-do 的关系

我们主要借用了 do-as-i-do 里“用手腕、index/middle/ring MCP 点构造掌心几何坐标系”的思路。

简单说，就是不完全相信 MANO 每一帧的手腕朝向，而是从手部关键点里估计一个更稳定的掌心方向。这个信息在 G1 上没有直接用作完整 wrist orientation，而是只提取“掌心是否翻转”的 roll 分量。

这样做的原因是：G1 的 wrist 结构和 MANO/人手并不是一一对应，强行追完整手腕朝向会让 pitch/yaw 乱动；只保留 roll 会更稳。

## 主要涉及文件

- `GMR-master/scripts/smpl_npz_to_robot_headless.py`
  - GMR 主转换脚本。
  - 读取 hand sidecar。
  - 做手指 retarget。
  - 做 G1 palm roll 映射。
  - 做 G1 wrist pitch/yaw 稳定。

- `GMR-master/general_motion_retargeting/hand_wrist_utils.py`
  - 放 do-as-i-do 风格的 MCP 掌心几何 frame 计算。

- `GMR-master/general_motion_retargeting/utils/smpl.py`
  - 支持从 sidecar 读取手部 pose、valid mask 和 wrist/palm 诊断信息。

- `GVHMR-hand/GVHMR-main/tools/pipeline/convert_to_npz.py`
  - 负责生成 `001_smplx_hands.npz`。
  - 里面的 wrist-frame 方向已经和 do-as-i-do 的实现对齐。

- `GMR-master/run_show_gmr_batch.sh`
  - GMR 批处理、渲染和 2x2 拼接脚本。
  - 默认走 `GMR_CAMERA_SOURCE=gvhmr`，渲染时会传 `--camera_static`，也就是使用 GVHMR 第一帧相机。

- `GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6.sh`
  - 从 `dataset_new6` 跑完整流程的 batch 脚本。
  - 当前默认已经改成 G1 路线。

## 当前 G1 推荐默认值

```text
GMR_ROBOT=unitree_g1
GMR_MODEL_TYPE=smplh
GMR_AUTO_HAND_NPZ=1
GMR_HAND_WRIST_ORIENTATION_MODE=diagnostic
GMR_PALM_ROLL_MODE=auto
GMR_WRIST_PITCH_YAW_STABILIZE=auto
GMR_CAMERA_SOURCE=gvhmr
GMR_MUJOCO_GL=osmesa
```

含义：

- 不对 G1 启用完整 wrist orientation override。
- 有 hand sidecar 时自动启用 palm roll。
- 自动固定 wrist pitch/yaw 到 neutral。
- 使用 GVHMR 相机第一帧做稳定渲染视角。

## 保留但不建议 G1 使用的功能

`override_frames` 还保留着，但不建议 G1 默认使用。

它的用途主要是：

- 做诊断对比。
- 给 H1 hand-wrist 这类 wrist orientation 更匹配的机器人继续实验。
- 以后如果要测试完整 wrist quaternion，可以用它做入口。

所以没有把它删掉，只是在 batch 里防止 G1 默认误开。

## 已做的小清理

- 清理了 `smpl_npz_to_robot_headless.py` 里未使用的 import。
- 清理了 sidecar 读取路径里未使用的局部变量。
- 统一了 GVHMR sidecar wrist-frame 的方向，避免和 do-as-i-do / GMR helper 不一致。

## 目前已验证样例

已验证的 2x2 结果：

```text
GMR-master/output/show_gmr/unitree_g1_palm_roll_wrist_stable_2x2_gvhmr_static/
  Date02_Sub02_boxsmall_hand.1.color/
    Date02_Sub02_boxsmall_hand.1.color.mp4
```

对应 `robot_motion.pkl` 的关键状态：

```text
palm_roll_applied: True
palm_roll_sides: ['left', 'right']
left_palm_roll_frames: 1309
right_palm_roll_frames: 1397
wrist_pitch_yaw_stabilized: True
hand_wrist_orientation_override_enabled: False
relaxed_orientation_bodies: ['left_wrist', 'right_wrist']
```

## dataset_new6 批处理

当前 `dataset_new6` 里有五个视频：

```text
Date02_Sub02_boxsmall_hand.1.color.mp4
Date02_Sub02_chairwood_hand.0.color.mp4
Date02_Sub02_suitcase_lift.2.color.mp4
Date02_Sub02_tablesquare_move.2.color.mp4
Date02_Sub02_yogaball_play.0.color.mp4
```

推荐命令：

```bash
cd <repo-root>

PY_GMR="${HOME}/miniconda3/envs/locomotion/bin/python" \
bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6.sh
```

如果想显式指定输出目录，可以用：

```bash
cd <repo-root>

env \
  PY_GMR="${HOME}/miniconda3/envs/locomotion/bin/python" \
  DATASET="${PWD}/dataset_new6" \
  OUTPUT_BASE="${PWD}/output_dir/dataset_new6_hand" \
  GMR_OUT_ROOT="${PWD}/GMR-master/output/show_gmr/dataset_new6_g1_palm_roll_wrist_stable" \
  bash GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6.sh
```

## 批处理后检查

查看生成的视频：

```bash
find "${PWD}/GMR-master/output/show_gmr" \
  -name '*.mp4' -print
```

检查每个 clip 是否真的启用了 palm roll 和 wrist stabilization：

```bash
${HOME}/miniconda3/envs/locomotion/bin/python - <<'PY'
import pickle
from pathlib import Path

root = Path.cwd() / 'GMR-master/output/show_gmr'
for p in sorted(root.glob('*/*/robot_motion.pkl')):
    with p.open('rb') as f:
        data = pickle.load(f)
    print(p.parent.name)
    print('  palm_roll_applied:', data.get('palm_roll_applied'))
    print('  palm_roll_sides:', data.get('palm_roll_sides'))
    print('  wrist_pitch_yaw_stabilized:', data.get('wrist_pitch_yaw_stabilized'))
    print('  wrist_override:', data.get('hand_wrist_orientation_override_enabled'))
PY
```

## 已知限制

- G1 不能自然复现完整 MANO 手腕朝向，所以现在只保留 palm roll。
- 如果 HaMeR/GVHMR-hand 手部检测丢失，掌心翻转也会受影响。
- 手指细节精度受 MANO 到 G1 手指结构映射限制。
- 个别手肘异常可能来自 GVHMR body pose，本身不一定是 GMR wrist 逻辑造成的。
