# 09. human-only 输出协议与接收验收

## 目的

本文件定义对接方可依赖的最终接口。工作区中的 PT、PKL、缓存、临时视频和中间 NPZ 只能由本工程读取；对外仅以模块 08 生成的资产包为准。

每个资产包的机器契约是：

~~~text
<asset>/<clip>/
  motion.npz                # 必需：human__/robot__/sharpa__/camera__ 命名空间
  quality_report.json
  final_motion_selection.json
  preview_2x2.mp4
  manifest.json
  pipeline_config.yaml
  checksums.sha256
~~~

所有 NPZ 必须能以 numpy.load(..., allow_pickle=False) 读取，禁止 dtype=object，数值字段不得包含 NaN 或 Inf。

## 单一 motion.npz 与人体命名空间

当前导出器只写一个 `motion.npz`，而不是多个 human/robot/camera 文件。顶层 `schema_version`、`format` 与 `components` 描述容器；`components` 必须为 `[human, robot, sharpa, camera]`（顺序不构成语义）。人体和 MANO 字段统一以 `human__` 开头：

| 字段 | shape | 坐标/单位 | 说明 |
|---|---:|---|---|
| `human__schema_version` | scalar | int | human component schema |
| `human__fps` / `human__frame_count` | scalar | 30 / T | 全部人体时间数组的基准 |
| `human__coordinate_system` | scalar string | `gvhmr_world_gravity_negative_y` | 世界重力为 -Y |
| `human__rotation_representation` | scalar string | `axis_angle_radians` | 旋转表示 |
| `human__units` | scalar string | `meter_radian` | 位置和角度单位 |
| `human__source_body_stage` | scalar string | smoothed、phc 等 | 本次导出的身体来源 |
| `human__root_orient_axis_angle` | (T,3) | 弧度 | 人体根朝向 |
| `human__body_pose_axis_angle` | (T,63) | 弧度 | 21 个身体关节 |
| `human__translation` | (T,3) | 米 | GVHMR 世界根位置 |
| `human__betas` | (B,) | 无量纲 | SMPL-H shape |
| `human__gender` | scalar string | 通常 neutral | 模型元数据 |
| `human__left/right_hand_pose_axis_angle` | (T,45) | 弧度 | 每只手 15 个 MANO 指关节 |
| `human__left/right_hand_valid` | (T,) | bool | 最终手部轨迹可用性 |
| `human__hand_backend` | scalar string | hand4wholepp | 手部来源 |

可选的质量字段包括 left/right_hand_quality、source_reliable、source_repaired、bad_mask、spike_mask、bbox_xyxy 与 reproj_error。可选字段不存在时，消费者应保守降级；不能伪造为全有效。

PHC 不是单独的 `human_phc_motion.npz`。成功时 `human__source_body_stage` 必须与 `final_motion_selection.json.selected_stage` 一致，例如 `phc_smoothed_grounded`；若为 `smoothed`，接收方必须把它视为 PHC 回退并读取质量报告的 warning。

## robot__/sharpa__ 命名空间

`motion.npz` 中的机器人字段：

| 字段 | shape | 坐标/单位 |
|---|---:|---|
| `robot__root_position` | (T,3) | MuJoCo world，Z-up，米 |
| `robot__root_quat_xyzw` | (T,4) | 单位四元数，xyzw |
| `robot__dof_position` | (T,D) | 弧度 |
| `robot__dof_names` | (D,) | 与 dof_position 列严格一一对应 |
| `robot__embodiment` | scalar string | 如 unitree_g1 |
| `robot__fps` / `robot__frame_count` | scalar | 应与人体一致 |

`motion.npz` 中的 Sharpa 字段：

| 字段 | shape | 说明 |
|---|---:|---|
| `sharpa__left_qpos` / `sharpa__right_qpos` | (T,22) | Sharpa 两只手的 22-DoF 关节角 |
| `sharpa__left_qpos_names` / `sharpa__right_qpos_names` | (22,) | 对应关节列名 |
| `sharpa__left_valid` / `sharpa__right_valid` | (T,) | 源 MANO 手的有效性 |
| `sharpa__left_reliability` / `sharpa__right_reliability` | (T,) | 软可靠度 |
| `sharpa__embodiment` | scalar string | sharpa_dual_hand |

机器人根四元数使用 xyzw。不得与物体或其他系统常见的 wxyz 顺序混用。

## camera__ 命名空间

`motion.npz` 的 camera namespace 保留 GVHMR 世界、OpenCV 相机和渲染世界的关系：

| 字段 | shape | 说明 |
|---|---:|---|
| `camera__world_coordinate_system` | scalar string | gvhmr_world_gravity_negative_y |
| `camera__render_coordinate_system` | scalar string | mujoco_world_z_up |
| `camera__camera_convention` | scalar string | opencv_x_right_y_down_z_forward |
| `camera__camera_pos_world` / `camera__camera_target_world` | (T,3) | GVHMR 世界相机位置与目标 |
| `camera__camera_pos_isaac` / `camera__camera_target_isaac` | (T,3) | Z-up 渲染相关副本 |
| `camera__subject_world` / `camera__subject_isaac` | (T,3) | 人体参考点 |
| `camera__T_w2c` | (T,4,4) | 世界到相机外参 |
| `camera__K_fullimg` | (T,3,3) | OpenCV 内参 |
| `camera__world_to_isaac` | (3,3) | 坐标轴转换 |
| `camera__alignment_offset_world` | (3,) | Locomotion 对齐偏移 |
| `camera__gravity_axis` | scalar string | neg_y |

不能将 human_motion.translation 直接当成 MuJoCo root_position。人体、相机、机器人有明确不同坐标契约，任何转换都应在消费者自己的适配层完成并写入版本记录。

## 发送方验收清单

### 结构与完整性

1. 对 checksums.sha256 执行 sha256sum -c，全部成功。
2. manifest.json 的 validation.status 为 passed，clip、frame_count、fps 与实际文件一致。
3. 所有必需 NPZ 可用 allow_pickle=False 打开，且无 object dtype、NaN 或 Inf。
4. human、robot、Sharpa、camera 以及 preview 的帧数一致，或至少达到质量阈值中的匹配比。
5. 机器人 root 四元数范数接近 1，dof_names 数量等于 dof_position 的列数。

### 数据语义

1. final_motion_selection.json 的 selected_file 与 manifest 的 source_body_stage 一致。
2. PHC 已启用时，若 final 回退到 smoothed，quality_report 必须有对应 warn，不能宣称 PHC 成功。
3. hand_valid 和 reliability 在遮挡段应反映不确定性，不能因为插值而全部标 true。
4. 坐标系统、旋转表示、四元数顺序和单位必须随资产一起传递。

### 人工审核

1. 1_incam 或预览中的 GVHMR 人体与原视频主体一致。
2. 手部没有明显左右互换、腕部翻转或单帧手指爆跳。
3. PHC 面板无持续地面穿透、无异常爆炸或失去运动。
4. GMR 面板的根朝向、支撑脚和关节运动合理，Sharpa 左右手挂载正确。
5. 2×2 不应包含任何场景/物体模块内容；本交付范围仅为 human-only。

## 接收方最低验证命令

~~~bash
cd <asset>/<clip>
sha256sum -c checksums.sha256

/opt/conda/envs/locomotion/bin/python - <<'PY'
from pathlib import Path
import numpy as np

root = Path(".")
for path in sorted(root.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            value = np.asarray(data[name])
            if value.dtype == object:
                raise RuntimeError(f"{path}: forbidden object dtype in {name}")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise RuntimeError(f"{path}: non-finite numeric field {name}")
    print("OK", path)
PY
~~~

该脚本只做安全读取与有限数检查，不能替代质量报告或人工运动审核。

## 问题单必须附带的材料

当接收方报告问题，请同时提供：

1. 资产包 clip 名、manifest.json、quality_report.json、final_motion_selection.json。
2. 发生问题的文件名、字段、帧号和预期/实际值。
3. preview_2x2.mp4 的时间戳或截图；手部问题附手部诊断视频时间戳。
4. pipeline_config.yaml、镜像 tag、GPU/驱动与运行命令。
5. 若问题发生在生产工作区，提供对应 <clip>.pipeline.log；不要发送模型 checkpoint、密钥或不必要原视频。

这样可以将问题定位到输入准入、GVHMR、手部、Locomotion、PHC、GMR、导出或消费者坐标适配中的一个明确边界。
