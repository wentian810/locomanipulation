# 04. SMPL-H/SMPL-X 转换与相机导出

## 模块职责

此模块将内部的 GVHMR PyTorch 结果与最终 MANO 参数转换为稳定的 NumPy 接口。它一次输出两类互补数据：

1. 001_converted.npz：现有 Locomotion 和人体消费者兼容的 SMPL-H 身体轨迹。
2. 001_smplx_hands.npz：与身体同帧的 SMPL-X/MANO 手部 sidecar，保留手部姿态、关节、有效性与诊断信息。

随后从 GVHMR 结果导出 gvhmr_camera.npz，使相机位置、朝向、内参和人体轨迹处于同一约定下。

实现文件：

~~~text
GVHMR-hand/GVHMR-main/tools/pipeline/convert_to_npz.py
GVHMR-hand/GVHMR-main/tools/pipeline/export_gvhmr_camera.py
~~~

## 自动工作流

~~~text
hmr4d_results.pt + 最终 mano_params
  -> 优先读取 smpl_params_global
  -> 选定 person_idx=0，旋转矩阵转 axis-angle
  -> 写 001_converted.npz
  -> 按最终 MANO pose 生成 001_smplx_hands.npz
  -> 以 001_smoothed 的空间为参考导出 gvhmr_camera.npz
~~~

正常运行时该流程由 run_pipeline.sh 自动执行，执行次序在手部最终过滤之后、Locomotion 之前。不要先导出 sidecar 再替换 MANO 文件，否则身体预览、关节与 sidecar 可能来自不同版本。

## 手动恢复命令

以下命令只用于已有结果的恢复、schema 排障或单元测试。常规交付仍使用总入口。

~~~bash
PY_LOCO=/opt/conda/envs/locomotion/bin/python
ROOT=/path/to/Loco-manipulation-human-pipeline-support-contacts
CLIP_DIR=/data/human_output/<clip>

$PY_LOCO $ROOT/GVHMR-hand/GVHMR-main/tools/pipeline/convert_to_npz.py \
  --gvhmr_results $CLIP_DIR/gvhmr_out/<clip>/hmr4d_results.pt \
  --mano_params $CLIP_DIR/mano_params_direct_mano_recomputed.pt \
  --output $CLIP_DIR/001_converted.npz \
  --smplx_output $CLIP_DIR/001_smplx_hands.npz \
  --fps 30 --person_idx 0 --space global \
  --hand_backend hand4wholepp \
  --hand_refine_mode raw \
  --hand_reproj_error_thr 75 \
  --hand_reproj_error_ratio_thr 0.45 \
  --hand_wrist_offset_mode temporal
~~~

实际 MANO 输入文件由运行脚本按当前过滤链确定。如果 direct_mano 重算文件不存在，应先检查 pipeline log 和指纹，不应盲目改成最早的 mano_params.pt。

相机导出应在 001_smoothed.npz 已生成后运行：

~~~bash
$PY_LOCO $ROOT/GVHMR-hand/GVHMR-main/tools/pipeline/export_gvhmr_camera.py \
  --gvhmr_results $CLIP_DIR/gvhmr_out/<clip>/hmr4d_results.pt \
  --reference_npz $CLIP_DIR/001_smoothed.npz \
  --output $CLIP_DIR/gvhmr_camera.npz \
  --gravity_axis neg_y \
  --person_idx 0
~~~

### 转换/相机参数速查

| 参数 | 当前值 | 作用与禁止事项 |
|---|---:|---|
| `--gvhmr_results` | 当前 clip 的 `hmr4d_results.pt` | 只接受模块 02 同一 clip 的结果 |
| `--mano_params` | 当前最终 MANO 文件 | 必须与本次过滤链一致，不能任取最早 `mano_params.pt` |
| `--output/--smplx_output` | 001_converted / 001_smplx_hands | 两者必须在同一 `<clip>` 目录并成对更新 |
| `--fps` | 30 | 下游帧率契约；改动需要重采样全链路 |
| `--person_idx` | 0 | 当前只支持单人主体；不应用于多人视频 |
| `--space` | global | 输出 GVHMR 世界空间；不能和 Isaac/MuJoCo 坐标混用 |
| `--hand_backend` | hand4wholepp | 写入 sidecar 溯源字段；必须和 human.backend 一致 |
| `--hand_refine_mode` | raw | 当前最终 MANO 的细化标签；用于诊断溯源而非重新细化 |
| `--hand_reproj_error_thr/ratio_thr` | 75/0.45 | 标记异常手部证据的阈值；不应放宽来掩盖坏手 |
| `--hand_wrist_offset_mode` | temporal | 手腕偏移时序模式；改动需要重新检查身体-手连接 |
| 相机 `--reference_npz` | 001_smoothed.npz | 定义与 Locomotion 对齐的相机参考；不得跨 clip 使用 |
| `--gravity_axis` | neg_y | 写入相机坐标契约；必须和 Locomotion 的 y- 对应 |

## 输出文件 1：身体 NPZ

001_converted.npz 的稳定核心字段如下。转换器同时会写 poses、global_orient、transl、trans_original 等兼容别名，但新的对接代码应优先使用表中字段。

| 字段 | shape | 类型/单位 | 含义 |
|---|---:|---|---|
| root_orient | (T, 3) | float32，axis-angle 弧度 | 根节点全局朝向 |
| pose_body | (T, 63) | float32，axis-angle 弧度 | 21 个身体关节，不含根和手 |
| trans | (T, 3) | float32，米 | 根节点世界平移 |
| betas | (B,) | float32 | 静态体型参数 |
| gender | scalar | string | 模型性别，通常 neutral |
| mocap_frame_rate | scalar | float | 当前契约为 30 |

T 是视频帧数。body 轨迹仍是 GVHMR 原始值，下一模块会根据该文件生成优化/平滑后的 001_smoothed.npz。

## 输出文件 2：手部 sidecar

001_smplx_hands.npz 的必要字段如下：

| 字段 | shape | 含义 |
|---|---:|---|
| full_pose_smplx | (T, 165) | 根、身体、面部占位和双手组成的 SMPL-X axis-angle pose |
| left_hand_pose / right_hand_pose | (T, 45) | 每侧 15 个 MANO 手指关节旋转 |
| left_hand_global_orient / right_hand_global_orient | (T, 3) | 每侧腕/手全局朝向 |
| left_hand_joints_3d / right_hand_joints_3d | (T, 21, 3) | 每侧 21 个三维手关节 |
| left_hand_valid / right_hand_valid | (T,) | 最终轨迹在该帧是否可用 |
| hand_backend | scalar string | 产出该手的后端；当前为 hand4wholepp |
| root_orient、pose_body、trans、betas、mocap_frame_rate | 与身体 NPZ 同义 | 让 sidecar 可单独审计帧对齐 |

可选诊断字段包含：原始手部 pose/朝向/关节、bbox、重投影误差及其相对值、quality、source_reliable、source_repaired、bad_mask、spike_mask、手腕偏移统计。可选字段未出现时，消费者必须降级处理，不能假定所有老数据具有新诊断字段。

## 输出文件 3：相机 NPZ

gvhmr_camera.npz 主要包含：

| 字段 | shape | 用途 |
|---|---:|---|
| camera_pos_world | (T, 3) | 世界空间相机位置 |
| camera_target_world | (T, 3) | 世界空间相机观察目标 |
| camera_pos_isaac / camera_target_isaac | (T, 3) | 兼容 Isaac 的坐标副本；人体交付一般不消费 |
| subject_world | (T, 3) | 用于核对相机与人体相对位置 |
| T_w2c | (T, 4, 4) | world-to-camera 外参 |
| K_fullimg | (T, 3, 3) 或兼容 shape | 原图内参 |
| horizontal_fov_deg | scalar | 水平视场角 |
| world_to_isaac | (3, 3) | 坐标适配矩阵 |
| gravity_axis | scalar string | 当前为 neg_y |

相机文件的帧数必须与 001_smoothed.npz 相同。下游若不需要相机，仍建议随人体包交付，以便后续视觉重投影审核或再渲染。

## 坐标与数据语义

人体交付坐标系统为 gvhmr_world_gravity_negative_y：重力沿负 Y，平移单位为米，旋转为右手 axis-angle 弧度。它不是 MuJoCo、Isaac 或任意机器人坐标。对接方需在自己的边界层声明转换，并保留原始坐标系统字段。

转换器通过 rotation matrix 到 axis-angle 保留每帧旋转。不要对 axis-angle 三个分量直接线性平均来实现插值；模块 05 使用四元数连续化后再平滑。

## 故障边界

| 问题 | 判断 | 修复方向 |
|---|---|---|
| 结果中没有 smpl_params_global | 转换器会回退到兼容字段并打印来源 | 保留日志；若回退到 incam，必须在工单中标明 |
| body 与 sidecar 帧数不同 | 任何一个上游结果或缓存不一致 | 强制重跑手部前处理和转换，不能通过截断悄悄交付 |
| MANO pose 与 joints 不一致 | 未启用或未完成 direct MANO 重算 | 检查 GVHMR_RECOMPUTE_DIRECT_MANO=1 与最终 MANO 文件 |
| 相机 FOV/轨迹错误 | 使用了错误 clip 或 reference NPZ | 重新从同一 hmr4d_results.pt 和同一 clip 的 001_smoothed.npz 导出 |

此模块完成后，下游获得明确的 NPZ 契约；不需要读取 hmr4d_results.pt 或任何 PyTorch pickle。
