# 05. GMR、G1 与 Sharpa

## 目标

把人体身体轨迹重定向到 Unitree G1，并把 Hand4Whole++ 双手 sidecar 通过 Sharpa 22-DoF IK 生成独立手部轨迹，再用 MuJoCo 渲染机器人视频。

## 工作流

```text
final PHC body（失败时 001_smoothed.npz）
  -> SMPL-X/GMR kinematic model -> G1 whole-body IK / support-aware foot geometry
  -> robot_motion.pkl

001_smplx_hands.npz -> morphology-normalized finger targets
  -> Sharpa temporal IK / validity repair -> 001_sharpa_chain_hands.npz

body + Sharpa sidecar + GVHMR camera -> MuJoCo EGL render -> GMR MP4 + 2×2 MP4
```

当前生产选择由 `GMR_HAND_MODEL=sharpa` 和 `gmr.hand_model: sharpa` 固定为 G1 body + external Sharpa；G1 原生 Dex3 手不是当前交付路径。

## 启动方法

正式总入口：

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage human --clip-filter chairwood
```

在已有输出上单独调 GMR：

```bash
PIPELINE_ROOT="$PWD" \
SHOW_ROOT="$PWD/output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned" \
GMR_HAND_MODEL=sharpa GMR_SOURCE=final bash GMR-master/run_show_gmr_batch.sh
```

单独重算 Sharpa sidecar：

```bash
SHOW_ROOT="$PWD/output_dir/dataset_new6_hand4wholepp_directmano_gmr_sharpa_aligned" \
OUT_ROOT="$SHOW_ROOT" bash GMR-master/run_sharpa_hand_batch.sh
```

## 关键输入和输出

| 输入 | 输出 |
|---|---|
| `001_final.npz`/`001_phc*.npz`/`001_smoothed.npz` | `robot_motion.pkl` |
| `001_smplx_hands.npz` | `001_sharpa_chain_hands.npz`，左右 `(T,22)` qpos |
| G1 XML、SMPL-X、Sharpa XML/mesh、GVHMR camera | GMR MP4、2×2 MP4 |

最终导出器把可信 pickle 转成 `robot_motion.npz`，写入 `root_quat_xyzw`、`dof_position`、`dof_names`，商品包不保留 pickle。

## GMR source 语义

```text
final        -> PHC 成功的最终候选；失败时按 wrapper 保底
smoothed     -> 直接使用 001_smoothed.npz
phc_smoothed -> 强制使用 PHC 平滑候选
```

改 source 是数据语义变化，不是普通缓存开关。GMR 根坐标是 MuJoCo z-up，方向由 GVHMR 轴变换和 `human_yaw_offset_deg` 共同决定。

## 验收和故障

1. `robot_motion.pkl` 的帧数/FPS 与身体、手部、相机一致，根四元数归一化。
2. `dof_names` 数量等于 `dof_pos` 列数；Sharpa qpos 为 `(T,22)`。
3. Sharpa XML/mesh 找不到时检查镜像是否保留 `do-as-i-do-main/.../robots/sharpa`。
4. 左右手方向异常时先核对 left/right mount quaternion，不要先改人体坐标轴。
