# 管线工程与数据协议

更新日期：2026-07-03

本文定义模块边界、坐标系、最终 NPZ 协议、质量门禁和文件保留规则。执行命令见
[PIPELINE_README.md](PIPELINE_README.md)，YAML 参数见
[configs/README.md](configs/README.md)。

## 1. 总体架构

```text
输入 RGB 视频
  |
  +-- 人体：人物框 -> ViTPose -> GVHMR -> SMPL-H
  |
  +-- 双手：Hand4Whole++ -> MANO -> temporal/finger filters
  |
  +-- 身体后处理：Locomotion -> 平滑 -> PHC
  |
  +-- 机器人：GMR -> H1 + Sharpa -> MuJoCo 渲染
  |
  +-- 可选物体：
  |     近人体 YOLO -> SAM-HQ -> Cutie
  |     -> Hunyuan3D -> MoGe-2 米制尺度 -> MegaPose 6D
  |     -> GVHMR 世界 -> GMR z-up 世界
  |
  +-- Original / GVHMR / PHC / GMR 2×2
  |
  +-- 无 GT 质量分层 -> 安全资产导出 -> manifest -> SHA-256 -> 可选清理 scratch
```

### 1.1 模块与生产者

| 模块 | 主要生产者 |
| --- | --- |
| YAML 启动与检查 | `scripts/run_pipeline_from_config.py` |
| 人体/手批处理 | `GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6_*.sh` |
| SMPL/手 sidecar | `GVHMR-hand/GVHMR-main/tools/pipeline/convert_to_npz.py` |
| GVHMR 相机 | `GVHMR-hand/GVHMR-main/tools/pipeline/export_gvhmr_camera.py` |
| Locomotion/PHC 调度 | `GVHMR-hand/GVHMR-main/tools/pipeline/run_pipeline.sh` |
| GMR | `GMR-master/scripts/smpl_npz_to_robot_headless.py` |
| Sharpa IK | `GMR-master/scripts/sharpa_hand_retarget.py` |
| 自动目标检测 | `do-as-i-do-main/reconstruction/scripts/detect_object_near_person.py` |
| 分割与传播 | `run_samhq_cutie_video.py` |
| 单目物体总调度 | `do-as-i-do-main/reconstruction/run_pipeline.sh` |
| 人体/物体坐标桥 | `GMR-master/scripts/import_foundationpose_object.py` |
| 物体 QA | `validate_object_adapter.py`、`validate_object_projection.py` |
| 2×2 | `GMR-master/scripts/render_composite_2x2.py` |
| 无 GT 质量分层 | `scripts/evaluate_clip_quality.py` |
| 最终资产 | `scripts/export_dataset_product.py` |

## 2. 三类目录

三类数据必须物理分开：

| 类型 | 典型位置 | 生命周期 |
| --- | --- | --- |
| 输入 | `dataset_new6/` | 原始数据，不由管线删除 |
| 工作区 | `output_dir/` 或 `scratch/`、`object_work/` | 可续跑；批处理可清理 |
| 最终资产 | `assets/` | 长期保存、校验、交付 |

正式批处理只允许把 `scratch/<dataset>/<clip>` 当作可删除工作区。最终资产不能位于
该 clip 工作目录之内。

## 3. 阶段状态

| 阶段 | 初始数据 | 成功输出状态 |
| --- | --- | --- |
| Human | 视频 | 身体、手、相机、Locomotion、PHC、GMR 与 2×2 完成 |
| Object | 完整 Human 输出 + 视频 | 米制网格、相机/GMR 轨迹、QA、GVHMR/GMR 重渲染完成 |
| Quality | 完整 Human 或 Object 输出 | 每 clip 分层报告及全数据集 CSV/JSONL 汇总 |
| Product | 完整工作输出 | 压缩 NPZ、预览、清单、权利状态和校验和完成 |

`--stage object` 是追加操作，不会重建人体。`--stage quality` 只读已有结果。
`--stage product` 不运行模型；启用商品质量门禁时会先执行同样的只读评价。

默认 `gmr.source: smoothed`。PHC 虽然在 Human 阶段执行，但 GMR 不自动读取 PHC
轨迹；改为 `phc_smoothed` 是一次数据语义变化，不是简单缓存开关。

## 4. 工作文件

### 4.1 人体

`001_converted.npz`、`001_smoothed.npz` 和 PHC NPZ 使用相同身体核心：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `root_orient` | `(T,3)` | 根 axis-angle，弧度 |
| `pose_body` | `(T,63)` | 21 个身体关节 axis-angle |
| `trans` | `(T,3)` | 根平移，米 |
| `betas` | `(16,)` | 身形 |
| `gender` | 标量字符串 | SMPL 性别 |
| `mocap_frame_rate` | 标量 | FPS |

`001_smplx_hands.npz` 是同步手部 sidecar，包含左右 MANO 45 维 pose、有效帧、质量、
修复标记、bbox 和重投影误差。原始与诊断副本只用于工作区，不进入最终资产。

### 4.2 机器人

工作文件 `robot_motion.pkl` 由本工程生成并只在可信工作区加载：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `root_pos` | `(T,3)` | MuJoCo z-up 根位置 |
| `root_rot` | `(T,4)` | 根四元数，`xyzw` |
| `dof_pos` | `(T,D)` | 机器人关节角 |
| `dof_names` | `(D,)` | 新输出显式保存列名 |
| `fps` | 标量 | 机器人 FPS |
| `human_yaw_offset_deg` | 标量 | 人体到机器人世界的 yaw |

旧 PKL 没有 `dof_names` 时，资产导出器从对应机器人 XML 的 worldbody 关节顺序恢复，
并验证数量等于 `D`。

Sharpa 独立输出 `001_sharpa_chain_hands.npz`，每只手使用 `(T,22)` qpos，并携带
22 个关节名、有效性和可靠度。

### 4.3 相机

`gvhmr_camera.npz`：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `T_w2c` | `(T,4,4)` | GVHMR 世界到 OpenCV 相机 |
| `K_fullimg` | `(T,3,3)` | 完整图像内参 |
| `subject_world` | `(T,3)` | GVHMR 世界中的人体参考根 |
| `camera_pos_world` | `(T,3)` | GVHMR 世界相机位置 |
| `world_to_isaac` | `(3,3)` | GVHMR 世界到 z-up 的轴变换 |
| `alignment_offset_world` | `(3,)` | Locomotion 与 GVHMR 的对齐偏移 |
| `gravity_axis` | 字符串 | 当前为 `neg_y` |

### 4.4 物体

统一单目适配器至少包含：

```text
monocular_adapter/
  mesh/mesh.obj
  pose.npy                 # (To,4,4), T_camera_object
  pose_valid.npy
  pose_observed.npy
  cam_K.json
  adapter_manifest.json
```

桥接后工作区有：

```text
object_reconstruction/
  object_motion_gvhmr.npz
  object_motion_gmr.npz
  object_manifest.json
  object_adapter_validation.json
  collision/
```

`object_unavailable.json` 的存在优先表示当前物体阶段失败。即使目录里残留旧 mesh 或
旧 NPZ，产品导出器也不会将其视为有效物体。

## 5. 最终资产协议

所有最终 NPZ：

- 使用 `np.savez_compressed`；
- 禁止 `dtype=object`；
- 必须能以 `allow_pickle=False` 读取；
- 数值必须有限；
- 路径必须是资产包内相对路径，不泄露本机绝对路径；
- 字段名直接包含四元数顺序。

### 5.1 `human_motion.npz`

| 字段 | 形状 | 坐标/单位 |
| --- | --- | --- |
| `root_orient_axis_angle` | `(T,3)` | GVHMR world，弧度 |
| `body_pose_axis_angle` | `(T,63)` | 局部关节 axis-angle，弧度 |
| `translation` | `(T,3)` | GVHMR world，米，重力 `-Y` |
| `betas` | `(16,)` | SMPL-H shape |
| `left/right_hand_pose_axis_angle` | `(T,45)` | MANO 局部关节，弧度 |
| `left/right_hand_valid` | `(T,)` | 最终有效性 |
| `left/right_hand_quality` | `(T,)` | `[0,1]` 软质量 |
| `left/right_hand_*mask` | `(T,)` | 异常、修复信息 |
| `left/right_hand_bbox_xyxy` | `(T,4)` | 像素；缺失值统一为 `-1` |

元数据字段明确写出：

```text
coordinate_system = gvhmr_world_gravity_negative_y
rotation_representation = axis_angle_radians
units = meter_radian
source_body_stage = smoothed / converted / phc_smoothed
```

### 5.2 `human_phc_motion.npz`

若 `phc.enabled` 与 `product.include_phc_motion` 同时为真，还会输出
`human_phc_motion.npz`。其身体字段与 `human_motion.npz` 相同，但
`source_body_stage = phc_smoothed`，不重复保存 MANO 手部字段。

### 5.3 `robot_motion.npz`

| 字段 | 形状 | 坐标/单位 |
| --- | --- | --- |
| `root_position` | `(T,3)` | MuJoCo world z-up，米 |
| `root_quat_xyzw` | `(T,4)` | 单位四元数，`xyzw` |
| `dof_position` | `(T,D)` | 弧度 |
| `dof_names` | `(D,)` | 与列严格一一对应 |
| `embodiment` | 字符串 | 如 `unitree_h1_with_hand` |

### 5.4 `robot_hand_motion.npz`

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `left/right_qpos` | `(T,22)` | Sharpa 两手关节角 |
| `left/right_qpos_names` | `(22,)` | 关节列名 |
| `left/right_valid` | `(T,)` | 源手有效性 |
| `left/right_reliability` | `(T,)` | 软可靠度 |

### 5.5 `camera.npz`

保留第 4.3 节全部相机/人体参考字段，并补充：

```text
world_coordinate_system = gvhmr_world_gravity_negative_y
render_coordinate_system = mujoco_world_z_up
camera_convention = opencv_x_right_y_down_z_forward
fps / frame_count / units
```

### 5.6 `object_motion.npz`

一个文件同时承载 GVHMR 与 GMR 两套视图所需轨迹：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `camera_pose_object` | `(To,4,4)` | `T_camera_object` |
| `gvhmr_world_pose_object` | `(To,4,4)` | `T_world_object` |
| `camera_valid` | `(To,)` | 相机轨迹有效性 |
| `camera_intrinsics` | `(3,3)` 或 `(To,3,3)` | OpenCV K |
| `robot_position` | `(T,3)` | GMR/MuJoCo z-up 位置 |
| `robot_quat_wxyz` | `(T,4)` | 物体方向，`wxyz` |
| `robot_valid` | `(T,)` | GMR 帧有效性 |
| `source_frame_index` | `(T,)` | 对应相机物体帧 |
| `visual_mesh_path` | 字符串 | 包内相对路径 |
| `collision_mesh_paths` | `(C,)` | 包内相对路径 |

机器人根四元数是 `xyzw`，物体四元数是 `wxyz`。这是现有渲染/仿真接口的不同约定，
不能靠统一命名掩盖；最终字段名已显式区分。

## 6. 坐标链

### 6.1 坐标定义

| 名称 | 约定 |
| --- | --- |
| OpenCV camera | `+X` 右、`+Y` 下、`+Z` 前 |
| GVHMR world | 相机恢复世界，当前重力轴 `-Y` |
| MuJoCo/GMR world | `+Z` 上 |
| `T_w2c` | 世界点变换到相机 |
| `T_c_o` | 物体局部点变换到相机 |

### 6.2 相机物体到 GVHMR 世界

列向量齐次变换定义：

```text
T_w_o = inverse(T_w2c) @ T_c_o
T_w_o.translation += alignment_offset_world
```

因此 `pose.npy` 必须是 `T_camera_object`，不能传入其逆矩阵。

### 6.3 GVHMR 世界到机器人世界

代码内部数组使用行向量：

```text
object_zup  = object_world  @ world_to_isaac.T
subject_zup = subject_world @ world_to_isaac.T
relative    = object_zup - subject_zup
position_robot = robot_root + relative @ yaw.T + robot_offset
```

方向：

```text
R_robot_object = yaw @ world_to_isaac @ R_world_object
```

`yaw` 来自 `gmr.human_yaw_offset_deg`。人体 GMR 和物体桥接必须使用同一个值；当前
默认 `0°`。如果某批数据确实需要整体转向，再显式设为 `180°`。

物体位置保留“物体相对人体参考根”的偏移，再加到机器人根周围。它不会把物体硬绑定
到机器人手，因此手没有真正接触物体时，GMR 面板会如实暴露误差。

### 6.4 FPS 映射

物体源帧与机器人帧可能不同：

```text
source_position = (robot_frame - frame_offset) * source_fps / robot_fps
source_frame_index = round(source_position)
```

映射后要求索引单调且在 `[0, To-1]` 内。边界外帧标记无效，不循环复用。

## 7. 质量门禁

质量分两层：7.1–7.3 是资产协议的硬校验；7.4 是无 GT 统计评价。硬校验失败不能导出，
统计评价输出 `pass | warn | fail`，用于批量筛坏片和人工复核排序。

### 7.1 人体/机器人

- `T > 0` 且所有时间数组首维一致；
- human、robot、Sharpa、camera 的 FPS 一致；
- `root_quat_xyzw` 范数误差不超过 `1e-3`；
- `dof_names` 数量等于 `dof_position.shape[1]`；
- 手部 pose、valid、quality 与人体帧数一致；
- 所有交付数值有限。

### 7.2 物体

- 不存在当前失败状态 `object_unavailable.json`；
- `object_adapter_validation.json.ok == true` 且无 errors；
- mesh 尺寸位于 clip 的 `expected_size_m`；
- 有效率/观测率达到 YAML 阈值；
- `T_camera_object` 为有限刚体变换，物体深度为正；
- GMR 物体帧数与 robot motion 一致；
- `robot_quat_wxyz` 在有效帧为单位四元数；
- 坐标标签必须分别为 `opencv_camera` 与 `mujoco_robot_world_zup`。

自动分割还包含近人体硬约束。错误墙体即使 Cutie 跟踪很稳定，也会因 ROI 或尺寸门禁
失败，不能以“时序一致”替代“目标正确”。

### 7.3 资产完整性

导出顺序：

```text
临时目录
  -> 写压缩 NPZ
  -> allow_pickle=False 重开
  -> 写配置快照和 rights
  -> 计算 payload SHA-256
  -> 写 manifest
  -> 写并复核 checksums.sha256
  -> 原子替换最终 clip 目录
```

任一步失败都删除临时导出目录并保留完整工作区。

### 7.4 无 GT 统计指标

`scripts/evaluate_clip_quality.py` 只读取可信工作产物，输出：

```text
output.root/<clip>/quality_report.json
output.root/quality_summary.csv
output.root/quality_summary.jsonl
```

| 类别 | 指标与来源 |
| --- | --- |
| 文件 | 必需文件、安全读取、各模态帧数匹配率 |
| 身体 | `trans` 差分乘 FPS 得到根速度 P95 |
| 手 | sidecar 中 valid、相对重投影误差 P90、spike、repaired |
| GMR | 根四元数范数、机器人 XML 关节限位违反率 |
| 物体 | valid、相邻位置/四元数跳变、网格投影框与分割框 IoU |
| 接触 | GMR 正运动学手腕到运动物体 OBB 的距离代理 |
| 动态 | `object_dynamic_sim.json` 的抬起、爆炸和力异常状态 |
| 可视化 | 2×2 可读性及与人体帧数一致率 |

物体姿态跳变同时检查平移和旋转，四元数用绝对内积处理 `q/-q` 等价。投影指标沿
`mesh local -> T_camera_object -> OpenCV K` 计算，因此也会暴露位姿方向或内参错配。
接触率不是碰撞引擎接触力：大物体 OBB 会高估接触，只能作为筛查代理；严格验收应启用
MuJoCo 动态验证并保留少量人工抽检。

分层优先于分数。文件、人体、手、GMR 或物体阶段失败进入
`critical_fail_reasons`，整条为 `fail`；接触和非强制动态失败进入 `warn_reasons`。
加权分仅用于同一分层内排序。

## 8. 保留与清理

`product.retention.prune_workspace_after_export: true` 时：

1. 再次验证最终 `manifest.json` 和 `checksums.sha256`；
2. 再次以 `allow_pickle=False` 打开所有 NPZ；
3. 确认资产目录不在 clip 工作目录内；
4. 只允许删除 `output.root` 的直接子目录 `<clip>`；
5. 可选删除 `object.monocular.work_root` 的直接子目录 `<clip>`；
6. 在 `output.root/_exported/<clip>.json` 写出校验和清理记录。

没有“按通配符清空 output root”的逻辑。默认配置和调试 YAML 均不清理。

每次批量导出结束（即使中途某条失败）都会从已完成的 per-clip manifest 原子重建
`product.root/catalog.jsonl`。索引仅保存相对 bundle 名，不替代单条 SHA-256 校验。

### 8.1 最小保留集

| 文件 | 保留原因 |
| --- | --- |
| `human_motion.npz` | 标准人体 + MANO 动作 |
| `human_phc_motion.npz` | PHC 最终修复身体轨迹 |
| `robot_motion.npz` | 机器人主体动作 |
| `robot_hand_motion.npz` | Sharpa 双手动作 |
| `camera.npz` | 重投影、视角和坐标复现 |
| `object_motion.npz` + mesh | 物体 6D 与可视/碰撞资产 |
| `quality_report.json` | 自动分层、指标明细和拒绝原因 |
| `preview_2x2.mp4` | 人工验收和商品预览 |
| `manifest/config/rights/checksums` | 语义、复现、授权和完整性 |

不保留逐帧 mask、MoGe depth、Hunyuan 原始返回、模型 feature、诊断视频、PHC/GMR
pickle 和重复身体 NPZ。

## 9. 安全与商业许可

- `.pkl` 和 `.pt` 可能触发 Python 反序列化，只能加载本工程可信输出；
- 商品包不包含 pickle、checkpoint、SMPL/MANO 模型或密钥；
- `pipeline_config.yaml` 排除 runtime、输入/输出绝对路径和物体 work path；
- `visual_mesh_path` 和碰撞路径被改写为包内相对路径；
- YAML 中出现 secret 字段会被启动器拒绝。

`rights.json.commercial_ready` 只有以下四项全部为真时才为真：

```text
输入视频商业权利
人物/受试者授权
GVHMR 商业许可
其余第三方模型与资产已核验
```

它是交付门禁信息，不是法律意见，也不能替代授权文件。当前默认全部为 `false`。

## 10. 验证命令

```bash
# 配置和依赖检查
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --check --print-config

# 已有工作结果导出
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage product --clip-filter chairwood

# 只重算无 GT 质量报告
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage quality --clip-filter chairwood

# 单元测试
/home/zun/miniconda3/envs/locomotion/bin/python -m pytest -q \
  tests/test_product_export.py \
  tests/test_object_coordinate_contract.py \
  tests/test_rgb_object_frontend.py \
  tests/test_clip_quality.py
```
