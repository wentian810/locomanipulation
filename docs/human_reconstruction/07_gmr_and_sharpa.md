# 07. GMR 身体重定向与 Sharpa 手部 IK

## 模块职责

GMR 将人类 SMPL-H 身体轨迹重定向为机器人根位姿与关节角，并使用独立的 Sharpa Wave 22-DoF 双手链重定向 MANO 手势。当前正式 embodiment 是 Unitree G1 身体加外置 Sharpa 双手。

GMR 属于 human-only 流程，因为它只消费人体、手部和相机数据；没有 object.enabled 时不会读取任何物体/场景重建产物。

~~~yaml
gmr:
  enabled: true
  hand_model: sharpa
  source: final
  height_adjust_mode: support_aware_foot_geom
  camera_source: gvhmr
  render: true
  composite_2x2: true
~~~

## 输入契约

| 输入 | 用途 |
|---|---|
| 001_final.npz | 本次明确选出的身体来源；通过 final_motion_selection.json 可追溯到 PHC 或 smoothed |
| 001_smplx_hands.npz | MANO 手势、有效性、可靠度、手部质量和最终关节 |
| gvhmr_camera.npz | 机器人渲染使用与输入一致的 GVHMR 相机 |
| GMR robot XML/mesh | Unitree G1 body 与 Sharpa 外置手的运动学/渲染资产 |
| SMPL-X body model | 将人体动作与 GMR 的 IK 目标关联 |

发布运行时还必须具有 Unitree G1 网格包 `human-only-gmr-unitree-g1-assets_20260806.tar.zst`。它由 bootstrap 解压、由 release runner 只读挂载到 `GMR-master/assets/unitree_g1/`；其中缺少任意 STL 都可能让重定向成功但 MuJoCo 视频渲染失败。不要把网格提交到 Git 或让 GMR 在运行时改写资产目录；渲染临时 XML 写入容器 `/tmp`。

GMR source 是数据语义，不是渲染选项。当前 source=final，代表使用 PHC 成功时的 grounded 身体，或当前运行明示的 smoothed 回退。改成 smoothed、converted 或 phc_smoothed 后，必须重跑 GMR、质量和产品导出，并在资产的 source_body_stage 中反映变化。

## 参数速查

| 参数 | 当前默认/生产值 | 作用与变更后果 |
|---|---:|---|
| `gmr.enabled` | true | 是否产生机器人重定向；false 时 human-only 只交付人体，不可要求 robot component |
| `gmr.hand_model` | sharpa | Unitree G1 身体 + 外置 Sharpa；改为 g1/brainco 会改变机器人接口 schema |
| `gmr.source` | final | 读取 final selection；改动会改变人体来源，必须全量重新验收 |
| `gmr.target_fps` | 30 | 机器人目标频率；必须与输入重采样、质量门限同步考虑 |
| `height_adjust_mode` | support_aware_foot_geom | 支撑段地面高度策略；不是 PHC 的替代品 |
| `support_contact_height` | 0.08 m | 足底候选接触高度 |
| `support_max_vertical_speed` | 1.20 m/s | 快速上/下运动脚不视为支撑 |
| `support_min_contact_run/max_contact_gap` | 3/1 帧 | 支撑连续性和允许的小缺口 |
| `support_root_step_limit` | 0.03 m/frame | 支撑切换的 root 过渡限制；0 表示关闭 |
| `human_yaw_offset_deg` | 0.0 | 人体到机器人全局偏航；改动必须同步重渲染/重导出 |
| `camera_source` | gvhmr | GMR/Isaac 审核相机来源；需与 camera sidecar 同版 |
| `render/composite_2x2` | true/true | 是否输出 GMR 与四路审核视频；产品设置 require_preview 时不得关闭 |
| `mujoco_gl/render_width/render_height` | egl/960/720 | Headless 后端与渲染尺寸；仅影响审核视频，不改变数值 IK |

## 工作流

~~~text
001_final.npz + final_motion_selection.json
  -> 身体坐标与目标机器人映射
  -> support_aware_foot_geom 足部支撑高度调整
  -> GMR IK / 机器人 root 与 dof
  -> robot_motion.pkl

001_smplx_hands.npz
  -> MANO 指关节目标、valid/reliability 权重
  -> Sharpa 双手独立 IK 与时间约束
  -> 001_sharpa_chain_hands.npz

GMR 结果 + GVHMR/PHC 视频
  -> GMR robot render
  -> 2×2 对比视频
  -> 稳定 videos/ 别名
~~~

support_aware_foot_geom 根据左右脚支撑、竖直速度和连续帧约束处理地面高度，飞行段保留垂直运动。它不是全动力学仿真；真正的物理纠正属于 PHC。

Sharpa 的手腕与手指责任分离：GMR 的人体/SMPL 臂链负责肩、肘和手腕连接，Hand4Whole++/MANO 主要提供手指目标。这样可避免将 crop-camera 手腕朝向硬塞入机器人手臂 IK，造成身体与手掌不连续。

## 启动与续跑

GMR 由人类全流程的 batch wrapper 在所有 clip 的 GVHMR、Locomotion 和 PHC 完成后统一调用：

~~~bash
PYTHONUNBUFFERED=1 /opt/conda/envs/locomotion/bin/python \
  scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --clip-filter <clip>
~~~

没有独立的 --stage gmr。需要只重算 GMR 时，使用已有缓存并强制 GMR：

~~~bash
... --set resume.force_gmr=true --clip-filter <clip>
~~~

如果改变 GMR source、机器人 hand_model、support 参数、人体 yaw 或 Sharpa 成本函数，必须视为新的机器人版本。不要覆盖旧机器人结果；至少保留 pipeline_config.yaml、final_motion_selection.json、robot_motion.pkl 的 source_motion 与新旧渲染。

底层 GMR-master/run_show_gmr_batch.sh 可用于开发排障，但它要求完整环境变量、OUTPUT_ROOT、GMR_INCLUDE_CLIPS 与资产路径，不能作为外部固定接口。

## 输出

每条 clip 的核心输出：

~~~text
<out>/<clip>/
  robot_motion.pkl
  001_sharpa_chain_hands.npz
  <clip-slug>__gmr.mp4
  <clip-slug>__2x2.mp4 或 composite_2x2.mp4
  videos/
    <clip-slug>__gvhmr.mp4
    <clip-slug>__phc.mp4
    <clip-slug>__gmr.mp4
    <clip-slug>__2x2.mp4
~~~

robot_motion.pkl 是可信工作文件，可能包含 Python pickle，禁止直接对外发放。资产导出器会将机器人与 Sharpa 字段写入 `motion.npz` 的 `robot__*` 与 `sharpa__*` 命名空间；所有最终 NPZ 均可用 `allow_pickle=False` 打开。001_sharpa_chain_hands.npz 仍是工作区数值文件，不是对外接口。

## 机器人与手部数据语义

机器人工作轨迹至少包含：

| 字段 | shape | 语义 |
|---|---:|---|
| root_pos | (T, 3) | MuJoCo world，Z-up，米 |
| root_rot | (T, 4) | root 四元数，xyzw |
| dof_pos | (T, D) | 机器人关节角，弧度 |
| dof_names | (D,) | 新输出显式保存；旧文件由 XML 关节顺序恢复 |
| fps | scalar | 与人体同为 30 |
| source_motion | string | 当前 GMR 消费的身体文件，可追溯性检查使用 |

Sharpa sidecar 至少包含左右 (T,22) qpos、22 个 qpos 名、源手 valid 和可选 reliability。它不等价于 G1 原生手，也不能被当作人体 MANO pose。

## 审核与质量边界

审核视频需要检查：

1. 人体根方向、机器人行进方向和 GMR 相机是否整体一致。
2. 脚在支撑段没有持续漂浮或地面穿透；快速腾空不能被强行压地。
3. 机器人关节没有突跳、机械限位附近抖动或异常反向。
4. Sharpa 左右手是否正确挂载，手指是否与 MANO 手势大致一致，低可靠度段是否不过度追随噪声。

手部的“跟随”不代表机器人可以真实抓取，GMR 这里是运动学重定向和 MuJoCo 视觉审核，不是接触力验证。

## 常见问题

| 现象 | 首先检查 | 处理 |
|---|---|---|
| GMR 用了旧 PHC 结果 | final_motion_selection.json 与 robot_motion.pkl 的 source_motion | 重新执行当前 run；不要在目录里手换 symlink |
| 机器人整体偏航 180 度 | gmr.human_yaw_offset_deg 与 render | 显式设置并让人体、相机、任何后续物体桥使用同一值 |
| 脚抖或漂浮 | PHC 输出、support 参数、人体根轨迹 | 先查模块 05/06；不要用渲染偏移掩盖 |
| 手腕扭曲但手指可用 | crop-camera wrist 与人体臂链坐标差异 | 维持 diagnostic wrist 模式，检查 palm-roll 与 SMPL wrist 连接 |
| Sharpa 缺文件或帧数不一致 | 001_smplx_hands.npz、valid/reliability、GMR log | 重新运行 GMR；不能用复制的旧 Sharpa 文件补齐 |
| headless 渲染失败 | EGL/MuJoCo/驱动 | 数值 pkl 仍须验证；产品要求 preview 时会阻止导出 |

不包含场景/物体时，2×2 面板固定表达原视频、GVHMR、PHC/Isaac 与 GMR 四路人体结果；不应出现任何物体轨迹或网格。
