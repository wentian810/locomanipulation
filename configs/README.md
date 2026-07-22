# YAML 配置说明

所有正式运行由两层 YAML 合并：

```text
configs/default.yaml
  + configs/pipelines/<pipeline>.yaml
  + 进程环境变量
  + CLI
```

优先级：`CLI > 环境变量 > pipeline YAML > default.yaml`。密钥禁止写入 YAML，
只能放在被 Git 忽略的 `configs/secrets.env` 并由 shell 加载。

## 1. 顶层与输入输出

| 参数 | 中文含义 |
| --- | --- |
| `schema_version` | 配置协议版本，当前只能为 `1` |
| `project_root` | 工程根目录；相对默认 YAML 解析 |
| `input.dataset_dir` | 原始视频目录 |
| `input.extensions` | 扫描的视频扩展名 |
| `input.clip_filter` | 文件名子串过滤；空字符串表示全部 |
| `input.max_videos` | 在文件名排序和 `clip_filter` 后只取前 N 条；适合小批量验证 |
| `input.fullbody_preflight.enabled` | 在昂贵的人体流程前启用 YOLO-Pose 全身准入检查 |
| `input.fullbody_preflight.mode` | `gate` 排除不合格视频；`report` 仅记录、不拦截 |
| `input.fullbody_preflight.model` | 本地 YOLO-Pose 权重；默认 `models/yolo11n-pose.pt` |
| `input.fullbody_preflight.samples` | 每个视频均匀抽样的帧数，默认 32 |
| `input.fullbody_preflight.min_*_ratio` | 主体、头部、左右手腕、左右脚踝与完整全身证据的最小覆盖率 |
| `input.work_video.enabled` | 是否生成统一尺寸工作视频 |
| `input.work_video.directory` | 工作视频缓存目录 |
| `input.work_video.width/height` | 工作视频目标尺寸 |
| `input.work_video.crf` | H.264 质量；越小质量越高、文件越大 |
| `input.work_video.force` | 是否强制重新编码 |
| `output.root` | 过程工作区；批处理配置应放在 `scratch/` |

全身准入不是普通 YOLO 人框筛选：它在同一主体的时间采样中要求头部、左右手腕和左右脚踝证据。结果集中写到 `output.root/fullbody_preflight.csv` 与 `fullbody_preflight.jsonl`；被 `gate` 排除的片段不会进入 GVHMR、手部、PHC 或 GMR。

## 2. 最终资产与保留策略

| 参数 | 中文含义 |
| --- | --- |
| `product.enabled` | 是否启用最终资产导出 |
| `product.root` | 永久资产根目录，必须与 clip 工作区分离 |
| `product.export_after_stages` | 哪些 stage 完成后自动导出，如 `[all]` |
| `product.include_camera` | 是否交付 `camera.npz` |
| `product.include_phc_motion` | PHC 启用时是否交付最终 PHC 身体轨迹 |
| `product.include_preview` | 是否交付 2×2 视频 |
| `product.require_preview` | 预览缺失是否使导出失败 |
| `product.object_policy` | `exclude`、`if_valid` 或 `require_valid` |
| `product.minimum_quality_status` | 商品最低自动质量结论：`warn` 或 `pass`；`fail` 永不导出 |
| `retention.prune_workspace_after_export` | 校验成功后删除 `output.root/<clip>` |
| `retention.prune_object_work_after_export` | 同时删除 `object.monocular.work_root/<clip>` |

## 3. 自动质量评价

| 参数 | 中文含义 |
| --- | --- |
| `quality_evaluation.enabled` | 是否生成每 clip 的无 GT 质量报告 |
| `quality_evaluation.run_after_stages` | 哪些 stage 后自动评价 |
| `quality_evaluation.require_for_product` | 商品导出前是否必须存在当次质量报告 |
| `quality_evaluation.pass_score` | 无警告时达到 `pass` 的最低加权分 |
| `quality_evaluation.projection_samples` | 物体网格投影 IoU 的均匀抽样帧数 |
| `weights.human_only.*` | 人体-only 的文件/身体/手/GMR/可视化权重 |
| `weights.object.*` | 含物体的文件/身体/手/物体/接触/动态权重 |

每次评价后，根目录只生成紧凑的 `quality_overview.csv` / `quality_overview.json`：

- `score`：按上述权重得到的 0–100 管线质量分（PQI）；
- `grade`：A–E 的便于记录等级；
- `verdict`：`accept`、`review` 或 `reject`；
- `weakest_stage` 和 `stage_overview`：直接指出最弱环节及每阶段的总览。

每个 clip 的 `quality_report.json` 仍保存底层有效帧、重投影、四元数、关节范围等证据；它们不再被重复铺到批处理总表中。

常用阈值：

| 参数 | 中文含义 |
| --- | --- |
| `frame_count_warn/fail_ratio` | 各模态帧数最小/最大值之比 |
| `hand_valid_warn/fail_ratio` | 左右手有效帧率下限 |
| `hand_reproj_p90_warn/fail_relative` | 手部相对人体框尺度的重投影误差 P90 上限 |
| `hand_spike_warn/fail_ratio` | 手部时序尖峰率上限 |
| `hand_repaired_warn/fail_ratio` | 手部被自动修复帧率上限 |
| `root_speed_p95_warn/fail_mps` | 人体根平移速度 P95 上限，单位 m/s |
| `joint_limit_violation_warn/fail_ratio` | GMR 关节越限元素比例上限 |
| `object_valid_warn/fail_ratio` | 物体 6D 有效帧率下限 |
| `object_position_jump_m` | 相邻有效帧平移跳变阈值，单位米 |
| `object_rotation_jump_deg` | 相邻有效帧旋转跳变阈值，单位度 |
| `object_jump_warn/fail_ratio` | 物体跳变步数比例上限 |
| `projection_iou_warn/fail` | 网格投影框与分割框 IoU 下限 |
| `contact_distance_m` | 手腕到物体 OBB 的接触代理距离 |
| `contact_frame_pass/fail_ratio` | 接触代理帧率分层阈值 |

修改阈值时应先在已人工标注的好/坏 clip 小集合上校准，不要只为提高通过率而放宽。

## 4. 续跑

| 参数 | 中文含义 |
| --- | --- |
| `resume.skip_existing` | 已有有效结果时跳过 |
| `resume.force_hand_preprocess` | 强制重跑手部估计与清洗 |
| `resume.force_locomotion` | 强制重跑 Locomotion |
| `resume.force_smoothing` | 强制重跑身体平滑 |
| `resume.force_phc` | 强制重跑 PHC |
| `resume.force_gmr` | 强制重跑 GMR 和最终渲染 |

## 5. 人体、手和物理阶段

| 参数 | 中文含义 |
| --- | --- |
| `human.backend` | 人手后端：`hand4wholepp`、`hamer`、`wilor` |
| `human.constraint_profile` | 手部清洗约束组合，生产默认 `conservative` |
| `human.gvhmr_batch_size` | GVHMR 推理 batch |
| `human.hand_batch_size` | 手部模型 batch；显存/内存不足时设 `1` |
| `human.vitpose_image_scale` | ViTPose 输入缩放 |
| `human.low_memory` | 低内存模式 |
| `human.isolate_hand_process` | 手部阶段是否独立进程，便于释放内存 |
| `human.filters.wrist` | 腕部候选滤波；当前默认关闭 |
| `human.filters.temporal` | 手部时序异常修复 |
| `human.filters.fingers` | 手指异常和平滑修复 |
| `human.diagnostics` | 是否生成大体积手部诊断视频 |
| `locomotion.enabled` | 是否运行身体高度/接触优化 |
| `locomotion.smooth_window` | 身体平滑窗口 |
| `phc.enabled` | 是否运行 PHC 修复与 Isaac Gym 渲染 |

## 6. GMR 与 2×2

| 参数 | 中文含义 |
| --- | --- |
| `gmr.enabled` | 是否执行机器人重定向 |
| `gmr.hand_model` | `sharpa`、`g1`、`brainco` |
| `gmr.source` | `converted`、`smoothed`、`phc_smoothed`、`auto` |
| `gmr.target_fps` | 机器人动作目标 FPS |
| `gmr.human_yaw_offset_deg` | 人体世界到机器人 z-up 世界的 yaw；物体共享此值 |
| `gmr.camera_source` | 机器人渲染相机来源 |
| `gmr.render` | 是否渲染机器人视频 |
| `gmr.composite_2x2` | 是否生成 2×2 |
| `gmr.mujoco_gl` | 无头渲染后端，服务器常用 `osmesa` |
| `gmr.render_width/height` | GMR 单面板尺寸 |

## 7. 物体总开关

| 参数 | 中文含义 |
| --- | --- |
| `object.enabled` | 是否启用物体 |
| `object.mode` | `monocular` 或 `foundationpose` |
| `object.only_configured_clips` | 只处理 `object.clips` 中列出的 clip |
| `object.rerender_gmr_after_import` | 物体导入后重渲染 GMR 和 2×2 |
| `object.collision.method` | `auto`、`coacd` 或 `convex-hull` |
| `object.collision.density_kg_m3` | 默认密度 |
| `object.collision.mass_kg` | 已知质量；非空时反算密度 |
| `object.overlay.enabled/alpha` | GVHMR 物体覆盖层开关/透明度 |
| `object.quality.min_valid_ratio` | 6D 轨迹最小有效帧比例 |
| `object.quality.min_observed_ratio` | 非插值真实观测最小比例 |

## 8. 单目物体

| 参数 | 中文含义 |
| --- | --- |
| `monocular.run_reconstruction` | 是否运行分割/网格/尺度/跟踪 |
| `monocular.work_root` | 大体积物体临时目录 |
| `monocular.segmentation_prompt_mode` | `auto` 无交互；`click` 人工点选 |
| `monocular.segmentation_prompt` | 文本提示；通常由 clip 的物体名覆盖 |
| `monocular.samhq_checkpoint` | 自定义 SAM-HQ 权重路径；空值自动解析 |
| `monocular.cutie_checkpoint` | 自定义 Cutie 权重路径 |
| `monocular.require_segmentation_checkpoint` | 缺权重时是否直接失败 |
| `person_roi.enabled` | 是否启用近人体硬约束 |
| `person_roi.bbox_scale` | 人体框扩大倍数 |
| `person_roi.min_mask_inside_ratio` | mask 位于人体 ROI 内的最小比例 |
| `auto_detector.model` | YOLO 权重路径 |
| `auto_detector.confidence` | 候选最低置信度 |
| `auto_detector.min_box_inside_person_roi` | 候选框位于人体 ROI 的最小比例 |
| `auto_detector.box_padding_ratio` | 送入 SAM-HQ 前扩大 box 的比例 |
| `monocular.hunyuan_face_count` | Hunyuan 目标面数 |
| `monocular.hunyuan_timeout` | 云任务超时秒数 |
| `monocular.hunyuan_region` | 腾讯云区域 |
| `monocular.moge_resolution_level` | MoGe-2 分辨率/显存档位 |
| `megapose.pose_hypotheses` | 6D 初始候选数量 |
| `megapose.coarse_grid_stride` | 粗搜索步长 |
| `megapose.refiner_iterations` | 单帧精修次数 |
| `megapose.temporal_refiner_iterations` | 时序精修次数 |
| `megapose.reinit_interval` | 周期重初始化间隔 |
| `megapose.min_iou` | 轨迹保留最低 IoU |
| `megapose.reinit_iou` | 触发重初始化的 IoU |
| `megapose.max_interpolation_gap` | 允许插值的最大连续缺口 |

每条物体 clip：

| 参数 | 中文含义 |
| --- | --- |
| `object_name` | 目标类别；自动检测会做 chair/suitcase/yogaball 等别名归一化 |
| `detector_labels` | 可选 YOLO 类别列表，覆盖自动别名 |
| `reference_frame` | 网格生成、初始分割和尺度估计参考帧 |
| `expected_size_m` | 物体合理最小/最大边长范围（米） |
| `allow_no_object` | `false` 表示物体失败时整条失败 |
| `dynamic_release_frame` | 动态仿真释放帧 |
| `frame_offset` | 物体轨迹相对人体的帧偏移 |
| `robot_offset` | 物体在机器人世界的人工 xyz 偏移 |

## 9. 动态验证与运行环境

| 参数 | 中文含义 |
| --- | --- |
| `dynamic_validation.enabled` | 是否运行 MuJoCo 自由物体接触验证 |
| `dynamic_validation.required` | 动态验证缺失或失败时是否把 clip 判为关键失败 |
| `dynamic_validation.substeps` | 每帧仿真子步数 |
| `dynamic_validation.lift_threshold_m` | 判定抬起的高度阈值 |
| `dynamic_validation.object_floor_collision` | 物体是否与地面碰撞 |
| `dynamic_validation.render_video` | 是否输出动态验证视频 |
| `runtime.conda_base` | Conda 根目录 |
| `runtime.python` | 主启动器/检测器 Python |
| `runtime.cuda_visible_devices` | GPU 编号 |
| `runtime.segmentation_display` | `click` 模式的 DISPLAY；auto 模式不使用窗口 |
| `runtime.conda_envs.*` | SAM-HQ、MoGe-2、MegaPose 环境名 |
| `environment` | 映射到底层 Bash 的高级环境变量；不得存密钥 |

## 10. 常用 CLI

CLI 主要用于选择任务，不建议承载长期参数：

```text
--config-dir PATH
--stage human|object|quality|product|all
--clip-filter TEXT
--output-root PATH
--check
--dry-run
--print-config
--set dotted.path=value
```

长期参数应写进新的 pipeline YAML 并纳入版本管理。
