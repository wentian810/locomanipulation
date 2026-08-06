# 08. human-only 质量分层与安全产品导出

## 模块职责

质量模块只读完整的 human-only 工作区，生成每条 clip 的 pass、warn 或 fail 结论与数据集汇总。产品模块只读取已可信的工作结果，将必要数值转换为压缩 NPZ、复制预览、写入清单和 SHA-256；它不重跑 GVHMR、PHC 或 GMR。

human_sharpa.yaml 的生产设置为：

~~~yaml
quality_evaluation:
  enabled: true
  run_after_stages: [human, object, all]
  require_for_product: true
product:
  enabled: true
  root: assets/dataset_new6_human_sharpa
  export_after_stages: [human, all]
  object_policy: exclude
~~~

上面的 `root` 是直接以 Python 入口开发调试时的 YAML 默认值。发布启动器会额外传入 `--set product.root=/data/output/product`，因此接收方实际拿到的资产在 `<output-root>/product/<clip>/`，不会留在容器临时文件系统。`product.root` 必须与 `output.root` 不同，但可以是 output root 的子目录。

严格保证 object.enabled=false，因此不会读取、导出或等待任何场景/物体资产。

## 参数速查

| 参数 | 默认/生产值 | 作用与交接规则 |
|---|---:|---|
| `quality_evaluation.enabled` | true | 生成质量报告；产品交付不可关闭 |
| `run_after_stages` | human/object/all | 哪些 stage 后自动评估；human-only 实际由 human/all 触发 |
| `require_for_product` | true | 没有当前质量报告时拒绝导出 |
| `pass_score` | 80 | 宏观评分分界；critical failure 仍可覆盖分数 |
| `projection_samples` | 40 | 投影/可视化抽样数；增大更慢但不产生 GT |
| `weights.human_only` | files 15, human 25, hands 35, gmr 15, visualization 10 | 各部分对总分的权重；改动代表新的质量版本 |
| `thresholds.*` | 见本模块指标表/default.yaml | 帧数、手部、速度、关节限位等 warn/fail 门限；任何变更需写入版本记录 |
| `product.enabled` | true | 是否产出对外资产包 |
| `product.root` | YAML 为 assets/...；release 为 `<out>/product` | 发布脚本强制宿主可持久化路径 |
| `minimum_quality_status` | warn | pass/warn 可导出，fail 永不导出 |
| `include_camera/include_phc_motion/include_preview` | true/true/true | 数据/预览包含策略；当前综合 motion.npz 始终含可用 camera/PHC provenance |
| `require_preview` | true（human_sharpa） | 缺 2×2 审核视频则拒绝产品导出 |
| `object_policy` | exclude | human-only 必须为 exclude |
| `retention.prune_*` | false | 调试配置不删除工作区；清理只允许批处理已核验输出 |

## 质量工作流

~~~text
可信工作区
  -> 必要文件、NPZ 安全读取和帧数/FPS 一致性
  -> 人体根速度与时序检查
  -> MANO valid、重投影代理、spike、修复率
  -> GMR 四元数、关节限位和源轨迹溯源
  -> Sharpa 帧数与手链数据
  -> 2×2 视频可读性
  -> quality_report.json + 全集 CSV/JSONL
~~~

对于 human-only，必要工作文件包含 gmr.source 选定身体 NPZ、001_smplx_hands.npz、robot_motion.pkl、gvhmr_camera.npz、PHC 启用时的 001_phc_smoothed.npz、Sharpa 手模型时的 001_sharpa_chain_hands.npz，以及启用 composite 时的 composite_2x2.mp4。

质量分数只用于排序，不能覆盖关键失败。files、human、hands 或 gmr 阶段为 fail 时会进入 critical_fail_reasons，整条输出 fail。PHC 回退到 smoothed 不是静默成功：当 GMR source=final，评估器会读取 final_motion_selection.json 并以 warn 记录 final_motion_phc_fallback 原因。

## 启动方法

全流程结束后，质量会自动运行。对已有工作区单独重算：

~~~bash
PY_LOCO=/opt/conda/envs/locomotion/bin/python

$PY_LOCO scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage quality \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter <clip>
~~~

输出位置：

~~~text
<out>/<clip>/quality_report.json
<out>/quality_overview.csv
~~~

质量报告必须属于本次 clip，并使用当前 schema 版本；不要将不同输出根或不同配置生成的旧报告复制进来绕过产品门禁。

## 主要指标与默认阈值

| 类别 | 指标 | pass / warn / fail 的关键边界 |
|---|---|---|
| 帧数 | 所有模态最小帧数 / 最大帧数 | ≥0.995 / [0.98,0.995) / <0.98 |
| 手部可用性 | 左右 hand_valid ratio | ≥0.85 / [0.50,0.85) / <0.50 |
| 手部重投影代理 | 相对误差 P90 | ≤0.35 / (0.35,0.60] / >0.60 或缺失 |
| 手部突跳 | spike ratio | ≤0.03 / (0.03,0.10] / >0.10 |
| 手部修复 | repaired ratio | ≤0.35 / (0.35,0.60] / >0.60 |
| 身体速度 | root speed P95 | ≤3.0 m/s / (3.0,6.0] / >6.0 m/s |
| GMR | 四元数、关节限位、机器人帧数 | 有限、单位四元数、违反率处于 YAML 阈值内 |
| 可视化 | 2×2 可读性与长度 | 可解码并与人体帧数一致 |

手部重投影是适配器侧代理而非 GT 精度；source_reliable 只计直接源观测，不把时序补帧当视觉证据。质量报告给出 metric details、数据来源和解释，人工审核应结合 1_incam、PHC 与 GMR 视频。

## 产品导出流程

~~~text
quality_report 达到 minimum_quality_status
  -> 临时 stage 目录
  -> 将 human、robot、Sharpa、camera 合并为一个安全的 motion.npz（命名空间字段）
  -> 复制质量报告、最终选择记录和 2×2 预览
  -> 写脱敏 pipeline_config、rights、manifest
  -> 写 SHA-256
  -> allow_pickle=False 复核全部 NPZ
  -> 原子替换 <asset>/<clip>
  -> 重建 <asset>/catalog.jsonl
~~~

产品导出器从可信工作区读取 robot_motion.pkl，但生成的资产包不包含 pickle 或 PT。所有最终 NPZ 使用 np.savez_compressed，禁止 object dtype，必须以 allow_pickle=False 再次打开，并拒绝 NaN/Inf。

单独导出已有结果：

~~~bash
$PY_LOCO scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage product \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter <clip>
~~~

产品门禁要求 quality_report.json 存在、所属 clip 正确、schema 不低于 minimum_quality_schema_version，并且状态不低于 product.minimum_quality_status。若失败，工作区会被保留；不要手工创建空质量报告。

## human-only 资产包内容

~~~text
<asset>/<clip>/
  motion.npz                    # 必需；human__/robot__/sharpa__/camera__ 字段
  quality_report.json
  final_motion_selection.json
  preview_2x2.mp4
  pipeline_config.yaml
  manifest.json
  checksums.sha256
~~~

`motion.npz` 不是不透明压缩包：`manifest.json.data_contract.motion.components` 必须列出 human、robot、sharpa、camera，所有字段以双下划线命名空间隔离，例如 `human__translation`、`robot__dof_position`、`sharpa__left_qpos`、`camera__K_fullimg`。其精确字段与 shape 由模块 09 约束。

不会交付 hmr4d_results.pt、MANO PT、ViTPose feature、robot_motion.pkl、PHC pickle、checkpoint、SMPL/MANO 模型、密钥、原视频、分割 mask 或任何物体/场景目录。当前导出器也不会凭空生成 rights.json；输入/人物/模型许可须由发布方在交付系统或项目工单中单独记录。

## 资产包验证与可选清理

接收方可执行：

~~~bash
cd <asset>/<clip>
sha256sum -c checksums.sha256
~~~

并以 allow_pickle=False 打开每个 NPZ。manifest.json 的 validation.status 必须为 passed，所有路径必须相对资产目录，不应泄露生产机绝对路径。

批处理配置 human_sharpa_batch.yaml 可在资产包及 SHA-256 复核成功后清理对应 scratch 工作区。调试配置 human_sharpa.yaml 默认不清理。不要手工删除 output.root 或使用通配符清空工作目录；清理只允许针对已验证的 clip 直接子目录。

## 交付许可边界

当前 `motion.npz` 产品包不包含 `rights.json`，也不会把人物授权、输入视频或模型许可证伪装成技术字段。发布方必须在交付工单、数据治理系统或同级发布清单中记录输入视频、人物授权、GVHMR/Hand4Whole++/SMPL/Unitree 资产的许可状态。它们是交付门禁信息，不是法律意见；即使技术质量为 pass/warn，只要权利状态不满足目标用途，也不能对外分发或商业使用。
