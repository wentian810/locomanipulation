# 03. Hand4Whole++ 与 MANO 手部重建

## 模块职责

本模块在 GVHMR 身体结果的基础上，恢复左右手的 MANO 关节旋转、手腕全局朝向、21 个三维关节、可用性与质量信息。它的设计目标是让最终可视化、手部 sidecar 和后续消费者使用同一条经过时序处理的手部轨迹。

当前生产后端是 Hand4Whole++，不是 HaMeR，也不是 WiLoR。配置必须写为：

~~~yaml
human:
  backend: hand4wholepp
~~~

Hand4Whole++ 通过单次全身估计得到手部候选。工程再利用 GVHMR 的人体跟踪与 ViTPose whole-body 证据进行时序判别，因此“手部质量”不是单模型的一次性置信度。

## 输入

| 输入 | 提供者 | 用途 |
|---|---|---|
| GVHMR 人体轨迹与人物跟踪 | 模块 02 | 锁定单人、提供与身体同帧的上下文 |
| ViTPose whole-body 关键点 | GVHMR 前处理 | 手部可见性、2D 误差和可见手细化证据 |
| Hand4Whole++ snapshot_6.pth | 第三方模型目录 | 手部/全身参数回归 |
| MANO/SMPL-X 资产 | 模型目录 | 将姿态转为统一的关节和 sidecar |
| 手部配置 | human 及 environment 字段 | batch、裁剪跟踪、过滤、直接 MANO 重算 |

主模型与 checkpoint 的默认位置如下，交付容器必须按 Docker 文档挂载：

~~~text
GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE/
  demo/snapshot_6.pth
GVHMR-hand/GVHMR-main/inputs/checkpoints/
  vitpose/...
  body_models/...
~~~

## 处理流程

~~~text
GVHMR 的人物轨迹和全身关键点
  -> Hand4Whole++ 单次手部估计
  -> 短缺口 crop 跟踪（可选，当前生产开启 flow_kalman）
  -> 时序异常、重投影误差和手框异常识别
  -> 手指局部过滤与旋转连续化
  -> 可见手 2D 细化（可选，当前生产开启非指尖拟合 + 指尖留出验证）
  -> 从最终 MANO pose 重新计算 21 个 joints
  -> 最终 MANO 参数与诊断材料
~~~

不能将这些步骤理解为“凭空修复看不见的手”。短缺口允许使用光流/Kalman 连接，长期遮挡仍会降低 source_reliable 或有效性；时序平滑保证连续性，不能证明掌心/手背语义正确。

## 当前 human-only 配置的关键开关

| 配置 | 当前值 | 作用 |
|---|---:|---|
| human.hand_batch_size | 1 | 优先避免显存峰值，适合交付运行 |
| human.hand_crop_tracking.mode | flow_kalman | 只桥接短检测缺口 |
| max_gap / max_prediction_gap | 8 / 2 帧 | 长遮挡不伪装成直接观测 |
| direct_observation_quality | 0.75 | 高质量直接观测优先于预测 |
| filters.temporal / fingers | true / true | 去除时序突跳并平滑指关节 |
| visible_hand_refine.enabled | true | 有充分 2D 证据才尝试细化 |
| fit_partition | non_tip | 拟合非指尖，指尖只作为 holdout 检验 |
| GVHMR_RECOMPUTE_DIRECT_MANO | 1 | 最终 pose、关节和 sidecar 同源 |

约束 profile 为 conservative。它是针对手指角度与形态的保护性约束，而不是“真实手势”保证。若进行 A/B 试验，必须保存 profile、模型 snapshot 和有效配置，且禁止覆盖交付目录。

### 可见手细化与过滤参数

| 参数组 | 当前生产值/默认值 | 作用与限制 |
|---|---|---|
| `filters.temporal/fingers` | true/true | 时序与手指局部平滑；`wrist_mode` 保持 smooth，不能用它推断掌心语义 |
| `filters.global_orient_fill_mode` | interpolate | 仅补连续性；preserve 用于 A/B，不可把空洞当观测 |
| `visible_hand_refine.device/batch_size` | auto/16 | 细化设备和 batch；仅影响可见手细化阶段 |
| `steps/lr/prior_weight` | 8/0.02/0.02 | 2D 细化的步数、学习率、先验；改动必须保留前后诊断 |
| `fit_confidence/min_keypoints` | 0.60/12 | 足够 2D 证据才拟合；低于门限时应跳过而非强行修手 |
| `fit_partition` | non_tip | 用非指尖拟合，指尖留作验证；all 仅兼容旧实验 |
| `holdout_min_keypoints/max_relative_regression_px` | 3/0.0 | 至少可验证点数和允许的留出指尖回退；生产不允许回退 |
| `max_delta_degrees` | 25 | 单次细化允许的最大角度变化，防止观测噪声拉飞手势 |
| `min_relative_improvement/min_relative_improvement_px` | 0.25/2.0 | 接受细化所需的最小改善 |
| `max_absolute_regression_px/max_anchor_error_*` | 2.0/20 px/0.25 bbox | 保护身体-手腕连接，超过即拒绝细化候选 |
| `evidence_*` | 见 default.yaml | 低置信度、过小手框、过短连续段均不应触发细化 |

## 启动方式

正常情况下不单独启动手部模块，它由人体总入口作为同一条视频的后续阶段运行：

~~~bash
PY_LOCO=/opt/conda/envs/locomotion/bin/python

$PY_LOCO scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter <clip>
~~~

若只想使已存在 clip 的手部前处理、过滤和转换重新执行：

~~~bash
$PY_LOCO scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --set resume.force_hand_preprocess=true \
  --clip-filter <clip>
~~~

这会触发依赖于手部过滤指纹的下游转换和人体平滑重建。不要手工删掉某个 mano_params 文件来“强制刷新”，因为缓存指纹、渲染和 sidecar 会失去一致性。

## 过程输出

典型输出位于 <out>/<clip>/ 下。不同版本或不同开关会少量增减中间文件，以下是常见命名：

~~~text
mano_params.pt
mano_params_temporal_fixed.pt
mano_params_finger_fixed.pt
mano_params_visible_refined.pt
mano_params_final_finger_smoothed.pt
mano_params_direct_mano_recomputed.pt
hamer_diagnostics/
  <clip>_hamer_diag_<track>.mp4
  <clip>_hamer_diag_<track>.json
~~~

最终被转换模块消费的是当前运行选出的 MANO 文件，而不是由对接方凭文件名任选一个。后续正式接口是 001_smplx_hands.npz；该文件包含最终手部 pose、有效性、质量与原始值的对照字段，见模块 04 与 06。

## 诊断与验收要点

手部诊断视频叠加了视频、关键点、有效性和误差信息；JSON 可用于保留人工判定理由。审核至少覆盖：

1. 左右手是否与身体左右侧一致，手腕是否突然翻面。
2. 拇指和四指是否出现一两帧的大幅折叠或张开。
3. 遮挡段是否只在允许的短缺口内插值，长期遮挡是否被明确标记为低证据。
4. 细化后是否保持了未参与拟合的指尖误差不回退；当前生产配置要求不允许回退。

重投影误差是与现有 2D 适配器的一致性代理，不是带标注手部真值误差。它应结合 source_reliable、valid、spike 和视频共同解释。

## 常见问题

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| 手部阶段 CUDA OOM | batch、模型加载与 GVHMR 同时占显存 | 保持 hand_batch_size=1、low_memory=true、isolate_hand_process=true；单 clip 排查 |
| 手框在短时间消失 | 快速移动、运动模糊或 2D 关键点漏检 | 查看诊断；flow_kalman 仅允许短缺口，不要扩大 max_gap 以掩盖输入问题 |
| 手腕或掌心偶发翻转 | 单目歧义、遮挡、低关键点质量 | 标记 review；检查输入/证据，不要仅靠增大平滑窗口 |
| 手势僵硬 | 过滤过强或原始图像证据不足 | 与原始 mano_params 对比后做隔离 A/B；不要改动已交付结果 |
| 左右手互换 | 人体跟踪/画面镜像/输入异常 | 回看模块 01 与 GVHMR 预览；属于上游身份/几何问题 |

此模块输出的是人体手部重建数据，不输出任何机器人手关节、Sharpa 链或 GMR 文件。
