# 01. 输入规范化与全身准入

## 模块职责

本模块在任何 3D 推理前回答两个问题：视频能否稳定解码，以及它是否提供了单人、全身、双手可见的最低 2D 证据。它不估计 3D，不给姿态打“好坏分”，也不尝试把半身、多人或长期遮挡的视频补成可用全身数据。

实现入口为 scripts/run_pipeline_from_config.py 中的 run_human() 与 scripts/preflight_fullbody_gate.py；视频统一化由 GVHMR-hand/GVHMR-main/tools/pipeline/run_batch_dataset6.sh 完成。

## 输入契约

| 项目 | 要求 | 原因 |
|---|---|---|
| 文件格式 | .mp4、.avi、.mov、.mkv 或 .m4v | 配置入口会枚举这些扩展名 |
| 时长 | 默认至少 8 秒 | 短片段无法提供稳定的运动/时序证据 |
| 主体数 | 一个主要人物 | 当前 GVHMR batch 路径按单人 person_idx=0 导出 |
| 可见性 | 在抽样帧内尽量出现头、左右手腕、左右脚踝 | 后续同时需要全身高度、身体根轨迹与双手 |
| 画面 | 人物完整、尺度充足、少遮挡，避免镜前路人 | 多人/裁切会破坏单人跟踪与手部 crop |
| 编码 | FFmpeg/OpenCV 可读 | 准入和后续推理都依赖稳定解码 |

不要以“有一个人框”替代全身要求。默认门禁会拒绝第二个尺度达到主人物框 15% 以上的人物；也会拒绝头、踝或双手腕在抽样帧中长期不可见的片段。

## 工作流

~~~text
原视频
  -> 按扩展名、clip_filter、最小时长筛选
  -> YOLO-Pose 均匀抽样（默认 32 帧）
  -> 单人连续性与全身关键点比例检查
  -> gate: 仅传入合格 clip；report: 只记录、不拦截
  -> FFmpeg 生成工作视频
  -> 将工作视频交给 GVHMR
~~~

工作视频使用 H.264、CRF 18、30 FPS，目标画幅上限是 1280×960。脚本使用保持长宽比的缩放，因此最终尺寸可能是小于等于该上限且至少一边贴边的偶数尺寸，而不是无条件拉伸为 1280×960。它会校验缓存视频的帧数、时长、FPS 与几何信息；缓存损坏、旧尺寸或源视频更新时会重新转码。

## 启动方法

推荐从总入口运行；门禁在 --stage all 的开头自动执行。

~~~bash
PY_LOCO=/opt/conda/envs/locomotion/bin/python

# 预检查：验证配置、输入目录、准入模型路径；不做视频推理
$PY_LOCO scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --check

# 只选择名称含 chairwood 的输入
$PY_LOCO scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter chairwood
~~~

若要收集不合格视频的统计而仍继续推理，可临时使用：

~~~bash
--set input.fullbody_preflight.mode=report
~~~

这只适合排查数据集；它会把不合格 clip 放进后续链路，不应作为交付生产参数。

## 输出与交接

| 输出 | 位置 | 用途 |
|---|---|---|
| 准入 JSONL | <out>/fullbody_preflight.jsonl | 每个源文件一条完整机器可读记录 |
| 准入 CSV | <out>/fullbody_preflight.csv | 人工筛选、表格审计 |
| 规范化视频 | input.work_video.directory/<clip>.mp4 | GVHMR、手部和相机估计的共同 RGB 来源 |
| 工作视频指纹 | <work-video>.config | 证明缓存对应源视频和转码参数 |
| 控制台记录 | <out>/<clip>.pipeline.log | 从转码到后续每阶段的可追溯日志 |

JSONL/CSV 记录至少包含源文件、可解码状态、是否 eligible、拒绝原因、抽样帧数，以及主人体、头部、左右腕、左右踝、全身和竞争人物比例。接收方需要的结论是 eligible=true，而不是仅看到报告文件存在。

## 默认门禁阈值

默认值在 configs/default.yaml 的 input.fullbody_preflight。最重要的阈值是：主人体检测比例 ≥0.80、人物高度比例 ≥0.40、可用尺度比例 ≥0.60、头部比例 ≥0.60、每只腕 ≥0.30、每只踝 ≥0.55、同时有头和双踝的全身比例 ≥0.35、竞争人物比例为 0。

这些值是准入下限而不是质量承诺。通过门禁也可能在严重运动模糊、手很小或自遮挡时得到低可信度手部；这会在第 03、06 模块暴露。

### 参数速查

| 参数 | 默认值 | 作用与交接规则 |
|---|---:|---|
| `input.extensions` | mp4/avi/mov/mkv/m4v | 允许枚举的源文件后缀；新增后缀前先验证 FFmpeg/OpenCV 可解码 |
| `input.clip_filter` | 空 | 文件名子串筛选；逗号可分隔多个子串 |
| `input.min_duration_seconds` | 8.0 | 转码前的最短时长；缩短会降低时序证据 |
| `input.max_videos` | null | 最多处理的已排序 clip 数；用于小规模试跑 |
| `fullbody_preflight.enabled/mode` | true/gate | gate 拒绝不合格输入；report 只用于数据盘点 |
| `device/image_size/samples` | auto/640/32 | YOLO 设备、推理尺度与时间抽样数；samples 增大更稳但更慢 |
| `person_confidence/keypoint_confidence` | 0.25/0.35 | 检测与关键点最低置信度；不要为了让坏视频通过而降低 |
| `min_*_ratio` | 见上节 | 主人、尺度、头、双腕、双踝、全身的时间比例门限 |
| `max_competing_person_ratio` | 0.0 | 默认任何达到竞争尺度的第二人都会拒绝 |
| `min_competing_person_scale_ratio` | 0.15 | 认定竞争人物相对主人框的最小尺度 |
| `work_video.enabled/directory` | true/`dataset_new6_work_1280` | 是否生成、存放规范化视频；目录必须可写且不与原视频目录混用 |
| `width/height/crf/fps` | 1280/960/18/30 | 最大画幅、H.264 质量和固定 FPS；任一改变会失效缓存并影响下游帧数 |
| `work_video.force` | false | true 时无条件重转码；仅源视频或转码参数已改变时使用 |

## 常见问题与处理边界

| 现象 | 首先检查 | 正确处理 |
|---|---|---|
| video_unreadable | 原视频能否被 ffprobe/OpenCV 打开 | 重新导出为标准 H.264 MP4；不要跳过门禁 |
| 长度不足 | CSV 中的时长或源文件元数据 | 合并为足够长的连续片段，或经负责人批准调整 min_duration_seconds |
| competing_person | 画面中路人、镜像、海报是否被检测为人 | 裁剪/重拍/拆分视频；不要直接关闭单人限制 |
| 脚踝或头部比例低 | 是否长期只拍上半身/脚被遮挡 | 更换全身视频；这不是手部模型可补偿的问题 |
| 工作视频重复使用旧内容 | <clip>.mp4.config 与源视频修改时间 | 保持默认缓存校验；必要时设置 input.work_video.force=true |

本模块合格后交给下一模块的是规范化 RGB 视频和一个确定的单人 clip 名，而不是 YOLO 关键点作为最终人体运动数据。
