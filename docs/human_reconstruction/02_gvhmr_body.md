# 02. GVHMR 身体重建

## 模块职责

GVHMR 从已准入的单人工作视频估计身体的世界轨迹、全局朝向、身体关节旋转、形状和相机相关数据。它是身体运动的唯一上游估计器；Hand4Whole++ 只补充 MANO 手部，不替换身体、肩肘或根轨迹。

在当前工程中，配置入口选择 Hand4Whole++ 后端包装器：

~~~text
scripts/run_pipeline_from_config.py
  -> tools/pipeline/run_batch_dataset6_hand4wholepp.sh
       -> tools/pipeline/run_batch_dataset6.sh
            -> tools/pipeline/run_pipeline.sh
                 -> GVHMR/HMR2/ViTPose 推理
~~~

外部对接只能调用顶层 Python 入口。后三个 Bash 文件用于冻结内部环境变量、缓存指纹、模型路径和重跑规则，不能视为稳定公共 API。

## 输入

| 输入 | 来源 | 说明 |
|---|---|---|
| 工作视频 | 模块 01 | 30 FPS、等比例缩放后的单人 RGB 视频 |
| GVHMR/HMR2 checkpoint | GVHMR-hand/GVHMR-main/inputs/checkpoints/ | 不随 Git 提交，运行时挂载 |
| ViTPose whole-body checkpoint | 同上 | 提供人体/手部 2D 证据 |
| SMPL/SMPL-X/MANO 模型 | 模型挂载目录 | 用于人体参数化和后续手部一致性 |
| 配置参数 | YAML + CLI | 后端、batch size、低显存、缓存/重跑策略 |

human.gvhmr_batch_size 控制 GVHMR 主体 batch；human.low_memory 和 human.isolate_hand_process 用于避免手部前处理与 HMR2 长时间共占显存。对接方先使用默认值，不要为了吞吐把多个视频同时起多个进程。

## 处理流程

~~~text
工作视频
  -> 人体检测与跟踪 / ViTPose
  -> GVHMR 时序人体恢复
  -> HMR4D 世界空间 SMPL 参数
  -> GVHMR 视频渲染和中间缓存
  -> hmr4d_results.pt
~~~

脚本会为输入、手部代码和过滤配置写缓存指纹。resume.skip_existing: true 时，仅在视频、模型配置和结果一致时复用已有内容；改变手部过滤配置会使相关转换/渲染重算。常规重跑不需要先删除输出目录。

## 启动和诊断

常规运行由总入口触发：

~~~bash
/opt/conda/envs/locomotion/bin/python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all \
  --dataset-dir /data/raw_videos \
  --output-root /data/human_output \
  --set input.work_video.directory=/data/human_work \
  --clip-filter <clip>
~~~

仅展开将要调用的命令、不实际推理：

~~~bash
/opt/conda/envs/locomotion/bin/python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --dry-run --clip-filter <clip>
~~~

排障时可以强制重做人体/手部前处理，而不删除其他 clip：

~~~bash
... --set resume.force_hand_preprocess=true --clip-filter <clip>
~~~

run_pipeline.sh <video> <output-dir> 是单条内部脚本，只有在已由配置入口导出完整环境变量后才可用于开发调试。它不适合交接方直接运行，尤其不能假定直接调用时会得到正确的模型、Conda、PHC 或 GMR 开关。

## 输出

每条 clip 的主要输出为：

~~~text
<out>/<clip>/
  gvhmr_out/<clip>/
    hmr4d_results.pt
    preprocess/...
    1_incam.mp4 或同类 incam 预览
    2_global.mp4（版本/设置允许时）
  .gvhmr_input_cache_v1
  <clip>.pipeline.log
~~~

hmr4d_results.pt 是转换模块的权威输入。它内部携带 smpl_params_global（优先）或兼容的世界/相机空间 SMPL 参数与相机数据；不要把该 PyTorch pickle 作为对外数据格式，也不要让接收方直接依赖内部 key。对外接口由模块 04 的 NPZ 定义。

1_incam.mp4 是人体贴回输入相机的审核证据。没有该视频时，先检查是否显式设置了 GVHMR_SKIP_RENDER=1，再检查日志中的渲染错误；视频缺失不等于 hmr4d_results.pt 一定失效，但交付前应补齐人工审核材料。

## 成功判定

最低技术成功条件：

1. hmr4d_results.pt 存在且能由当前 locomotion 环境读取。
2. 模块 04 能从其中导出 001_converted.npz，且帧数与工作视频匹配。
3. 根轨迹与身体旋转没有 NaN/Inf，预览中人物没有整体跳帧、身份切换或全程倒置。

GVHMR 的世界坐标重力约定在后续交付中标为 gvhmr_world_gravity_negative_y。下游若使用 Z-up、左手系或机器人坐标，必须在自己的适配层显式转换；不能靠猜测交换平移分量。

## 失败定位

| 症状 | 可能边界 | 优先动作 |
|---|---|---|
| 初始化时找不到 checkpoint | 模型挂载/路径 | 执行 Docker check，核对模型目录而非下载到仓库 |
| CUDA OOM | 主体/手部进程显存冲突 | 保持 hand_batch_size: 1、low_memory: true、isolate_hand_process: true；先单条运行 |
| 人物跟错或身份跳变 | 输入不满足单人假设 | 回到模块 01 检查多人、反射和遮挡，修输入而非平滑输出 |
| 预览正常但后续转换失败 | 结果 schema/文件损坏 | 保留 hmr4d_results.pt 和 pipeline log，使用模块 04 的手动转换命令定位 |
| 轨迹尺度/地面异常 | 单目世界尺度的不确定性 | 交给模块 05 的高度优化；不要在此处手改 trans |

本模块结束时，身体仍是原始 GVHMR 估计，尚未与最终 MANO 手部封装成对外交付文件。
