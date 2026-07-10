# Loco-manipulation

人体动作到人形机器人全身重定向管线（Loco-Manipulation Pipeline）。

## 功能

从 RGB 视频出发，端到端生成物理可行的机器人全身动作数据：

```text
输入视频 -> GVHMR 人体姿态 -> Hand4Whole++ 手部 -> Locomotion 高度优化
         -> 时序平滑 -> PHC 物理修复 -> GMR 机器人重定向 -> Sharpa 手部 IK
         -> 2×2 对比渲染 -> 无 GT 质量分层 -> 安全 NPZ 资产导出
```

## 目录结构

```
├── PIPELINE_README.md              # 执行命令与数据流程
├── PIPELINE_ENGINEERING.md         # 模块边界、NPZ 协议、坐标链
├── configs/
│   ├── default.yaml                # 基础默认参数
│   ├── README.md                   # YAML 参数中文说明
│   └── pipelines/
│       ├── human_sharpa.yaml       # 人体-only 保留工作区
│       └── human_sharpa_batch.yaml # 人体-only 批处理(自动清理)
├── scripts/
│   ├── run_pipeline_from_config.py # 工程化主启动器
│   ├── evaluate_clip_quality.py    # pass/warn/fail 自动质量评价
│   └── export_dataset_product.py   # 最终资产导出
├── GVHMR-hand/                     # GVHMR + 手部估计
├── GVHMR-main/                     # 平滑/PHC 辅助
├── locomotion_pipeline-main/       # 高度/接触优化
├── phc-dev-felix-pipeline/         # 物理修复 (PHC)
├── GMR-master/                     # 机器人重定向与渲染
└── tests/                          # 单元测试
```

## 环境要求

- Conda 环境:
  - `locomotion`: GVHMR 推理 + Locomotion + 平滑
  - `phc`: Isaac Gym PHC 修复
  - `gmr`: GMR 机器人重定向 + MuJoCo 渲染
- SMPL-H / MANO 模型文件 (需单独下载)
- Unitree H1 机器人 MuJoCo XML 资产

## 快速开始

```bash
# 1. 检查配置
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --check --print-config

# 2. 运行人体-only 全流程
PYTHONUNBUFFERED=1 python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --clip-filter chairwood

# 3. 只导出已有结果
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage product --clip-filter chairwood
```

## 最终资产

```
assets/<dataset>/<clip>/
  human_motion.npz          # SMPL-H 身体 + MANO 双手
  human_phc_motion.npz      # PHC 物理修复轨迹
  robot_motion.npz          # H1 机器人关节角
  robot_hand_motion.npz     # Sharpa 22-DoF 手部
  camera.npz                # 相机参数
  quality_report.json       # pass/warn/fail 与指标明细
  preview_2x2.mp4           # 四画面对比
  manifest.json             # 资产清单
  checksums.sha256          # SHA-256 校验
```

## 执行顺序

| 阶段 | 输入 | 输出 |
|------|------|------|
| GVHMR | 工作视频 | hmr4d_results.pt |
| 手部 | 人物框 + ViTPose | MANO 参数 |
| 转换 | GVHMR + MANO | 001_converted.npz |
| Locomotion | converted NPZ | 高度/接触优化 |
| 平滑 | Locomotion 结果 | 001_smoothed.npz |
| PHC | smoothed NPZ | PHC 修复轨迹 |
| GMR | smoothed NPZ | robot_motion.pkl |
| Sharpa | 手部 sidecar | 001_sharpa_chain_hands.npz |
| 2×2 | 四路视频 | composite_2x2.mp4 |
| 质量 | 完整工作输出 | 分层报告与批量 CSV/JSONL |
| 导出 | 工作输出 | 安全 NPZ 资产包 |

## 依赖模型

运行前需下载以下模型到对应目录：

- SMPL-H: `GVHMR-hand/GVHMR-main/deps/smplh/`
- MANO: `GVHMR-hand/GVHMR-main/deps/mano/`
- ViTPose: `GVHMR-hand/GVHMR-main/deps/vitpose/`
- Hand4Whole++: `GVHMR-hand/GVHMR-main/deps/hand4wholepp/`
- GVHMR checkpoint: `GVHMR-hand/GVHMR-main/deps/gvhmr/`
- PHC checkpoint: `phc-dev-felix-pipeline/output/HumanoidIm/`

详见 `PIPELINE_README.md` 和各子模块的 `CLAUDE.md`。

## License

各子模块许可证独立。`product.rights` 默认全部为 `false`，商业使用前须逐项确认。
