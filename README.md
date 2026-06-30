# Locomanipulation

从单目 RGB 视频恢复人体、手部和交互物体，并将动作重定向到带灵巧手的机器人。

```text
RGB video
  -> GVHMR body + MANO hands
  -> locomotion / PHC
  -> GMR + Sharpa / G1 / BrainCo
  -> RGB-only or RGB-D object reconstruction
  -> MuJoCo collision and grasp validation
  -> 2x2 visualization
```

## Documentation

- [PIPELINE_ENGINEERING.md](PIPELINE_ENGINEERING.md)：工程架构、配置优先级、数据协议、
  `.pt/.npz/.pkl` 字段和物体重建流程。
- [PIPELINE_README.md](PIPELINE_README.md)：现有流水线的逐阶段命令和运行参数。
- [THIRD_PARTY.md](THIRD_PARTY.md)：外部仓库、固定版本、许可证和不能上传的资产。

## Quick start

```bash
git clone https://github.com/wentian810/locomanipulation.git
cd locomanipulation

# 恢复固定版本的第三方源码并应用本仓库 patch。
bash scripts/bootstrap_external_repos.sh core

# 检查源码布局；模型和人体资产仍需按文档下载。
python scripts/audit_public_repository.py

# 查看最终配置，不执行推理。
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/rgb_monocular_sharpa.yaml \
  --stage human \
  --check
```

配置优先级严格为：

```text
CLI > environment variables > pipeline YAML > configs/default.yaml
```

## Repository boundary

本仓库只发布源码、配置、许可证允许随源码分发的机器人描述和安装脚本，不包含：

- 数据集、输入视频、运行日志和渲染输出；
- `.ckpt/.pth/.pt` 模型权重；
- MANO、SMPL、SMPL-H、SMPL-X 模型文件；
- 腾讯云或其他服务的凭证；
- NVIDIA Isaac Gym SDK 二进制包；
- 由 bootstrap 脚本恢复的外部 Git 仓库。

这些文件不能通过 GitHub 仓库补发。请按
[PIPELINE_ENGINEERING.md](PIPELINE_ENGINEERING.md) 的安装清单从官方来源获取。

## License

该仓库组合了具有不同许可证的上游项目，不提供覆盖全部目录的统一许可证。
使用或再分发某个组件前，请阅读其目录内许可证和
[THIRD_PARTY.md](THIRD_PARTY.md)。特别注意 GVHMR 仅允许教育、研究和非营利用途。
