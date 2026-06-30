# Third-party sources and restricted assets

Locomanipulation 是多项目集成仓库。各组件继续适用其原始许可证；本文件不替代许可证
正文。

## Included source snapshots

| 目录 | 上游/用途 | 许可证提示 |
| --- | --- | --- |
| `GVHMR-hand/GVHMR-main` | GVHMR + 手部集成 | 教育、研究、非营利；商业使用需联系作者 |
| `GMR-master` | 机器人动作重定向 | MIT；机器人资产可能有独立许可证 |
| `phc-dev-felix-pipeline` | PHC | BSD-3-Clause-Clear；资产许可证独立 |
| `locomotion_pipeline-main` | 身体高度/接触优化 | 以目录内声明为准 |
| `do-as-i-do-main` | 单目手物重建与 Sharpa 描述 | MIT；子模块许可证独立 |
| `foundationpose-plus-plus-main` | RGB-D/FoundationPose 适配 | FoundationPose、Cutie、SAM-HQ 分别适用目录内许可证 |

## Restored by bootstrap

`scripts/bootstrap_external_repos.sh` 将以下仓库固定到已验证版本。它们不会作为嵌套
Git 仓库上传。

| 目标目录 | 上游 | Commit |
| --- | --- | --- |
| `GVHMR-hand/GVHMR-main/third-party/DPVO` | `princeton-vl/DPVO` | `859bbbfdac6c6185f345003b3c473901fcd13ace` |
| `GVHMR-hand/GVHMR-main/third-party/hamer` | `geopavlakos/hamer` | `3a01849f4148352e9260b69bf28b65d1671a4905` |
| `GVHMR-hand/GVHMR-main/third-party/WiLoR` | `rolpotamias/WiLoR` | `fcb911312a38fa8badd30d9656a167485d61b8f9` |
| `GVHMR-hand/GVHMR-main/third-party/Hand4Whole-plus-plus_RELEASE` | `mks0601/Hand4Whole-plus-plus_RELEASE` | `f81d35ddd2b74206c40142243eb62b6d64ce0d65` |
| `phc-deps/SMPLSim` | `ZhengyiLuo/SMPLSim` | `b5c08720503ad5fff64050c4d289c42d947fcf8d` |
| `phc-deps/chumpy` | `mattloper/chumpy` | `580566eafc9ac68b2614b64d6f7aaa84eebb70da` |
| `phc-deps/smplx_fork` | `ZhengyiLuo/smplx` | `a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813` |

`do-as-i-do-main/reconstruction/setup/00_init_submodules.sh` 负责恢复该物体管线自己的
SAM3、SAM3D、Fast-SAM3D、HaWoR 和 TAPIR 固定版本。

## Local patches

以下 patch 会由 bootstrap 自动应用：

- `patches/third_party/wilor_gvhmr.patch`：降低 checkpoint 加载峰值内存，并暴露
  Hand4Whole++ 所需的 ViT feature。
- `patches/third_party/hand4wholepp_gvhmr.patch`：允许缺失可选 FLAME 索引，并导出
  手存在性和根姿态。

## Never redistribute through this repository

- MANO/SMPL/SMPL-H/SMPL-X 模型、blend shape 和人体模板；
- GVHMR、HaMeR、WiLoR、Hand4Whole++、PHC、SAM3/SAM3D 等 checkpoint；
- NVIDIA Isaac Gym SDK 和其闭源二进制；
- 腾讯云密钥、Hugging Face token 或其他凭证；
- 公司/受试者视频、导出的动作文件和仿真结果。

MANO/SMPL 系列必须由使用者接受官方许可证后自行下载。模型下载地址、目标路径与
检查命令见 `PIPELINE_ENGINEERING.md`。
