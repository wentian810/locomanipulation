# 模型权重与依赖下载指南

本文档列出运行人体-only 全流程所需的所有外部模型权重和 checkpoints。
这些文件不上传到代码仓库，需按以下说明自行下载。

## 一、SMPL 人体模型

### SMPL-H (必需)

- **用途**: GVHMR 人体姿态估计输出格式
- **下载**: https://mano.is.tue.mpg.de/ (需注册 SMPL+H 许可)
- **放置路径**: `GVHMR-hand/GVHMR-main/deps/smplh/`

### MANO 手部模型 (必需)

- **用途**: Hand4Whole++ 手部姿态估计
- **下载**: https://mano.is.tue.mpg.de/ (需注册 MANO 许可)
- **放置路径**: `GVHMR-hand/GVHMR-main/deps/mano/`

### SMPL-X (GMR 需要)

- **用途**: GMR IK 求解器参考人体
- **下载**: https://smpl-x.is.tue.mpg.de/
- **放置路径**: `GMR-master/assets/body_models/smplx/`

## 二、GVHMR 姿态估计

| 模型 | 下载来源 | 放置路径 |
|------|----------|----------|
| GVHMR checkpoint | [GVHMR](https://github.com/zju3dv/GVHMR) | `GVHMR-hand/GVHMR-main/deps/gvhmr/` |
| ViTPose-H | [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | `GVHMR-hand/GVHMR-main/deps/vitpose/` |
| Hand4Whole++ | 参考项目文档 | `GVHMR-hand/GVHMR-main/deps/hand4wholepp/` |

## 三、PHC 物理修复

| 组件 | 下载来源 | 说明 |
|------|----------|------|
| Isaac Gym | [NVIDIA](https://developer.nvidia.com/isaac-gym) | 需开发者账号，安装到 `phc` conda 环境 |
| PHC primitive | 训练输出 | `phc-dev-felix-pipeline/output/HumanoidIm/phc_3/Humanoid.pth` |
| PHC composer | 训练输出 | `phc-dev-felix-pipeline/output/HumanoidIm/phc_comp_3/Humanoid.pth` |

## 四、GMR 机器人重定向

- **SMPL-X body model**: `GMR-master/assets/body_models/smplx/`
- **Unitree H1 MuJoCo XML**: 已在 `GMR-master/assets/unitree_h1/` 中（无需额外下载）

## 五、Conda 环境创建

```bash
# GVHMR + Locomotion + 平滑
conda create -n locomotion python=3.10 -y
conda activate locomotion
pip install torch numpy scipy opencv-python smplx ...
# 按 GVHMR-hand 子项目文档补充完整依赖

# PHC (Isaac Gym 物理仿真)
conda create -n phc python=3.8 -y
conda activate phc
# 安装 Isaac Gym Preview 4
# 按 phc-dev-felix-pipeline 文档补充完整依赖

# GMR (机器人重定向 + MuJoCo 渲染)
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e GMR-master/
conda install -c conda-forge libstdcxx-ng -y
```

## 六、Locomotion Pipeline 资源文件

以下资源文件需要从上游 pipeline 仓库同步，不上传到代码仓库。

### 主配置文件

- **源路径**: `pipeline/locomotion_pipeline-main/code/configs/config.yaml`
- **目标路径**: `locomotion_pipeline-main/code/configs/config.yaml`
- **用途**: `optimizer_v2.py` 的默认配置（重力、穿透检测、速度检测参数）

### SMPL-H 分段定义

| 文件 | 目标路径 | 用途 |
|------|----------|------|
| `arm_leg_seg.json` | `locomotion_pipeline-main/assets/smplh-seg/` | 手臂/腿部顶点分段 |
| `arm_leg_seg_faces.json` | `locomotion_pipeline-main/assets/smplh-seg/` | 手臂/腿部分段面索引 |
| `coarse_seg.json` | `locomotion_pipeline-main/assets/smplh-seg/` | 粗粒度顶点分段 |
| `coarse_seg_faces.json` | `locomotion_pipeline-main/assets/smplh-seg/` | 粗粒度分段面索引 |

引用这些文件的代码：
- `code/optim/smplh_provider.py:51,63` — 加载 arm_leg_seg
- `code/utils/geometry_utils.py:133,142` — 几何工具
- `code/core/penetration.py:273` — 穿透检测

### SMPL-H 身体模型

| 文件 | 目标路径 | 说明 |
|------|----------|------|
| `SMPLH_MALE.pkl` / `.npz` | `locomotion_pipeline-main/assets/smplh/` | 男性模板 |
| `SMPLH_FEMALE.pkl` / `.npz` | `locomotion_pipeline-main/assets/smplh/` | 女性模板 |
| `SMPLH_NEUTRAL.pkl` / `.npz` | `locomotion_pipeline-main/assets/smplh/` | 中性模板 |
| `kid_template.npy` | `locomotion_pipeline-main/assets/smplh/` | 儿童模板 |

> **注意**: 这些与 GVHMR 的 SMPL-H 模型是不同副本，需分别放置。

### ACCAD 动作捕捉数据（252 个 .npz 文件）

- **目标路径**: `locomotion_pipeline-main/assets/ACCAD/`
- **用途**: 高度优化参考动作库
- **来源**: `pipeline/locomotion_pipeline-main/assets/ACCAD/`（完整 rsync）

### 同步命令

```bash
SRC=/home/zun/桌面/datapipeline/pipeline/locomotion_pipeline-main
DST=/home/zun/桌面/datapipeline/Loco-manipulation-human-only-release/locomotion_pipeline-main

# 配置文件
cp "$SRC/code/configs/config.yaml" "$DST/code/configs/config.yaml"

# SMPL-H 分段
rsync -av "$SRC/assets/smplh-seg/" "$DST/assets/smplh-seg/"

# ACCAD 动作数据
rsync -av "$SRC/assets/ACCAD/" "$DST/assets/ACCAD/"
```

## 七、验证安装

```bash
python scripts/run_pipeline_from_config.py \
  --config_dir configs/pipelines/human_sharpa.yaml \
  --stage all --check --print-config
```

`--check` 会检查依赖和配置完整性，不运行推理。

## 八、PHC 物理修复组件

PHC (`phc-dev-felix-pipeline/`) 需要从上游 pipeline 完整同步以下目录，不上传到代码仓库。

### 核心 Python 包

| 目录 | 目标路径 | 说明 |
|------|----------|------|
| `phc/` | `phc-dev-felix-pipeline/phc/` | PHC 主包（env, learning, utils, assets） |
| `poselib/` | `phc-dev-felix-pipeline/poselib/` | 姿态工具库（骨骼、旋转、可视化） |

### 数据处理脚本

目标路径: `phc-dev-felix-pipeline/scripts/data_process/`

| 文件 | 用途 |
|------|------|
| `convert_zitai_to_phc.py` | 将姿态数据转为 PHC 输入格式 |
| `batch_repair_zitai.py` | 批量物理修复主入口 |
| `export_repaired_smpl_npz.py` | 导出修复后的 SMPL 数据 |
| `fit_smpl_motion.py` / `fit_smpl_shape.py` | SMPL 运动/形状拟合 |
| `hmr2_video_to_smpl.py` | HMR2 视频转 SMPL |
| `repair_from_fail_filter.py` | 失败过滤后重修复 |
| 其余脚本 | 数据转换、关键点提取等 |

### 配置与资产

目标路径: `phc-dev-felix-pipeline/phc/data/`

| 路径 | 说明 |
|------|------|
| `cfg/config.yaml` | 主配置 |
| `cfg/env/` | 环境配置（im, h1_phc, g1_phc 等） |
| `cfg/robot/` | 机器人定义（H1, G1, SMPL humanoid） |
| `cfg/learning/` | 学习/训练配置 |
| `assets/mjcf/` | MuJoCo 人体模型 XML |
| `assets/robot/unitree_h1/` | H1 机器人 mesh + URDF |
| `assets/robot/unitree_g1/` | G1 机器人 mesh + XML |
| `assets/usd/smpl/` | SMPL USD 资产 |
| `assets/urdf/` | 通用碰撞体 URDF |
| `assets/mesh/smpl/` | SMPL 分段 STL mesh（24 个身体部位） |

### 采样数据

目标路径: `phc-dev-felix-pipeline/sample_data/`

| 文件 | 说明 |
|------|------|
| `accad_cleaned.pkl` | 清洗后的 ACCAD 动作数据索引 |
| `amass_isaac_*.pkl` | AMASS 数据索引（性别 beta、直立姿态） |
| `standing_test.pkl` | 站立测试索引 |
| `standing_test/standing_seq/` | 站立测试序列（~50 个 .npz） |

### 同步命令

```bash
SRC=/home/zun/桌面/datapipeline/pipeline/phc-dev-felix-pipeline
DST=/home/zun/桌面/datapipeline/Loco-manipulation-human-only-release/phc-dev-felix-pipeline

rsync -av --progress \
  --exclude='output/' \
  --exclude='runs/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='isaacgym/' \
  --exclude='.git/' \
  "$SRC/" "$DST/"
```

> **注意**: Isaac Gym 和 PHC 训练权重（`output/HumanoidIm/`）需单独配置，不在本次同步范围内。

## 九、GMR 机器人重定向组件

GMR (`GMR-master/`) 需要从上游 pipeline 完整同步以下目录，不上传到代码仓库。

### 机器人 Mesh 资产

| 路径 | 说明 | 文件数 |
|------|------|--------|
| `assets/unitree_h1/meshes/` | H1 机器人 STL+DAE mesh（~50 个 link） | ~100 |
| `assets/unitree_h1_2/` | H1 v2 变体（含 handless） | ~6 |
| `assets/kuavo_s45/` | Kuavo S45 机器人 | ~2 |
| `assets/booster_k1/` | Booster K1 | ~4 |
| `assets/booster_t1_29dof/` | Booster T1（含 meshes） | ~20 |
| `assets/berkeley_humanoid_lite/` | Berkeley Humanoid Lite | ~7 |
| `assets/stanford_toddy/` | Stanford Toddy | ~1 |
| `assets/openloong/` | 开龙门机器人 | ~8 |
| `assets/engineai_pm01/` | EngineAI PM01 | ~3 |
| `assets/galaxea_r1pro/` | Galaxea R1 Pro | ~2 |
| `assets/agibot_a2/` | 智元 A2（含 convex hulls） | ~50 |
| `assets/hard_motions/` | 困难动作参考 | ~2 |

### 人体模型

| 路径 | 说明 |
|------|------|
| `assets/body_models/smplh/` | → symlink to `../../../locomotion_pipeline-main/assets/smplh` |
| `assets/body_models/smplx/` | → symlink to `../../../GVHMR-main/inputs/checkpoints/body_models/smplx` |

> **注意**: 这两个是相对路径 symlink，指向项目内已有模型。如从上游 rsync 后需修复
> symlink 目标（旧 symlink 指向 pipeline 源路径）。命令：
> ```bash
> cd GMR-master/assets/body_models
> rm -f smplh smplx
> ln -s ../../../locomotion_pipeline-main/assets/smplh smplh
> ln -s ../../../GVHMR-main/inputs/checkpoints/body_models/smplx smplx
> ```

### 第三方库

| 路径 | 说明 |
|------|------|
| `third_party/poselib/` | 姿态工具库（GMR 依赖） |
| `third_party/robot_hands/brainco_description/` | BrainCo 灵巧手模型 |
| `third_party/robot_hands/unitree_xr_teleoperate/` | Unitree XR 手部模型 |

### 同步命令

```bash
SRC=/home/zun/桌面/datapipeline/pipeline/GMR-master
DST=/home/zun/桌面/datapipeline/Loco-manipulation-human-only-release/GMR-master

rsync -av --progress \
  --exclude='output/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  "$SRC/" "$DST/"
```
