# ext-phc 当前使用说明

## 1. 这个仓库在 `smpl_work` 里的职责

`ext-phc` 是 `smplpipeline` 的 stage2 修复后端，不是单独的数据筛选仓库。

当前主链路是：

1. `smplpipeline` stage1 在某个 run 目录下产出 `filter/fail_filter/`
2. `smplpipeline` 的 `run_two_stage_repair.py` 负责把待修复片段同步出来
3. `ext-phc/scripts/data_process/repair_from_fail_filter.py` 读取这些失败片段并持续修复
4. PHC 通过 `phc/run_hydra.py` 跑实际修复
5. 修复后的 `.npz`、summary、failure yaml 都回写到该 run 目录

排查问题时，优先把 `ext-phc` 当成 stage2 运行时来理解。

## 2. 当前重要目录

- `phc/`
  - PHC 核心代码
  - `phc/run_hydra.py` 是底层执行入口
- `phc/data/cfg/`
  - 关键配置都在这里
  - 常看：
    - `phc/data/cfg/sim/default_sim.yaml`
    - `phc/data/cfg/env/env_im_getup_mcp.yaml`
    - `phc/data/cfg/control/default_control.yaml`
- `scripts/data_process/repair_from_fail_filter.py`
  - 生产链路里真正持续跑的 stage2 watcher
- `scripts/data_process/batch_repair_zitai.py`
  - 直接对一个目录批量修复，适合单独 debug
- `scripts/data_process/export_repaired_smpl_npz.py`
  - 把修复 state 导出回标准 SMPL `.npz`
- `checkpoints/`
  - PHC 权重目录
- `data/`
  - 配置和运行所需数据
- `sample_data/`
  - 本地调试样例
- `output/`
  - 修复输出、导出 `.npz`、summary
- `runs/`
  - 训练/实验输出
- `isaacgym/`
  - 仓库内带了一份 Isaac Gym 树，但当前 `docker-compose.yml` 仍然写的是 `../isaacgym`

## 3. 构建容器

在 `/data/smpl_work/ext-phc` 下执行：

```bash
PHC_CODE_PATH=/data/smpl_work/ext-phc \
PHC_DATA_PATH=/data/smpl_work/ext-phc/data \
PHC_OUTPUT_PATH=/data/smpl_work/ext-phc/output \
PHC_RUNS_PATH=/data/smpl_work/ext-phc/runs \
docker compose build phc
```

注意：

- 当前 `docker-compose.yml` 的 `additional_contexts.isaacgym_src` 指向 `../isaacgym`
- 但当前仓库里也有 `ext-phc/isaacgym`
- 新机器如果直接按现在的 compose 重建，必须满足下面二选一：
  - 准备好 sibling 路径 `../isaacgym`
  - 或者先把 compose 里的 build context 改成你实际存在的路径

不要只看到仓库里有 `isaacgym/` 就默认 compose 一定会用它。

## 4. 直接单独调试 repair

### 4.1 用 `batch_repair_zitai.py` 跑单目录

适合单独拿一个输入目录做 repair 验证：

```bash
cd /data/smpl_work/ext-phc

python scripts/data_process/batch_repair_zitai.py \
  --input_root /path/to/input_root \
  --gravity_axis neg_y \
  --primitive_model_path checkpoints/HumanoidIm/phc_3/Humanoid.pth \
  --composer_checkpoint_path checkpoints/HumanoidIm/phc_comp_3/Humanoid.pth \
  --limit 1 \
  --repaired_root output/debug_npz \
  --states_root output/debug_states
```

常用参数：

- `--input_root`
- `--repaired_root`
- `--states_root`
- `--parallel_workers`
- `--limit`
- `--skip_existing`
- `--primitive_model_path`
- `--composer_checkpoint_path`

限制：

- `--parallel_workers > 1` 不能和 `--reuse_gym_viewer` 一起用
- `--parallel_workers > 1` 不能和 `--gym_viewer` 一起用
- `--parallel_workers > 1` 不能和 `--render_o3d` 一起用

也就是说，多并发修复只能 headless 跑。

### 4.2 用 `repair_from_fail_filter.py` 跑生产型 watcher

这是 `smplpipeline` stage2 真正依赖的入口。

典型能力：

- 读取 `fail_filter_pth`
- 找到原始 `.npz`
- 生成切片后的临时 repair 输入
- 调 PHC 修复
- 写成功/失败 summary
- `--watch` 模式下持续轮询新任务

关键参数：

- `--cfg`
- `--source_root`
- `--prepared_root`
- `--phc_motion_root`
- `--states_root`
- `--parallel_workers`
- `--gpu_ids`
- `--watch`
- `--watch_interval`

提醒：

- `data/cfg/data_read_cfg.yaml` 当前仍然是旧机器上的绝对路径示例，不能直接拿去当新机器默认值
- 跨机器 debug 时，优先显式传路径，或者确认上游脚本已经把 cfg 写对

## 5. 结果会写到哪里

如果是 `repair_from_fail_filter.py` 驱动：

- 成功修复后的文件：
  - `results_repair_pth/<seq_dir>/*_repaired.npz`
- 成功 summary：
  - `results_repair_pth/repair_summary.yaml`
- 失败 summary：
  - `results_fail_pth/repair_failures.yaml`
- 临时切片输入：
  - `fail_filter_pth.parent/_prepared_inputs/`

如果是 `batch_repair_zitai.py` 直接跑：

- 修复 `.npz` 默认写到：
  - `output/repaired_smpl_batch/`
- state 默认写到：
  - `output/states/`

## 6. 当前代码里的关键配置

当前分支里，和修复频率直接相关的默认值是：

- `phc/data/cfg/sim/default_sim.yaml`
  - `physx.step_dt: 1/60`
- `phc/data/cfg/env/env_im_getup_mcp.yaml`
  - `controlFrequencyInv: 2`
- `phc/data/cfg/control/default_control.yaml`
  - `decimation: 4`

这些值会影响：

- 仿真步长
- 控制频率
- 最终导出的时间分辨率
- 修复稳定性

不要把它们当成单纯的“提速参数”随便改。

## 7. 最常见的排查顺序

### 7.1 看是不是上游没把活派下来

先确认 `smplpipeline` 那边是否已经产出：

- `filter/fail_filter/`
- stage2 dispatcher 状态文件
- pending repair 队列

如果上游根本没写进来，`ext-phc` 本身不会凭空开始修。

### 7.2 看 repair watcher 是否还活着

重点看：

- `repair_from_fail_filter.py --watch` 进程是否还在
- 是否还有 `phc/run_hydra.py` 子进程
- `repair_summary.yaml` 和 `repair_failures.yaml` 是否还在刷新

### 7.3 看 prepared 输入有没有生成

如果 `_prepared_inputs/` 都没有新增，问题通常在：

- fail manifest 解析
- 原始 `.npz` 路径解析
- dispatcher 到 ext-phc 的路径映射

### 7.4 看是不是 GPU 并发配置不合适

`--parallel_workers` 是同时修几个 clip。

`--gpu_ids 0,0,1,1,2,2,...` 这种写法表示把 worker 轮询分到指定 GPU。

例如：

- `--parallel_workers 12`
- `--gpu_ids 0,0,1,1,2,2,3,3,4,4,5,5`

表示 6 张卡，每张卡 2 个 repair worker。

## 8. 实际工作时的建议

- 优先从 repo 根目录执行命令
- 大批量修复优先 headless，不要开 viewer
- 先确认 checkpoint、data、sample_data 是否齐
- 不要把 `output/`、`runs/` 里的常规产物提交到仓库
- 如果要跨机器重建环境，先确认 `isaacgym` build context 路径，再开始 build
