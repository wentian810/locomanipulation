# usage

## 使用环境

- docker镜像： img_gvhmr2smpl
- conda环境： smpl_env

## 容器构建

可参考 docker-compose.yml 文件， 对应修改映射的路径

## 代码结构

```text
.
├── _record
│   └── record.md
├── assets
│   └── smplh
│       ├── kid_template.npy
│       ├── SMPLH_female.pkl
│       ├── SMPLH_male.pkl
│       └── SMPLH_NEUTRAL.pkl
├── code
│   ├── __init__.py
│   ├── optim
│   │   ├── optimizer.py
│   │   └── smplh_provider.py
│   ├── scripts
│   │   ├── du_data.txt
│   │   ├── mc_scaffold_from_ls.py
│   │   ├── run_fsys_sync.sh
│   │   └── unzip_bash.sh
│   └── utils
│       ├── __init__.py
│       ├── log_utils.py
│       └── statics.py
├── log
├── output
│   ├── ...
│   └── zitai_n10000_idx002_mv-height
│       ├── record.log
│       └── sequence_statistics.csv
├── README.md
├── fsys_struct.log
├── optim_output.log
├── requirement.txt
├── install.sh
└── run_optim.sh
```

## 输入数据

输入数据的格式举例：

对于下载得到的zip文件，以存储服务器上的 syna/synadata-source-wlcb/output/shenbipai/batch1/zitai/zitai_n10000_idx000/motionx/100077_YAGP_Coaching___Mai_Ishiyama12.zip为例，解压之后的格式如下：

```text
"example_folder"/
└── 100077_YAGP_Coaching___Mai_Ishiyama12 # sequence name
    └── 4a0e8966b1a543129dd6ae6ea7f10e56 # sequence id
        ├── 4a0e8966b1a543129dd6ae6ea7f10e56.npz
        ├── global_human_motion.pkl
        ├── hmr4d_results.pt
        └── smpl_4a0e8966b1a543129dd6ae6ea7f10e56.npz
```

## 运行

### 运行入口

运行 run_optim.sh 或直接运行 code/optim/optimizer.py

- 运行之前需要先更新 optimizer.py 中的参数：
  - data_root： 改成目标文件夹， 例如上述的 "example_folder", 会读取该文件夹下所有 sequence 的 global_human_motion.pkl
  - exp_name： 输出文件夹的名称
  - save_optim_result： 是否保存优化后的 smplh 参数
- 其他
  - 建议开 tmux 运行
  - 建议开多个进程，分别处理不同的文件夹下的数据

### 运行过程

- 创建输出文件夹 output/"exp_name"
- 执行数据优化过程

### 运行结果

- 在 output/"exp_name" 文件夹下：
  - record.log 记录了运行过程所有打印的信息，文件末尾统计了运行优化后总体的合格情况
  - sequence_statistics.csv 文件记录了读取的每个序列的 sequence_name, 合格帧数， 总帧数

### 调整查询

- code/utils/statics.py 实例化后调用 load_from_csv 加载目标csv, 之后调用 get_summary 可以指定任意合格率阈值下的统计结果
  - 举例： get_summary([0.95]) 的输出是： 对所有合格帧数占序列帧数95%以上的序列统计其数量，以及占总序列数的比例
  - 如果传入 get_summary 的参数较多， 那么可以用 recursive_print 打印结果（输出树结构的统计信息）
