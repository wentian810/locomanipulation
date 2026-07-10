import os
import time
import argparse
import yaml
import shutil
from yaml.loader import SafeLoader
from utils.log_utils import MyLogger

class CfgParser:
    """
    YAML 配置包装器：
    - 支持点分 key 访问：cfg['output.root'] 等价于 cfg.cfg['output']['root']
    - 支持点分 key 写入：add_cfg('optimization.stage', 'stage1', mode='override')
    - cfg.cfg 保存原始 dict（必要时可直接取原始 stage 配置）
    """

    def __init__(self, config_path=None, cfg_dict=None):
        if config_path is not None:
            self.cfg = self.load_config(config_path)
        else:
            self.cfg = {} if cfg_dict is None else cfg_dict

    def load_config(self, path: str):
        with open(path, "r") as f:
            return yaml.load(f, Loader=SafeLoader)

    def get_cfg(self, key: str, split: str = "."):
        """读取点分 key；缺失 key 直接报错，避免 silent fallback。"""
        keys = key.split(split)
        out = self.cfg
        for k in keys:
            if isinstance(out, dict) and k not in out:
                raise KeyError(f"[CfgParser] Key '{k}' not found while accessing '{key}'")
            elif isinstance(out, list):
                k = int(k)
            out = out[k]
        return out

    def add_cfg(self, key: str, value, mode: str = "add", split: str = "."):
        """
        写入点分 key：
        - add：只允许新增，不允许覆盖
        - override：允许覆盖（命令行 runtime 参数应使用 override）
        """
        if mode not in ("add", "override"):
            raise ValueError("mode must be 'add' or 'override'")

        keys = key.split(split)
        # import pdb; pdb.set_trace()

        out = self.cfg

        # 走到最终写入位置的父 dict（中间层级不存在则自动创建）
        for k in keys[:-1]:
            if k not in out:
                out[k] = {}
            elif not isinstance(out[k], dict):
                raise KeyError(f"[CfgParser] Conflict at key '{k}', expecting dict")
            out = out[k]

        final_key = keys[-1]
        if final_key in out and mode == "add":
            raise KeyError(f"[CfgParser] Key exists '{key}', use override")
        out[final_key] = value

    def dump(self, path: str):
        """保存本次运行的最终配置快照（强制 block style，便于阅读与 diff）。"""
        with open(path, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)

    def __getitem__(self, key: str):
        return self.get_cfg(key)

    def __setitem__(self, name, value):
        self.add_cfg(name, value, mode="override")


def recursive_create_dirs(path_cfg, parent_dir: str):
    """
    根据 output.structure.sub_dirs 创建目录结构，并返回扁平 alias->path 映射。
    说明：alias 只来自叶子节点（str），中间层目录名（如 vis）不作为 alias。
    """
    path_dict = {}

    if isinstance(path_cfg, dict):
        # e.g. {"vis": ["initial", "optimized"]}
        for key, sub_cfg in path_cfg.items():
            sub_dir = os.path.join(parent_dir, key)
            os.makedirs(sub_dir, exist_ok=True)

            new_dict = recursive_create_dirs(sub_cfg, sub_dir)
            _check_conflict(path_dict, new_dict)
            path_dict.update(new_dict)

    elif isinstance(path_cfg, list):
        # e.g. ["log", "backup", {"vis": [...]}, "results"]
        for item in path_cfg:
            new_dict = recursive_create_dirs(item, parent_dir)
            _check_conflict(path_dict, new_dict)
            path_dict.update(new_dict)

    elif isinstance(path_cfg, str):
        # e.g. "log"
        sub_dir = os.path.join(parent_dir, path_cfg)
        os.makedirs(sub_dir, exist_ok=True)
        path_dict[path_cfg] = sub_dir

    return path_dict


def _check_conflict(base_dict, new_dict):
    """防止不同分支生成相同 alias，导致路径索引歧义。"""
    for k in new_dict:
        if k in base_dict:
            raise KeyError(f"[PathConflict] Duplicate directory alias key: {k}")


def initialize_paths(cfg_parser: CfgParser, ):
    """
    创建本次输出目录并写回配置：
    - output.output_dir：outputs/<prefix-timestamp-suffix>
    - output.structure.concrete_dir：alias->path 扁平路径索引
    同时保存 config.yaml 快照并备份代码目录（output.backup）。
    """
    root = cfg_parser["output.root"]
    prefix = cfg_parser["output.exp_dir.prefix"]
    suffix = cfg_parser["output.exp_dir.suffix"]
    use_timestamp = cfg_parser["output.exp_dir.use_timestamp"]

    assert (prefix or suffix or use_timestamp), "At least one of prefix/suffix/use_timestamp should be set to avoid silent overwriting of outputs/"

    name_components = []
    if prefix: 
        name_components.append(prefix)
    if use_timestamp:
        name_components.append(time.strftime("%Y%m%d-%H%M%S"))
    if suffix:
        name_components.append(suffix)

    output_dir = os.path.join(root, f'{"-".join(name_components)}')

    cfg_parser.add_cfg("output.output_dir", output_dir, mode="override")

    sub_cfg = cfg_parser["output.structure.sub_dirs"]
    path_dict = recursive_create_dirs(sub_cfg, output_dir)
    cfg_parser.add_cfg("output.structure.concrete_dir", path_dict, mode="override")

    cfg_parser.dump(os.path.join(output_dir, "config.yaml"))


def parse_args():
    parser = argparse.ArgumentParser(description="Hand-Object Optimization Config Parser")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--file_pattern", type=str, default=None)
    parser.add_argument("--seqID_index", type=int, default=None)
    parser.add_argument("--use_timestamp", type=int, default=0,)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)

    parser.add_argument("--optim_height", type=int, default=None, help="Whether to run height optimization")
    parser.add_argument("--gravity_axis", type=str, required=True, help="Gravity axis for height optimization, e.g. 'y-'")

    parser.add_argument("--check_penetration",  type=int, default=0, help="Whether to run penetration check on the input meshes")
    parser.add_argument("--check_speed", type=int, default=0, help="Whether to run speed check on the penetration detection")
    parser.add_argument("--gravity_alignment", type=int, default=0, help="Whether to run gravity alignment check")

    args = parser.parse_args()
    cfg_path = args.config

    cfg_parser = CfgParser(config_path=cfg_path)

    # input
    if args.target_dir is not None:
        cfg_parser.add_cfg("input.target_dir", args.target_dir, mode="override")
    if args.file_pattern is not None:
        cfg_parser.add_cfg("input.file_pattern", args.file_pattern, mode="override")
    if args.seqID_index is not None:
        cfg_parser.add_cfg("input.seqID_index", args.seqID_index, mode="override")

    # output
    if args.prefix is not None:
        cfg_parser.add_cfg("output.exp_dir.prefix", args.prefix, mode="override")
    if args.suffix is not None:
        cfg_parser.add_cfg("output.exp_dir.suffix", args.suffix, mode="override")
    if args.use_timestamp is not None:
        cfg_parser.add_cfg("output.exp_dir.use_timestamp", args.use_timestamp, mode="override")
    if args.output_root is not None:
        cfg_parser.add_cfg("output.root", args.output_root, mode
        ="override")

    # optim
    if args.optim_height is not None:
        cfg_parser['optimization.gravity.enabled'] = bool(args.optim_height)
    if args.gravity_axis:
        cfg_parser['optimization.gravity.params.gravity_axis'] = args.gravity_axis

    # checks
    if args.check_penetration:
        cfg_parser['check.penetration.enabled'] = True
    if args.check_speed:
        cfg_parser['check.speed.enabled'] = True
    if args.gravity_alignment:
        cfg_parser['check.gravity_alignment.enabled'] = True

    initialize_paths(cfg_parser, )

    # logger
    logger = MyLogger(
        name='main',
        log_dir=cfg_parser['output.structure.concrete_dir.log'],
        log_filename='main.log',
    )
    return cfg_path, cfg_parser, logger


if __name__ == "__main__":
    cfg_path, cfg_parser = parse_args()
    print(f"[Config] Loaded config from: {cfg_path}")
