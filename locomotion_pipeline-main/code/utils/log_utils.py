# code/utils/log_utils.py
"""
日志配置模块
"""

import logging
import os
import sys
from termcolor import colored

import torch
import numpy as np

class MyLogger:
    def __init__(
        self, 
        name: str = 'default.log', 
        level: int = logging.INFO, 
        log_filename: str = None, 
        log_dir: str = None,
        log_mode: str = 'w'
    ):
        if log_dir is None:
            # 默认：当前文件的上上级目录作为项目根，然后根目录下 log/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../.."))
            log_dir = os.path.join(project_root, "log")

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, log_filename)

        logging.basicConfig(
            filename = self.log_file, 
            level = level, 
            filemode = log_mode, 
            format = '%(levelname)s:%(asctime)s:%(message)s', 
            datefmt = '%Y-%d-%m %H:%M:%S'
        )

        self.logger = logging.getLogger(name)
        self.color_map = {
            logging.INFO: 'green', 
            logging.WARNING: 'yellow', 
            logging.ERROR: 'red', 
            logging.DEBUG: 'blue'
        }
    
    def message(self, msg, level=logging.INFO):
        self.logger.log(level, msg)
        print(colored(msg, self.color_map.get(level, 'white')))

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def export(self, data, filename=None):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        if isinstance(data, np.ndarray):
            if filename is None:
                filename = self.log_file.replace(".log", "_exported.npy")
            np.save(filename, data)
        else:
            self.logger.warning(f"Unsupported data type for export: {type(data)}")

    def recursive_log(
        self, 
        data, 
        prefix="", 
        traversal_iter: bool=True
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.recursive_log(value, prefix + f"{key}.", traversal_iter)
        elif isinstance(data, list) or isinstance(data, tuple) or isinstance(data, set):
            if traversal_iter:
                for idx, item in enumerate(data):
                    self.recursive_log(item, prefix + f"[{idx}].", traversal_iter)
            else:
                self.logger.info(f"{prefix}: {data}")
        else:
            self.logger.info(f"{prefix}: {data}")
