#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import config

def print_absolute_paths():
    # 获取 config.py 自身所在目录
    config_dir = os.path.dirname(os.path.abspath(config.__file__))
    
    print("===== config.py 中所有路径的 绝对路径 =====\n")

    # 遍历所有需要打印的路径变量
    path_vars = [
        "ROOT",
        "IMAGE_DIR",
        "GT_DIR",
        "TRAIN_LIST",
        "VAL_LIST",
        "SAVE_DIR",
    ]

    for var_name in path_vars:
        if hasattr(config, var_name):
            rel_path = getattr(config, var_name)
            # 转绝对路径
            abs_path = os.path.abspath(os.path.join(config_dir, rel_path))
            print(f"{var_name:<12} = {abs_path}")

if __name__ == "__main__":
    print_absolute_paths()