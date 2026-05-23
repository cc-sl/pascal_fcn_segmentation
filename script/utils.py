# utils.py
import random
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from config import NUM_CLASSES, IGNORE_INDEX

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Timer:
    def __init__(self):
        self.start_time = None
    def start(self):
        self.start_time = time.time()
    def stop(self):
        return time.time() - self.start_time

# 5类的调色板
PALETTE = [
    [0, 0, 0],        # 背景 - 黑
    [255, 0, 0],      # 人 - 红
    [0, 0, 255],      # 车 - 蓝
    [0, 255, 0],      # 猫 - 绿
    [255, 255, 0]     # 狗 - 黄
]

def colorize_mask(mask):
    """ 将标签索引图转为 RGB 彩色图 """
    mask = mask.cpu().numpy().astype(np.uint8)
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(NUM_CLASSES):
        color_mask[mask == label] = PALETTE[label]
    return color_mask

class AverageMeter:
    """ 计算均值 """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
