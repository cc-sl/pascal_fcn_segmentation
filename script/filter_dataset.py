# filter_dataset.py
"""筛选包含目标5类的图片，同时随机保留部分纯背景图片防止过拟合"""
import os
import random
import numpy as np
from PIL import Image
from config import *

TARGET_LABELS = set(VOC_CLASS_MAP.keys()) - {0}   # {15, 7, 8, 12}  不含背景0

def filter_list(list_path, output_path, bg_keep_ratio=0.3, seed=42):
    with open(list_path, 'r') as f:
        ids = [line.strip() for line in f.readlines()]

    kept, bg_only = [], []
    for img_id in ids:
        gt_path = os.path.join(GT_DIR, img_id + '.png')
        gt = np.array(Image.open(gt_path))
        if len(set(np.unique(gt)) & TARGET_LABELS) > 0:
            kept.append(img_id)
        else:
            bg_only.append(img_id)

    # 随机保留一部分纯背景图片
    rng = random.Random(seed)
    rng.shuffle(bg_only)
    n_bg_keep = int(len(bg_only) * bg_keep_ratio)
    kept += bg_only[:n_bg_keep]

    kept.sort()
    with open(output_path, 'w') as f:
        f.write('\n'.join(kept))

    print(f'{list_path}: {len(ids)} 张 → '
          f'前景 {len(kept) - n_bg_keep} + 纯背景保留 {n_bg_keep}/{len(bg_only)} '
          f'= 最终 {len(kept)} 张')
    return len(kept)


if __name__ == '__main__':
    out_dir = os.path.join(ROOT, 'ImageSets/Segmentation')
    filter_list(TRAIN_LIST, os.path.join(out_dir, 'train_filtered.txt'),
                bg_keep_ratio=0.3)      # ← 可调，0.3 表示保留 30% 纯背景图
    filter_list(VAL_LIST,   os.path.join(out_dir, 'val_filtered.txt'),
                bg_keep_ratio=0.3)
