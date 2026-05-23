# dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from config import *

class VOCSegDataset(Dataset):
    def __init__(self, list_path, split='train'):
        with open(list_path, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.split = split
        self.ignore_index = IGNORE_INDEX

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(IMAGE_DIR, img_id + '.jpg')
        gt_path = os.path.join(GT_DIR, img_id + '.png')
        
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path)          # PIL 模式为 'P'，即调色板模式，读取值为类别索引

        # 调整大小到 INPUT_SIZE x INPUT_SIZE
        img = TF.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=Image.BILINEAR)
        gt = TF.resize(gt, (INPUT_SIZE, INPUT_SIZE), interpolation=Image.NEAREST)

        # 数据增强：训练时随机水平翻转
        if self.split == 'train':
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                gt = TF.hflip(gt)

        # 转为 tensor
        img = TF.to_tensor(img)  # 范围 [0,1]
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        gt = torch.as_tensor(np.array(gt), dtype=torch.int64)  # (H, W)

        # 将原始标签映射到5类
        gt_mapped = torch.full_like(gt, self.ignore_index)  # 全设为忽略
        for orig_label, new_label in VOC_CLASS_MAP.items():
            gt_mapped[gt == orig_label] = new_label

        return img, gt_mapped
