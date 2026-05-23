# visualize.py
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from config import *
from dataset import VOCSegDataset
from models import FCN8s
from utils import set_seed, colorize_mask

def visualize(backbone, num_samples=3):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCN8s(backbone=backbone, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'{backbone}_best.pth'), map_location=device))
    model.eval()

    val_dataset = VOCSegDataset(VAL_LIST, split='val')
    indices = random.sample(range(len(val_dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        img, gt = val_dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_batch)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu()

        # 反标准化图像以便显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_vis = img * std + mean
        img_vis = img_vis.permute(1,2,0).numpy()
        img_vis = np.clip(img_vis, 0, 1)

        gt_vis = colorize_mask(gt)
        pred_vis = colorize_mask(pred)

        axes[i][0].imshow(img_vis)
        axes[i][0].set_title('Original Image')
        axes[i][0].axis('off')

        axes[i][1].imshow(gt_vis)
        axes[i][1].set_title('Ground Truth')
        axes[i][1].axis('off')

        axes[i][2].imshow(pred_vis)
        axes[i][2].set_title('Prediction')
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig(f'segmentation_{backbone}.png')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()
    visualize(args.backbone, args.num_samples)
