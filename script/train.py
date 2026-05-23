# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm

from config import *
from dataset import VOCSegDataset
from models import FCN8s
from utils import set_seed, Timer


class DiceLoss(nn.Module):
    """多类别 Dice Loss，对小目标前景类别很有效"""
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)          # (B, C, H, W)
        dice = 0.0
        for c in range(self.num_classes):
            pred_c = pred[:, c, :, :]
            target_c = (target == c).float()
            mask = (target != self.ignore_index).float()
            pred_c = pred_c * mask
            target_c = target_c * mask
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice += 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice / self.num_classes


def compute_class_weights(dataset, num_classes, ignore_index):
    counts = np.zeros(num_classes, dtype=np.float64)
    print("计算类别权重中...")
    for _, gt in tqdm(dataset, desc="统计类别频率"):
        gt_np = gt.numpy()
        for c in range(num_classes):
            counts[c] += (gt_np == c).sum()
    print(f"各类别像素数: {counts}")
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    print(f"类别权重: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


def main(backbone):
    set_seed(SEED)

    # 数据集
    train_dataset = VOCSegDataset(TRAIN_LIST, split='train')
    val_dataset = VOCSegDataset(VAL_LIST, split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCN8s(backbone=backbone, num_classes=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(train_dataset, NUM_CLASSES, IGNORE_INDEX)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=IGNORE_INDEX)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    criterion_dice = DiceLoss(NUM_CLASSES, IGNORE_INDEX)
    
    def criterion(outputs, labels):
        return criterion_ce(outputs, labels) + criterion_dice(outputs, labels)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    best_val_loss = float('inf')
    timer = Timer()
    timer.start()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('LR:', optimizer.param_groups[0]['lr'])
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{backbone}_best.pth'))
            print('  -> Best model saved')

    total_time = timer.stop()
    print(f'\nTraining finished. Total time: {total_time:.2f} seconds')
    return total_time


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    args = parser.parse_args()
    main(args.backbone)
