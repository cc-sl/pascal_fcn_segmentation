# evaluate.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import *
from dataset import VOCSegDataset
from models import FCN8s
from utils import set_seed

def compute_confusion_matrix(model, dataloader, device, num_classes):
    """ 累积混淆矩阵 """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            valid_mask = (labels != IGNORE_INDEX)
            # 展平，只保留有效像素
            labels_flat = labels[valid_mask].cpu().numpy()
            preds_flat = preds[valid_mask].cpu().numpy()
            # 累加到混淆矩阵
            for gt, pred in zip(labels_flat, preds_flat):
                confusion_matrix[gt, pred] += 1
    return confusion_matrix

def calculate_iou(confusion_matrix):
    """ 计算各类别 IoU 和 mIoU """
    ious = []
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        denom = tp + fp + fn
        if denom == 0:
            iou = float('nan')
        else:
            iou = tp / denom
        ious.append(iou)
    miou = np.nanmean(ious)
    return ious, miou

def evaluate(backbone):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCN8s(backbone=backbone, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'{backbone}_best.pth'), map_location=device))

    model.eval()
    print("权重加载成功，模型已切换为评估模式")

    val_dataset = VOCSegDataset(VAL_LIST, split='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    conf_mat = compute_confusion_matrix(model, val_loader, device, NUM_CLASSES)
    ious, miou = calculate_iou(conf_mat)

    # 计算像素准确率
    pa = np.diag(conf_mat).sum() / conf_mat.sum()

    class_names = ['bg', 'person', 'car', 'cat', 'dog']
    print(f'\nBackbone: {backbone}')
    print(f'{"Class":<10} {"IoU":>6}')
    for c in range(NUM_CLASSES):
        print(f'{class_names[c]:<10} {ious[c]:.4f}')
    print(f'{"mIoU":<10} {miou:.4f}')
    print(f'Pixel Accuracy: {pa:.4f}')
    return miou, pa

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    args = parser.parse_args()
    evaluate(args.backbone)
