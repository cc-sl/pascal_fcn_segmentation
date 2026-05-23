## 🔍 问题分析：为什么前景 IoU 全部为 0

### 根本原因：`CrossEntropyLoss` 缺少类别权重

```python
# train.py 第 52 行
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # ❌ 没有 weight
```

在 Pascal VOC 数据集中，**背景像素约占总像素的 87%**。当你不设置类别权重时，模型很快发现一条"捷径"：

> **永远预测背景（class 0），损失就已经很低了。**

模型没有梯度信号去学习前景类，导致对 person/car/cat/dog 的预测概率始终接近 0。

| 现象      | 数值   | 含义                          |
| --------- | ------ | ----------------------------- |
| bg IoU    | 0.868  | 背景预测正常                  |
| 前景 IoU  | 0.000  | 模型从不预测前景              |
| Pixel Acc | 0.868  | ≈ 背景占比，验证了"全预测bg" |
| mIoU      | 0.1736 | = 0.868 / 5，完全由 bg 贡献   |

---

### 次要问题

| # | 问题                                           | 位置                  | 影响                                                |
| - | ---------------------------------------------- | --------------------- | --------------------------------------------------- |
| 1 | **无类别权重**                           | `train.py:52`       | 🔴 主因，模型坍缩到背景                             |
| 2 | **`backbone_out_channels` 用常量写死** | `models.py:16-17`   | 🟡 resnet18/34 恰好都是 [128,256]，没问题但不可扩展 |
| 3 | **`evaluate.py` 逐像素循环**           | `evaluate.py:22-23` | 🟡 极慢但结果正确                                   |

---

## 🛠️ 修复方案

### 修复 1（最关键）：在 `train.py` 中添加类别权重

```python
# 在 main() 函数中，criterion 定义之前，添加：
# 计算类别权重：背景占比高 → 权重低；前景占比低 → 权重高
# 可以从训练集中统计，也可以用经验值
class_weights = torch.tensor([0.3, 3.0, 3.0, 3.0, 3.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
```

更好的做法是**从训练集中自动计算权重**：

```python
def compute_class_weights(dataset, num_classes, ignore_index):
    """统计训练集中各类别像素数，计算反比权重"""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, gt in dataset:
        gt_np = gt.numpy()
        for c in range(num_classes):
            counts[c] += (gt_np == c).sum()
  
    # 反比权重，归一化
    counts = np.maximum(counts, 1)  # 避免除零
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # 归一化
    return torch.tensor(weights, dtype=torch.float32)

# 在 main() 中使用：
train_dataset = VOCSegDataset(TRAIN_LIST, split='train')
class_weights = compute_class_weights(train_dataset, NUM_CLASSES, IGNORE_INDEX)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=IGNORE_INDEX)
```

---

### 修复 2（推荐）：使用更鲁棒的损失函数

仅仅加权重可能还不够，**建议组合 Dice Loss**，这对小目标前景类别非常有效：

```python
class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W), target: (B, H, W)
        pred = F.softmax(pred, dim=1)
        dice = 0.0
        for c in range(self.num_classes):
            pred_c = pred[:, c, :, :]
            target_c = (target == c).float()
            # 忽略 ignore_index
            mask = (target != self.ignore_index)
            pred_c = pred_c * mask
            target_c = target_c * mask
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice += 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice / self.num_classes

# 组合损失
criterion_ce = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=IGNORE_INDEX)
criterion_dice = DiceLoss(NUM_CLASSES, ignore_index=IGNORE_INDEX)

# 训练时：
loss = criterion_ce(outputs, labels) + criterion_dice(outputs, labels)
```

---

### 修复 3：优化 `evaluate.py` 的混淆矩阵计算

向量化替代逐像素循环：

```python
def compute_confusion_matrix(model, dataloader, device, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            valid_mask = (labels != IGNORE_INDEX)
          
            labels_valid = labels[valid_mask].long()
            preds_valid = preds[valid_mask].long()
          
            # 向量化累加（快几十倍）
            mask = labels_valid * num_classes + preds_valid
            cm = np.bincount(mask.cpu().numpy(), minlength=num_classes * num_classes)
            confusion_matrix += cm.reshape(num_classes, num_classes)
    return confusion_matrix
```

---

## 📊 验证步骤

修复后，添加调试代码确认模型不再全部预测背景：

```python
# 放在 evaluate() 的 compute_confusion_matrix 中
preds = torch.argmax(outputs, dim=1)
for c in range(NUM_CLASSES):
    count = (preds == c).sum().item()
    print(f"  pred class {c}: {count} pixels")
```

期望看到：前景类别有非零预测。

---

## 📌 总结

| 修复                             | 位置                  | 重要性  |
| -------------------------------- | --------------------- | ------- |
| CrossEntropyLoss 添加 `weight` | `train.py:52`       | 🔴 必须 |
| 加 Dice Loss                     | `train.py`          | 🟡 推荐 |
| 向量化混淆矩阵                   | `evaluate.py:22-23` | 🟢 可选 |

**核心结论**：你的模型架构和数据处理没有逻辑错误，问题完全出在损失函数没有处理极端的类别不平衡。给背景低权重、前景高权重，然后重新训练，前景 IoU 就会从 0 显著提升。
