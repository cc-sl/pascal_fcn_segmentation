import os

# 自动获取当前 config.py 所在目录（script 文件夹）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
ROOT = r'c:\code1\pascal_fcn_segmentation\VOCdevkit\VOC2012'
IMAGE_DIR = os.path.join(ROOT, 'JPEGImages')
GT_DIR = os.path.join(ROOT, 'SegmentationClass')

# 训练/验证列表文件
# TRAIN_LIST = os.path.join(ROOT, 'ImageSets/Segmentation/train.txt')
# VAL_LIST = os.path.join(ROOT, 'ImageSets/Segmentation/val.txt')
TRAIN_LIST = os.path.join(ROOT, 'ImageSets/Segmentation/train_filtered.txt')
VAL_LIST = os.path.join(ROOT, 'ImageSets/Segmentation/val_filtered.txt')

# 类别映射：VOC 原始标签 -> 新5类索引
VOC_CLASS_MAP = {
    0: 0,   # 背景
    15: 1,  # 人
    7: 2,   # 车
    8: 3,   # 猫
    12: 4   # 狗
}
NUM_CLASSES = 5
IGNORE_INDEX = 255

# 训练超参数
INPUT_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 10
LR_GAMMA = 0.1

# 随机种子
SEED = 42

# 模型保存路径
SAVE_DIR = r'c:\code1\pascal_fcn_segmentation\script\checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)