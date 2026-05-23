# models.py
import torch.nn as nn
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=5):
        super(FCN8s, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.backbone_out_channels = [128, 256]   # layer2 和 layer3 的输出通道
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.backbone_out_channels = [128, 256]
        else:
            raise ValueError("Only resnet18/resnet34 supported")

        # 前几层：conv1, bn1, relu, maxpool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 四个连续的 block 层
        self.layer1 = resnet.layer1   # 输出 1/4
        self.layer2 = resnet.layer2   # 输出 1/8
        self.layer3 = resnet.layer3   # 输出 1/16
        self.layer4 = resnet.layer4   # 输出 1/32

        # 分类头：对 layer4 输出做 1x1 卷积得到类别预测
        self.score_layer4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_layer3 = nn.Conv2d(self.backbone_out_channels[1], num_classes, kernel_size=1)
        self.score_layer2 = nn.Conv2d(self.backbone_out_channels[0], num_classes, kernel_size=1)

        # 上采样层
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        # 编码器
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)   # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        # 分类分数
        score4 = self.score_layer4(x4)          # 1/32
        score3 = self.score_layer3(x3)          # 1/16
        score2 = self.score_layer2(x2)          # 1/8

        # 上采样并融合
        upscore4 = self.upsample2(score4)        # 1/16
        fusion1 = upscore4 + score3
        upscore_fusion1 = self.upsample2(fusion1) # 1/8
        fusion2 = upscore_fusion1 + score2
        out = self.upsample8(fusion2)            # 原始大小

        return out
