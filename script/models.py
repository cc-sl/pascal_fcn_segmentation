# models.py
import torch
import torch.nn as nn
import torchvision.models as models


def bilinear_init_convtranspose(module):
    """用双线性插值核初始化 ConvTranspose2d"""
    if not isinstance(module, nn.ConvTranspose2d):
        return
    assert module.kernel_size[0] == module.kernel_size[1], '只支持正方形核'
    k = module.kernel_size[0]
    factor = (k + 1) // 2
    if k % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = 1.0 - torch.abs(torch.arange(k).float() - center) / factor
    og = torch.clamp(og, min=0.0)
    kernel = og[:, None] * og[None, :]
    kernel = kernel / kernel.sum()
    kernel = kernel[None, None, :, :].repeat(
        module.out_channels, module.in_channels, 1, 1
    )
    with torch.no_grad():
        module.weight.copy_(kernel)
        if module.bias is not None:
            module.bias.zero_()


class FCN8s(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=5):
        super(FCN8s, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone_out_channels = [128, 256, 512]
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.backbone_out_channels = [128, 256, 512]
        else:
            raise ValueError("Only resnet18/resnet34 supported")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        mid_channels = 256

        self.fc6 = nn.Sequential(
            nn.Conv2d(512, mid_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        self.score_layer4 = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self.score_layer3 = nn.Conv2d(self.backbone_out_channels[1], num_classes, kernel_size=1)
        self.score_layer2 = nn.Conv2d(self.backbone_out_channels[0], num_classes, kernel_size=1)

        self.upsample2_stage1 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )
        self.upsample2_stage2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )
        self.upsample8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4
        )

        self.apply(bilinear_init_convtranspose)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.fc6(x4)
        x4 = self.fc7(x4)

        score4 = self.score_layer4(x4)
        score3 = self.score_layer3(x3)
        score2 = self.score_layer2(x2)

        upscore4 = self.upsample2_stage1(score4)
        fusion1 = upscore4 + score3
        upscore_fusion1 = self.upsample2_stage2(fusion1)
        fusion2 = upscore_fusion1 + score2
        out = self.upsample8(fusion2)

        return out
