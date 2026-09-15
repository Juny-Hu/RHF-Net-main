import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        if in_channels != 3:
            pretrained_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                pretrained_conv.out_channels,
                kernel_size=pretrained_conv.kernel_size,
                stride=pretrained_conv.stride,
                padding=pretrained_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                adapted_weights = pretrained_conv.weight.mean(dim=1, keepdim=True)
                adapted_weights = adapted_weights.repeat(1, in_channels, 1, 1)
                adapted_weights *= 3.0 / in_channels
                backbone.conv1.weight.copy_(adapted_weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        return torch.flatten(self.features(x), 1)


class RHFNet(nn.Module):
    def __init__(self, num_classes=4, hsi_channels=20, dropout=0.3):
        super().__init__()
        self.rgb_backbone = ResNet18FeatureExtractor(3)
        self.hsi_backbone = ResNet18FeatureExtractor(hsi_channels)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, rgb, hsi):
        rgb_features = self.rgb_backbone(rgb)
        hsi_features = self.hsi_backbone(hsi)
        fused_features = torch.cat((rgb_features, hsi_features), dim=1)
        return self.classifier(fused_features)
