import torch
from torch import nn
from torch.nn import functional as F
from .blocks import WeightNormConv2d, ResBlock, BottleneckBlock

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Initial layer
        self.conv1 = WeightNormConv2d(
            3, 64, kernel_size=3, stride=1, padding=1
        )
        self.maxpool = nn.Identity()  # skip pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * (block.expansion if hasattr(block, "expansion") else 1), num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        expansion = block.expansion if hasattr(block, "expansion") else 1

        if stride != 1 or self.in_channels != out_channels * expansion:
            downsample = nn.Sequential(
                WeightNormConv2d(self.in_channels, out_channels * expansion,
                                 kernel_size=1, stride=stride)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels * expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes=33):
    return ResNet(ResBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=33):
    return ResNet(ResBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=33):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)

# def resnet101(num_classes=33):
#     return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes)

# def resnet152(num_classes=33):
#     return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes=num_classes)