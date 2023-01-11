import torch.nn as nn
from torch.nn.modules.flatten import Flatten
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveMaxPool2d, Linear, Sigmoid


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=3, padding=2),
            ResBlocks(64, 64, 1),
            ResBlocks(64, 128, 2),
            ResBlocks(128, 256, 2),
            ResBlocks(256, 512, 2),
            AdaptiveMaxPool2d(512),
            Flatten(),
            Linear(512, 2),
            Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)


class ResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(num_features=out_channels),
            ReLU(),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_channels),
            ReLU()
        )

        self.shortcut = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1),
            BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
