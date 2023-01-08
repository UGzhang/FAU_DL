import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass


class ResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
