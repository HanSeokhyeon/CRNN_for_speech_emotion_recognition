'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import math


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, feature_size):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        feature_size = math.ceil(feature_size / 2)
        feature_size = math.ceil(feature_size / 2)
        # feature_size = math.ceil(feature_size / 2)
        # feature_size = math.ceil(feature_size / 2)
        self.feature_size = feature_size * 256

        self.a3 = Inception(64,  32, 48, 64, 8, 16, 16)
        self.aa3 = Inception(128, 32, 48, 64, 8, 16, 16)
        self.b3 = Inception(128, 64, 96, 128, 16, 32, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.maxpool(out)
        out = self.aa3(out)
        out = self.aa3(out)
        out = self.maxpool(out)
        out = self.b3(out)
        # out = self.maxpool(out)
        # out = self.a4(out)
        # out = self.b4(out)
        # out = self.c4(out)
        # out = self.d4(out)
        # out = self.e4(out)
        # out = self.maxpool(out)
        # out = self.a5(out)
        # out = self.b5(out)

        out = out.transpose(1, 2)
        out = out.contiguous()
        sizes = out.size()
        out = out.view(sizes[0], sizes[1], sizes[2] * sizes[3])

        return out
