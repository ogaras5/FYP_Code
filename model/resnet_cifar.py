from __future__ import absolute_import
"""
ResNet implementation for Cifar dataset.
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
and
https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/resnet.py
"""

import torch.nn as nn
import math

__all__ = ['resnet']

class BasicBlock(nn.Module):
    """
    Residual layer with two 3x3 filter convolutional layers
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x

class BottleNeck(nn.Module):
    """
    Residual Layer with 3 convolutional layers
    The layer consists of two 1x1 filter convolutional layers,
    and one 3x3 filter convolutional layer
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # As noted in "Deep Residual Learning for Image Recognition"
        # Number of layers needs to be 6n + 2, to match ImageNet setup.
        assert (depth - 2) % 6 == 0, 'depth should be 6n + 2'
        n = (depth - 2) // 6

        block = BottleNeck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        seld.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.__make_layer(block, 16, n)
        self.layer2 = self.__make_layer(block, 32, n, stride=2)
        self.layer3 = self.__make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample == nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplane, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  # 32 x 32
            x = self.layer2(x)  # 16 x 16
            x = self.layer3(x)  # 8 x 8

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

def resnet(**kwargs):
    """
    Construct a ResNet model.
    """
    return ResNet(**kwargs)
