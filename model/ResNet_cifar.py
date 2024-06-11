import torch.nn as nn
import torch
import random
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from .utils import *


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet_modify(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100, nf=64, etf_cls=False, fnorm='none'):
        super(ResNet_modify, self).__init__()
        self.in_planes = nf
        self.num_classes = num_classes
        self.etf_cls = etf_cls
        self.fnorm = fnorm

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.out_dim = 4 * nf * block.expansion
        if fnorm == 'nn1':
            self.bn4 = nn.BatchNorm1d(self.out_dim, affine=False)
            bias = False
        elif fnorm == 'nn2':  # batch norm, normalize feature
            self.bn4 = nn.BatchNorm1d(self.out_dim)
            self.fc5 = nn.Linear(self.out_dim, self.out_dim)
            self.bn5 = nn.BatchNorm1d(self.out_dim, affine=False)
            bias = False
        elif fnorm == 'none' or fnorm == 'null':
            bias = True

        self.fc = nn.Linear(self.out_dim, num_classes, bias=bias)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)
        self.fc_cb = nn.Linear(self.out_dim, num_classes, bias=bias)

        hidden_dim = 128
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),

        )
        self.apply(_weights_init)

        if etf_cls:
            weight = torch.sqrt(torch.tensor(num_classes / (num_classes - 1))) * (
                    torch.eye(num_classes) - (1 / num_classes) * torch.ones((num_classes, num_classes)))
            weight /= torch.sqrt((1 / num_classes * torch.norm(weight, 'fro') ** 2))  # [K, K]

            self.fc_cb.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, self.out_dim)))  # [K, d]
            self.fc_cb.weight.requires_grad_(False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, ret=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        if self.fnorm == 'nn1':
            feature = self.bn4(feature)
        elif self.fnorm == 'nn2':
            feature = self.bn4(feature)
            feature = self.fc5(feature)
            feature = self.bn5(feature)
            
        if self.fnorm == 'nn1' or self.fnorm == 'nn2':
            feature = F.normalize(feature, p=2, dim=-1)

        if ret is None:
            out = self.fc_cb(feature)
            return out
        elif ret == 'o':
            out = self.fc_cb(feature)
            return out
        elif ret == 'of':
            out = self.fc_cb(feature)
            return out, feature
        elif ret == 'all':
            out = self.fc(feature)
            out_cb = self.fc_cb(feature)
            z = self.projection_head(feature)
            p = self.contrast_head(z)
            return out, out_cb, z, p, feature

    def forward_mixup(self, x, target=None, mixup=None, mixup_alpha=None):

        if mixup >= 0 and mixup <= 3:
            layer_mix = mixup
        elif mixup == 9:
            layer_mix = random.randint(0, 3)
        else:
            layer_mix = None

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.tensor([lam], dtype=torch.float32, device=x.device)
            lam = torch.autograd.Variable(lam)

        target = to_one_hot(target, self.num_classes)
        if layer_mix == 0:
            x, target = mixup_process(x, target, lam)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        if layer_mix == 1:
            x, target = mixup_process(x, target, lam)

        x = self.layer2(x)
        if layer_mix == 2:
            x, target = mixup_process(x, target, lam)

        x = self.layer3(x)
        if layer_mix == 3:
            x, target = mixup_process(x, target, lam)

        feat = F.avg_pool2d(x, x.size()[3])
        feat = feat.view(feat.size(0), -1)
        out = self.fc_cb(feat)

        return out, target, feat


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_class=100, etf_cls=False):
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_class

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper so conv2_x's stride is 1

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # from torch.nn import w

        self.fc = nn.Linear(512 * block.expansion, num_class)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)
        hidden_dim=256
        self.fc_cb = nn.Linear(512 * block.expansion, num_class)
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(512 * block.expansion, hidden_dim),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, ret=None):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        feature = output.view(output.size(0), -1)
        if ret== 'all':
            out = self.fc(feature)
            out_cb = self.fc_cb(feature)
            z = self.projection_head(feature)
            p = self.contrast_head(z)
            return out, out_cb, z, p, feature
        elif ret == 'of':
            out = self.fc_cb(feature)
            return out, feature
        else:
            out = self.fc_cb(feature)
            return out

    def forward_mixup(self, x, target=None, mixup=None, mixup_alpha=None):
        if mixup >= 0 and mixup <= 5:
            layer_mix = mixup
        elif mixup == 9:
            layer_mix = random.randint(0, 5)
        else:
            layer_mix = None

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.tensor([lam], dtype =torch.float32, device=x.device)
            lam = torch.autograd.Variable(lam)

        target = to_one_hot(target, self.num_classes)
        if layer_mix == 0:
            x, target = mixup_process(x, target, lam)

        x = self.conv1(x)
        if layer_mix == 1:
            x, target = mixup_process(x, target, lam)

        x = self.conv2_x(x)
        if layer_mix == 2:
            x, target = mixup_process(x, target, lam)

        x = self.conv3_x(x)
        if layer_mix == 3:
            x, target = mixup_process(x, target, lam)

        x = self.conv4_x(x)
        if layer_mix == 4:
            x, target = mixup_process(x, target, lam)

        x = self.conv5_x(x)
        if layer_mix == 5:
            x, target = mixup_process(x, target, lam)

        x = self.avg_pool(x)
        feat = x.view(x.size(0), -1)

        out = self.fc_cb(feat)
        return out, target, feat


def resnet18(num_class=100, etf_cls=False):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class=num_class, etf_cls=etf_cls)

def resnet32(num_class=10, etf_cls=False, fnorm='none'):
    return ResNet_modify(BasicBlock_s, [5, 5, 5], num_classes=num_class, etf_cls=etf_cls, fnorm=fnorm)

def resnet34(num_class=100, etf_cls=False):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class=num_class, etf_cls=etf_cls)


def resnet50(num_class=100, etf_cls=False):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_class=num_class, etf_cls=etf_cls)


def resnet101(num_class=100, etf_cls=False):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_class=num_class, etf_cls=etf_cls)


def resnet152(num_class=100, etf_cls=False):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_class=num_class, etf_cls=etf_cls)
