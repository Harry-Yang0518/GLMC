import torch.nn as nn
import torch
import random
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from .utils import *


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features,):
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        logits = F.linear(F.normalize(input), F.normalize(self.weight))   # [B, 10]
        return logits.clamp(-1, 1)


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BNSequential(nn.Sequential):
    def forward(self, x, label):
        for layer in self:
            if isinstance(layer, BalancedBatchNorm2d) or isinstance(layer, BasicBlock_s):
                x = layer(x, label)
            else:
                x = layer(x)
        return x


class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', norm_layer=nn.BatchNorm2d):
        super(BasicBlock_s, self).__init__()
        self.cbr1 = BNSequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(planes),
            nn.ReLU()
        )
        self.cb2 = BNSequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(planes)
        )

        self.shortcut = BNSequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = BNSequential(LambdaLayer(lambda x:
                                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",0)
                                                         )
                                             )
            elif option == 'B':
                self.shortcut = BNSequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    norm_layer(self.expansion * planes)
                )

    def forward(self, x, label):
        out = self.cbr1(x,  label)
        out = self.cb2(out, label)
        out += self.shortcut(x, label)
        out = F.relu(out)
        return out


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes * IRBlock.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * IRBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * IRBlock.expansion),
            )
        else:
            self.downsample = None
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

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


# ======================== modified ResNet ========================
class ResNet_modify(nn.Module):

    def __init__(self, block, num_blocks, nf=64, args=None):
        super(ResNet_modify, self).__init__()
        self.in_planes = nf
        self.args = args
        self.num_classes = args.num_classes
        self.etf_cls = args.etf_cls

        if args.bn_type == 'cbn':
            norm_layer = BalancedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        self.cbr = BNSequential(nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                                norm_layer(self.in_planes),
                                nn.ReLU()
                                )

        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2, norm_layer=norm_layer)
        self.out_dim = 4 * nf * block.expansion

        if self.args.feat == 'none' or self.args.feat == 'null':
            self.feature = BNSequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Flatten()
                                         )
        elif self.args.feat == 'b':
            self.feature = BNSequential(norm_layer(self.out_dim),
                                         nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Flatten()
                                         )
        elif self.args.feat == 'bfb':                                              # still need to fix norm layer
            self.feature = BNSequential(nn.BatchNorm2d(self.out_dim),
                                        nn.Flatten(),
                                        nn.Linear(self.out_dim * 4 * 4, self.out_dim),
                                        nn.BatchNorm1d(self.out_dim)
                                        )
        elif self.args.feat.startswith('bfb_d'):  # with_dropout
            dropout_rate = float(self.args.feat.replace('bfb_d', ''))
            self.feature = BNSequential(nn.BatchNorm2d(self.out_dim),
                                         nn.Flatten(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(self.out_dim * 4 * 4, self.out_dim),
                                         nn.BatchNorm1d(self.out_dim)
                                         )

        if args.loss.endswith('m'):  # m for margin loss, linear layer which normalize both feature and weight w
            self.fc = LinearLayer(self.out_dim, self.num_classes)
        else:
            self.fc = nn.Linear(self.out_dim, self.num_classes, bias=self.args.bias)  # may need to change the bias
            self.apply(_weights_init)

        if self.etf_cls:
            weight = torch.sqrt(torch.tensor(self.num_classes / (self.num_classes - 1))) * (
                    torch.eye(self.num_classes) - (1 / self.num_classes) * torch.ones((self.num_classes, self.num_classes)))
            weight /= torch.sqrt((1 / self.num_classes * torch.norm(weight, 'fro') ** 2))  # [K, K]

            self.fc.weight = nn.Parameter(torch.mm(weight, torch.eye(self.num_classes, self.out_dim)))  # [K, d]
            self.fc.weight.requires_grad_(False)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer=nn.BatchNorm2d):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer=norm_layer))
            self.in_planes = planes * block.expansion

        return BNSequential(*layers)

    def forward(self, x, label, ret=None):
        #x = x.float()  
        out = self.cbr(x, label)
        out = self.layer1(out, label)
        out = self.layer2(out, label)
        out = self.layer3(out, label)

        feat = self.feature(out, label)
        if self.args.norm:
            feat = F.normalize(feat, p=2, dim=-1)
        out = self.fc(feat)

        if ret == 'of':
            return out, feat
        else:
            return out

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

        feat = self.feature(x)
        out = self.fc(feat)

        return out, target, feat


class ResNet(nn.Module):

    def __init__(self, block, num_block, args=None):
        super().__init__()
        self.in_channels = 64
        self.num_class = args.num_classes
        self.args = args

        layer_kwargs = {}
        if 'use_se' in args and args.use_se:
            layer_kwargs = {'use_se': args.use_se}

        if args.arch.startswith('iresent'):
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))

        # we use a different input_size than the original paper so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64,  num_block[0], 1, **layer_kwargs)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, **layer_kwargs)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, **layer_kwargs)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, **layer_kwargs)
        self.out_dim = 512 * block.expansion

        if self.args.feat == 'none' or self.args.feat == 'null':
            self.feature = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Flatten()
                                         )
        elif self.args.feat == 'b':
            self.feature = nn.Sequential(nn.BatchNorm2d(512 * block.expansion),
                                         nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Flatten()
                                         )
        elif self.args.feat == 'bfb':
            self.feature = nn.Sequential(nn.BatchNorm2d(512 * block.expansion),
                                         nn.Flatten(),
                                         nn.Linear(512 * block.expansion * 4 * 4, 512 * block.expansion),
                                         nn.BatchNorm1d(512 * block.expansion)
            )
        elif self.args.feat.startswith('bfb_d'):   # with_dropout
            dropout_rate = float(self.args.feat.replace('bfb_d', ''))
            self.feature = nn.Sequential(nn.BatchNorm2d(512 * block.expansion),
                                         nn.Flatten(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(512 * block.expansion * 4 * 4, 512 * block.expansion),
                                         nn.BatchNorm1d(512 * block.expansion)
                                         )
        if args.loss.endswith('m'):  # m for margin
            self.fc = LinearLayer(512 * block.expansion, self.num_class)
        else:
            self.fc = nn.Linear(512 * block.expansion, self.num_class, bias=self.args.bias)                   # may need to change the bias
            self.apply(_weights_init)

        if self.args.etf_cls:
            weight = torch.sqrt(torch.tensor(self.num_class / (self.num_class - 1))) * (
                    torch.eye(self.num_class) - (1 / self.num_class) * torch.ones((self.num_class, self.num_class)))
            weight /= torch.sqrt((1 / self.num_class * torch.norm(weight, 'fro') ** 2))  # [K, K]

            self.fc.weight = nn.Parameter(torch.mm(weight, torch.eye(self.num_class, self.out_dim)))  # [K, d]
            self.fc.weight.requires_grad_(False)

    def _make_layer(self, block, out_channels, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, **kwargs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, ret=None):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)

        feat = self.feature(output)
        if self.args.norm:
            feat = F.normalize(feat, p=2, dim=-1)
        out = self.fc(feat)

        if ret == 'of':
            return out, feat
        else:
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
            lam = torch.tensor([lam], dtype=torch.float32, device=x.device)
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

        feat = self.feature(x)
        out = self.fc(feat)
        return out, target, feat


# ======================== Define ResNet ========================

def resnet18(args=None):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], args=args)


def mresnet32(args=None):
    return ResNet_modify(BasicBlock_s, [5, 5, 5], args=args)


def resnet34(args=None):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], args=args)


def resnet50(args=None):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], args=args)


def resnet101(args=None):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], args=args)


def resnet152(args=None):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], args=args)


def iresnet50(args=None):
    return ResNet(IRBlock, [3, 4, 14, 3], args=args)
