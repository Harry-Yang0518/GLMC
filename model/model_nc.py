import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .utils import *


class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


class ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, small_kernel=True, backbone='resnet18', args=None):
        super(ResNet, self).__init__()
        self.num_classes = num_classes

        if args.norm == 'bn':
            resnet_model = models.__dict__[backbone](pretrained=pretrained)
        else:
            resnet_model = models.__dict__[backbone](pretrained=pretrained, norm_layer=GroupNorm32)

        if small_kernel:
            conv1_out_ch = resnet_model.conv1.out_channels
            if args.dataset in ['fmnist']:
                resnet_model.conv1 = nn.Conv2d(1, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False)  # Small dataset filter size used by He et al. (2015)
            else:
                resnet_model.conv1 = nn.Conv2d(3, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False)  # Small dataset filter size used by He et al. (2015)
        resnet_model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        # Isolate the feature extraction layers
        self.layer0 = nn.Sequential(resnet_model.conv1,
                                    resnet_model.bn1,
                                    resnet_model.relu,
                                    resnet_model.maxpool)
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.feat = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten()
                                  )

        # Isolate the classifier layer
        self.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

        if args.etf_cls:
            weight = torch.sqrt(torch.tensor(num_classes / (num_classes - 1))) * (
                    torch.eye(num_classes) - (1 / num_classes) * torch.ones((num_classes, num_classes)))
            weight /= torch.sqrt((1 / num_classes * torch.norm(weight, 'fro') ** 2))

            self.fc.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, resnet_model.fc.in_features)))
            self.fc.weight.requires_grad_(False)

    def forward(self, x, ret='of'):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.feat(x)
        out = self.fc(x)

        if ret == 'of':
            return out, x
        else:
            return out

    def forward_mixup(self, x, target, mixup=None, mixup_alpha=None):

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
        x = self.layer0(x)

        if layer_mix == 1:
            x, target = mixup_process(x, target, lam)
        x = self.layer1(x)

        if layer_mix == 2:
            x, target = mixup_process(x, target, lam)
        x = self.layer2(x)

        if layer_mix == 3:
            x, target = mixup_process(x, target, lam)
        x = self.layer3(x)

        if layer_mix == 4:
            x, target = mixup_process(x, target, lam)
        x = self.layer4(x)

        if layer_mix == 5:
            x, target = mixup_process(x, target, lam)
        feat = self.feat(x)
        out = self.fc(feat)

        return out, target, feat


class MLP(nn.Module):
    def __init__(self, hidden, depth=6, fc_bias=True, num_classes=10):
        # Depth means how many layers before final linear layer

        super(MLP, self).__init__()
        layers = [nn.Linear(3072, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        for i in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, num_classes, bias=fc_bias)
        print(fc_bias)

    def forward(self, x, ret='of'):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        features = F.normalize(x)
        x = self.fc(x)
        if ret == 'of':
            return x, features
        else:
            return x
