import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.05, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu().long(), 1)

        if torch.cuda.is_available(): targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class CDTLoss(nn.Module):

    def __init__(self, Delta_list, gamma=0.5, weight=None, reduction=None, device = None):
        super(CDTLoss, self).__init__()
        self.gamma = gamma
        self.Delta_list = torch.pow(torch.FloatTensor(Delta_list), self.gamma)
        self.Delta_list = self.Delta_list.shape[0] * self.Delta_list / torch.sum(self.Delta_list)
        self.Delta_list = self.Delta_list.to(device)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        if self.reduction == "sum":
            output = x * self.Delta_list
            return F.cross_entropy(output, target, weight=self.weight, reduction='sum')
        else:
            output = x * self.Delta_list
            return F.cross_entropy(output, target, weight=self.weight)


class LDTLoss(nn.Module):

    def __init__(self, Delta_list, gamma = 0.5, weight=None, reduction = None, device = None):
        super(LDTLoss, self).__init__()
        self.gamma = gamma
        self.Delta_list = torch.pow(torch.FloatTensor(Delta_list), self.gamma)
        self.Delta_list = self.Delta_list.shape[0] * self.Delta_list / torch.sum(self.Delta_list)
        self.Delta_list = self.Delta_list.to(device)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        if self.reduction == "sum":
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight, reduction = 'sum')
        else:
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()