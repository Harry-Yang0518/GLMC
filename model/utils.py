import torch
import torch.nn as nn
import numpy as np


class BalancedBatchNorm2d(nn.Module):
    # num_features: the number of output channels for a convolutional layer.
    def __init__(self, num_features, ):
        super().__init__()
        shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # moving_mean and moving_var are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X, label):
        # If X is not on the main memory, copy moving_mean and moving_var to the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, label, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-6, momentum=0.1
        )
        return Y


def batch_norm(X, label, gamma, beta, moving_mean, moving_var, eps, momentum):
    max_label = label.max().item()  # Get the maximum label index used
    num_classes = max_label + 1    # Correct num_classes calculation

    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4), "Input tensor must be either 2D or 4D"
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            batch_size, C, H, W = X.shape
            sum_ = torch.zeros((num_classes, C, H, W), dtype=X.dtype, device=X.device)
            cnt_ = torch.zeros(num_classes, dtype=torch.float32, device=X.device)

            for lbl in range(num_classes):
                mask = label == lbl
                if mask.any():
                    sum_[lbl] = X[mask].sum(dim=0)
                    cnt_[lbl] = mask.sum()

            mean = (sum_ / cnt_[:, None, None, None]).mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = (1 - momentum) * moving_mean + momentum * mean
        moving_var = (1 - momentum) * moving_var + momentum * var

    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data





def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def to_one_hot(inp, num_classes):
    if inp is not None:
        y_onehot = torch.FloatTensor(inp.size(0), num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
        return torch.autograd.Variable(y_onehot.cuda(), requires_grad=False)
    else:
        return None


def mixup_process(out, target, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target = target * lam + target[indices] * (1 - lam)
    return out, target
