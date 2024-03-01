import torch
import numpy as np
import pdb

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


def cutmix_process(out, target, lam):
    indices = np.random.permutation(out.size(0))
    bbx1, bby1, bbx2, bby2 = rand_bbox(out.size(), lam)
    out[:, :, bbx1.item():bbx2.item(), bby1.item():bby2.item()] = out[indices, :, bbx1.item():bbx2.item(), bby1.item():bby2.item()]
    target = target * lam + target[indices] * (1 - lam)
    return out, target


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam.cpu().numpy())
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2