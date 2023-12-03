# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
from scipy.sparse.linalg import svds


def analysis(model, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N      = [0 for _ in range(args.num_classes)]   # within class sample size
    mean   = [0 for _ in range(args.num_classes)]
    Sw_cls = [0 for _ in range(args.num_classes)]
    loss   = 0
    n_correct = 0

    model.eval()
    criterion_summed = torch.nn.CrossEntropyLoss(reduction='sum')

    for batch_idx, (data, target) in enumerate(loader, start=1):

        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output, h = model(data, ret_feat=True)  # [B, C], [B, 512]

        loss += criterion_summed(output, target).item()
        net_pred = torch.argmax(output, dim=1)
        n_correct += torch.sum(net_pred == target).item()

        for c in range(args.C):
            idxs = torch.where(target == c)[0]

            if len(idxs) > 0:  # If no class-c in this batch
                h_c = h[idxs, :]  # [B, 512]
                mean[c] += torch.sum(h_c, dim=0)  #  CHW
                N[c] += h_c.shape[0]
    M = torch.stack(mean).T               # [512, K]
    M = M / torch.tensor(N, device=M.device).unsqueeze(0)  # [512, K]
    loss = loss / sum(N)
    acc = n_correct / sum(N)

    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output, h = model(data, ret_feat=True)  # [B, C], [B, 512]

        for c in range(args.C):
            idxs = torch.where(target == c)[0]
            if len(idxs) > 0:  # If no class-c in this batch
                h_c = h[idxs, :]  # [B, 512]
                # update within-class cov
                z = h_c - mean[c].unsqueeze(0)  # [B, 512]
                cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))  # [B 512 1] [B 1 512] -> [B, 512, 512]
                Sw_cls[c] += torch.sum(cov, dim=0)  # [512, 512]

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True)  # [512, C] -> [512, 1]

    # between-class covariance
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.C

    # ============== NC1 ==============
    Sw_all = sum(Sw_cls) / sum(N)  # [512, 512]
    for c in range(args.num_classes):
        Sw_cls[c] = Sw_cls[c] / N[c]

    Sw = Sw_all.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.num_classes - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T

    nc1 = np.trace(Sw @ inv_Sb)
    nc1_cls = [np.trace(Sw_cls1.cpu().numpy() @ inv_Sb) for Sw_cls1 in Sw_cls]
    nc1_cls = np.array(nc1_cls)

    # ============== NC2: norm and cos ==============
    W = model.fc_cb.weight.detach().T  # [512, C]
    M_norms = torch.norm(M_, dim=0)  # [C]
    W_norms = torch.norm(W , dim=0)  # [C]

    # angle between W
    W_nomarlized = W / W_norms  # [512, C]
    cos = (W_nomarlized.T @ W_nomarlized).cpu().numpy()  # [C, D] [D, C] -> [C, C]
    cos_avg = (cos.sum(1) - np.diag(cos)) / (cos.shape[1] - 1)

    # angle between H
    M_normalized = M_ / M_norms  # [512, C]
    h_cos = (M_normalized.T @ M_normalized).cpu().numpy()
    h_cos_avg = (h_cos.sum(1) - np.diag(h_cos)) / (h_cos.shape[1] - 1)

    # angle between W and H
    wh_cos = torch.sum(W_nomarlized*M_normalized, dim=0).cpu().numpy()  # [C]

    return {
        "loss": loss,
        "acc": acc,
        "nc1": nc1,
        "nc1_cls": nc1_cls,
        "w_norm": W_norms.cpu().numpy(),
        "h_norm": M_norms.cpu().numpy(),
        "w_cos": cos,
        "w_cos_avg": cos_avg,
        "h_cos":h_cos,
        "h_cos_avg": h_cos_avg,
        "wh_cos": wh_cos
    }

