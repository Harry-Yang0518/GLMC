# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
from scipy.sparse.linalg import svds


def compute_ETF(W, device):  # W [K, 512]
    K = W.shape[0]
    # W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)            # [K, 512] [512, K] -> [K, K]
    WWT /= torch.norm(WWT, p='fro')   # [K, K]

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, H, device):  # W:[K, 512] H:[512, K]
    """ H is already normalized"""
    K = W.shape[0]

    # W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.to(device))   # [K, 512] [512, K]
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


def analysis(model, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = [0 for _ in range(args.num_classes)]  # within class sample size
    mean = [0 for _ in range(args.num_classes)]
    Sw_cls = [0 for _ in range(args.num_classes)]
    loss = 0
    n_correct = 0

    model.eval()
    criterion_summed = torch.nn.CrossEntropyLoss(reduction='sum')

    for computation in ['Mean', 'Cov']:
        for batch_idx, (_, target, data) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output, h = model(data, ret='of')  # [B, C], [B, 256]

            for c in range(args.num_classes):
                idxs = (target == c).nonzero(as_tuple=True)[0] # idxs of class-c in this batch
                if len(idxs) == 0:  # If no class-c in this batch
                    continue

                h_c = h[idxs, :]  # [num_of_class-c_in_batch, 512]

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0)  #  CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov
                    z = h_c - mean[c].unsqueeze(0)  # [B, 512]
                    cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))   # [B 512 1] [B 1 512] -> [B, 512, 512]
                    Sw_cls[c] += torch.sum(cov, dim=0)  # [512, 512]

            # during calculation of class cov, calculate loss
            if computation == 'Cov':
                loss += criterion_summed(output, target).item()

                # 1) network's accuracy
                net_pred = torch.argmax(output, dim=1)
                n_correct += sum(net_pred == target).item()

        if computation == 'Mean':
            for c in range(args.num_classes):
                mean[c] /= N[c]
                M = torch.stack(mean).T
        elif computation == 'Cov':
            loss /= sum(N)
            acc = n_correct/sum(N)
            Sw = sum(Sw_cls) / sum(N)
            for c in range(args.num_classes):
                Sw_cls[c] = Sw_cls[c] / N[c]

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True)  # [512, C]

    # between-class covariance
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.num_classes

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.num_classes - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    nc1 = np.trace(Sw @ inv_Sb) # [1]
    nc1_cls = [np.trace(Sw_cls1.cpu().numpy() @ inv_Sb) for Sw_cls1 in Sw_cls]
    nc1_cls = np.array(nc1_cls) # [C]

    # ========== NC2.1 and NC2.2
    W = model.fc_cb.weight.detach().T  # [512, C]
    M_norms = torch.norm(M_, dim=0)  # [C]
    W_norms = torch.norm(W , dim=0)  # [C]

    # == NC2.1
    norm_M_CoV = (torch.std(M_norms) / torch.mean(M_norms)).item()
    norm_W_CoV = (torch.std(W_norms) / torch.mean(W_norms)).item()

    # angle between W
    W_nomarlized = W / W_norms  # [512, C]
    w_cos = (W_nomarlized.T @ W_nomarlized).cpu().numpy()  # [C, D] [D, C] -> [C, C]
    w_cos_avg = (w_cos.sum(1) - np.diag(w_cos)) / (w_cos.shape[1] - 1)

    # angle between H
    M_normalized = M_ / M_norms  # [512, C]
    h_cos = (M_normalized.T @ M_normalized).cpu().numpy()
    h_cos_avg = (h_cos.sum(1) - np.diag(h_cos)) / (h_cos.shape[1] - 1)

    # angle between W and H
    wh_cos = torch.sum(W_nomarlized * M_normalized, dim=0).cpu().numpy()  # [C]

    # == NC2.2
    def coherence(V):
        G = V.T @ V  # [C, D] [D, C]
        G += torch.ones((args.num_classes, args.num_classes), device=device) / (args.num_classes - 1)
        G -= torch.diag(torch.diag(G))  # [C, C]
        return torch.norm(G, 1).item() / (args.num_classes * (args.num_classes - 1))

    cos_M = coherence(M_ / M_norms)  # [D, C]
    cos_W = coherence(W / W_norms)

    # =========== NC2
    nc2_h = compute_ETF(M_.T, device)
    nc2_w = compute_ETF(W.T, device)

    # =========== NC3  ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W / torch.norm(W, 'fro')
    W_M_dist = (torch.norm(normalized_W - normalized_M) ** 2).item()

    # =========== NC3 (all losses are equal paper)
    nc3 = compute_W_H_relation(W.T, M_, device)

    return {
        "loss": loss,
        "acc": acc,
        "nc1": nc1,
        "nc1_cls": nc1_cls,
        "w_norm": W_norms.cpu().numpy(),
        "h_norm": M_norms.cpu().numpy(),
        "w_cos": w_cos,
        "w_cos_avg": w_cos_avg,
        "h_cos": h_cos,
        "h_cos_avg": h_cos_avg,
        "wh_cos": wh_cos, 
        "nc21_h": norm_M_CoV,
        "nc21_w": norm_W_CoV,
        "nc22_h": cos_M,
        "nc22_w": cos_W,
        "nc2_h": nc2_h,
        "nc2_w": nc2_w,
        "nc3": nc3,
        "nc3_1": W_M_dist,
    }


def analysis1(model, loader, args):
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
            output, h = model(data, ret='of')  # [B, C], [B, 512]

        loss += criterion_summed(output, target).item()
        net_pred = torch.argmax(output, dim=1)
        n_correct += torch.sum(net_pred == target).item()

        for c in range(args.num_classes):
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
            output, h = model(data, ret='of')  # [B, C], [B, 512]

        for c in range(args.num_classes):
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
    Sb = torch.matmul(M_, M_.T) / args.num_classes

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
