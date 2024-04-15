import sys
import math
import time
import wandb
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix

from utils import util
from utils.util import *
from utils.plot import plot_nc
from utils.measure_nc import analysis
from model.KNN_classifier import KNNClassifier
from model.loss import CrossEntropyLabelSmooth, CDTLoss, LDTLoss, CombinedMarginLoss


def soften_target(targets, num_classes, epsilon):
    targets = torch.zeros(targets.size(0), num_classes).scatter_(
        1,
        targets.unsqueeze(1).cpu(), 1)

    if torch.cuda.is_available(): targets = targets.cuda()
    targets = (1 - epsilon) * targets + epsilon / num_classes
    return targets


def get_samples_per_class(dataset, num_samples_per_class=10, num_classes=10):
    samples_per_class = {k: [] for k in range(num_classes)}
    for idx, (image, label) in enumerate(dataset):
        if len(samples_per_class[label]) < num_samples_per_class:
            samples_per_class[label].append(image.unsqueeze(0))
        if all(len(samples) == num_samples_per_class for samples in samples_per_class.values()):
            break
    return samples_per_class


class Trainer_bn(object):
    def __init__(self, args, model=None,train_loader=None, majority_loader=None, val_loader=None,weighted_train_loader=None,per_class_num=[],log=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.print_freq = args.print_freq
        self.label_weighting = args.label_weighting
        self.num_classes = args.num_classes

        self.train_loader = train_loader
        self.majority_loader = majority_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader

        # init queue
        samples_per_class = get_samples_per_class(train_loader.dataset,
                                                  num_samples_per_class=np.ceil(args.batch_size/args.num_classes),
                                                  num_classes=args.num_classes)
        self.queue = {k: torch.cat(samples_per_class[k], dim=0).to(self.device) for k in range(args.num_classes)}
        self.queue_ptr = {k: torch.zeros(1, dtype=torch.long) for k in range(args.num_classes)}

        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight
        self.log = log

        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        self.update_weight()
        self.set_loss()

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)

    def set_loss(self):
        if self.args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')  # train fc_bc
        elif self.args.loss == 'ls':
            self.criterion = CrossEntropyLabelSmooth(self.args.num_classes, epsilon=self.args.eps)
        elif self.args.loss == 'ldt':
            delta_list = self.cls_num_list / np.min(self.cls_num_list)
            self.criterion = LDTLoss(delta_list, gamma=0.5, device=self.device)
        elif self.args.loss == 'wce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=self.per_cls_weights)
        elif self.args.loss == 'hce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif self.args.loss == 'bce':
            self.criterion = nn.BCELoss(reduction='mean')
        elif self.args.loss == 'arcf' or self.args.loss == 'arcm':
            self.criterion = CombinedMarginLoss(64, self.args.margins[0], self.args.margins[1], self.args.margins[2])



    # def train_one_epoch(self):
    #     self.model.train()
    #     losses = AverageMeter('Loss', ':.4e')
    #     train_acc = AverageMeter('Train_acc', ':.4e')

    #     train_loader = self.weighted_train_loader if self.args.resample_weighting > 0 else self.train_loader

    #     for i, (inputs, targets) in enumerate(train_loader):
    #         inputs, targets = inputs.to(self.device), targets.to(self.device)
    #         output= self.model(inputs, targets)

    #         # Only consider majority classes
    #         majority_classes = torch.tensor([0, 1, 2, 3, 4], device=self.device)
    #         majority_mask = torch.isin(targets, majority_classes)   

    #         if majority_mask.any():
    #             majority_inputs = inputs[majority_mask]
    #             majority_targets = targets[majority_mask]
    #             majority_output = output[majority_mask]

    #             # Check if the outputs contain any NaN
    #             if torch.isnan(majority_output).any():
    #                 print(f"NaN detected in outputs at iteration {i}")
    #                 continue

    #             loss = self.criterion(majority_output, majority_targets)
    #             if torch.isnan(loss):
    #                 print(f"NaN detected in loss at iteration {i}")
    #                 continue

    #             losses.update(loss.item(), majority_inputs.size(0))
    #             train_acc.update((majority_output.argmax(1) == majority_targets).float().mean().item(), majority_inputs.size(0))

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
    #             self.optimizer.step()

    #     return losses, train_acc

    # def train_one_epoch(self):
    #     # switch to train mode
    #     self.model.train()
    #     losses = AverageMeter('Loss', ':.4e')
    #     train_acc = AverageMeter('Train_acc', ':.4e')

    #     if self.args.resample_weighting > 0:
    #         train_loader = self.weighted_train_loader
    #     else:
    #         train_loader = self.train_loader

    #     for i, (inputs, targets) in enumerate(train_loader):
    #         inputs, targets = inputs.to(self.device), targets.to(self.device)

    #         if self.args.aug == 'cm' or self.args.aug == 'cutmix':  # cutmix augmentation within the mini-batch
    #             cutmix = v2.CutMix(num_classes=self.args.num_classes)
    #             inputs, reweighted_targets = cutmix(inputs, targets)  # reweighted target will be [B, K]

    #         if self.args.mixup >= 0:
    #             output, reweighted_targets, h = self.model.forward_mixup(inputs, targets, mixup=self.args.mixup,
    #                                                                     mixup_alpha=self.args.mixup_alpha)
    #         else:
    #             freq = torch.bincount(targets, minlength=self.args.num_classes)
    #             cls_idx = torch.where(freq == 0)[0]

    #             if cls_idx.nelement() != 0:
    #                 bn_inputs = torch.cat([self.queue[k] for k in cls_idx.cpu().numpy()], dim=0).to(self.device)
    #                 bn_targets = torch.cat([torch.tensor(k).repeat(len(self.queue[0])) for k in cls_idx.cpu().numpy()], dim=0).to(self.device)

    #                 all_inputs = torch.cat([inputs, bn_inputs])
    #                 all_targets = torch.cat([targets, bn_targets])
    #             else:
    #                 all_inputs = inputs
    #                 all_targets = targets

    #             output_all, h_all = self.model(all_inputs, all_targets, ret='of')
    #             output, h = output_all[0:len(inputs)], h_all[0:len(inputs)]

    #         # update the img_bank with current batch
    #         for k in self.queue:
    #             cls_idx = torch.where(targets == k)[0]
    #             if len(cls_idx) == 0:
    #                 continue
    #             else:
    #                 ptr = self.queue_ptr[k]
    #                 num_cls = min(len(self.queue[k]) - ptr, len(cls_idx))  # Ensure we do not exceed buffer or available indices

    #                 # Dynamically adjust the shape of the source tensor to match the destination
    #                 self.queue[k][ptr:ptr + num_cls] = inputs[cls_idx][:num_cls]

    #                 # Safely update the pointer with wrap-around
    #                 self.queue_ptr[k] = (ptr + num_cls) % len(self.queue[k])

    #         # Only consider majority classes
    #         majority_classes = torch.tensor([0, 1, 2, 3, 4], device=self.device)
    #         majority_mask = torch.isin(targets, majority_classes)

    #         if majority_mask.any():
    #             majority_inputs = inputs[majority_mask]
    #             majority_targets = targets[majority_mask]
    #             majority_output = output[majority_mask]

    #             # Check if the outputs contain any NaN
    #             if torch.isnan(majority_output).any():
    #                 print(f"NaN detected in outputs at iteration {i}")
    #                 continue

    #             loss = self.criterion(majority_output, majority_targets)
    #             if torch.isnan(loss):
    #                 print(f"NaN detected in loss at iteration {i}")
    #                 continue

    #             losses.update(loss.item(), majority_inputs.size(0))
    #             train_acc.update((majority_output.argmax(1) == majority_targets).float().mean().item(), majority_inputs.size(0))

    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
    #             self.optimizer.step()

    #     return losses, train_acc



    def train_one_epoch(self):

        # switch to train mode
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')

        if self.args.resample_weighting > 0:
            train_loader = self.weighted_train_loader
        else:
            train_loader = self.train_loader
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.args.aug == 'cm' or self.args.aug == 'cutmix':     # cutmix augmentation within the mini-batch
                cutmix = v2.CutMix(num_classes=self.args.num_classes)
                inputs, reweighted_targets = cutmix(inputs, targets)   # reweighted target will be [B, K]

            if self.args.mixup >= 0:
                output, reweighted_targets, h = self.model.forward_mixup(inputs, targets, mixup=self.args.mixup,
                                                                         mixup_alpha=self.args.mixup_alpha)
            else:
                freq = torch.bincount(targets, minlength=self.args.num_classes)
                cls_idx = torch.where(freq==0)[0]
                bn_inputs = torch.cat([self.queue[k] for k in cls_idx.cpu().numpy()], dim=0).to(self.device)
                bn_targets = torch.cat([torch.tensor(k).repeat(len(self.queue[0])) for k in cls_idx.cpu().numpy()], dim=0).to(self.device)

                all_inputs = torch.cat([inputs, bn_inputs])
                all_targets = torch.cat([targets, bn_targets])

                output_all, h_all = self.model(all_inputs, all_targets, ret='of')
                output, h = output_all[0:len(inputs)], h_all[0:len(inputs)]

            # update the img_bank with current batch
            for k in self.queue:
                cls_idx = torch.where(targets == k)[0]
                if len(cls_idx) == 0:
                    continue
                else:
                    ptr = self.queue_ptr[k]
                    num_cls = min(len(self.queue[k]) - ptr, len(cls_idx))  # Ensure we do not exceed buffer or available indices

                    # Dynamically adjust the shape of the source tensor to match the destination
                    self.queue[k][ptr:ptr + num_cls] = inputs[cls_idx][:num_cls]

                    # Safely update the pointer with wrap-around
                    self.queue_ptr[k] = (ptr + num_cls) % len(self.queue[k])




            # ==== update loss and acc
            train_acc.update(torch.sum(output.argmax(dim=-1) == targets).item() / targets.size(0),
                             targets.size(0)
                             )
            loss = self.criterion(output, reweighted_targets if self.args.mixup >= 0 or self.args.aug == 'cm' or self.args.aug == 'cutmix' else targets)
            losses.update(loss.item(), targets.size(0))

            # ==== gradient update
            if self.args.loss != 'hce':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            elif self.args.loss == 'hce':

                # gradient of L wrt. b
                beta = self.per_cls_weights[targets]  # [B]
                P = nn.Softmax(dim=-1)(output.detach())  # [B, K]
                Y = torch.eye(self.args.num_classes, device=targets.device)[targets]  # [B, K]
                b_grad = beta.unsqueeze(1) * (P - Y)  # [B, K]
                b_grad = torch.sum(b_grad, dim=0) / len(b_grad)

                # gradient of L wrt. W
                weighted_P_Y = (P.detach() - Y) * beta.unsqueeze(1)  # [B, K]
                W_grad = torch.einsum('db, bk->dk', h.detach().T, weighted_P_Y) / len(output)  # [D, K]
                W_grad = W_grad.T  # [K, D]

                self.optimizer.zero_grad()
                loss.backward()
                self.model.fc_cb.bias.grad = b_grad
                self.model.fc_cb.weight.grad = W_grad
                self.optimizer.step()
        return losses, train_acc

    def train_base(self):
        best_acc1 = 0

        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        train_nc = Graph_Vars()

        for epoch in range(self.args.start_epoch, self.args.epochs):

            # ============ training ============
            start_time = time.time()
            losses, train_acc = self.train_one_epoch()
            epoch_time = time.time() - start_time
            self.log.info(
                '====>EPOCH{epoch}Train{iters}, Epoch Time:{epoch_time:.3f}, Loss:{loss:.4f}, Acc:{acc:.4f}'.format(
                    epoch=epoch + 1, iters=len(self.train_loader), epoch_time=epoch_time, loss=losses.avg, acc=train_acc.avg
                ))
            wandb.log({'train/train_loss': losses.avg,
                       'train/train_acc': train_acc.avg,
                       'lr': self.optimizer.param_groups[0]['lr']},
                      step=epoch + 1)

            # ============ evaluation ============
            acc1 = self.validate(epoch=epoch)
            if self.args.imbalance_type == 'step' and self.args.imbalance_rate < 1.0:
                knn_acc1 = self.validate_knn(epoch=epoch)

            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            self.model.train()

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, epoch + 1)

            # # ============ Measure NC ============
            if self.args.debug > 0:
                if (epoch + 1) % self.args.debug == 0:
                    nc_dict = analysis(self.model, self.train_loader, self.args)
                    self.log.info('Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f}, NC2h:{:.3f}, NC2W:{:.3f}, NC3:{:.3f}'.format(
                        nc_dict['loss'], nc_dict['acc'], nc_dict['nc1'], nc_dict['nc2_h'], nc_dict['nc2_w'],
                        nc_dict['nc3'],
                    ))
                    train_nc.load_dt(nc_dict, epoch=epoch + 1, lr=self.optimizer.param_groups[0]['lr'])
                    wandb.log({'nc/loss': nc_dict['loss'],
                               'nc/acc': nc_dict['acc'],
                               'nc/nc1': nc_dict['nc1'],
                               'nc/nc2h': nc_dict['nc2_h'],
                               'nc/nc2w': nc_dict['nc2_w'],
                               'nc/nc3': nc_dict['nc3'],
                               'nc/nc3d': nc_dict['nc3_d'],
                               },
                              step=epoch + 1)
                    if self.args.imbalance_type == 'step' and self.args.imbalance_rate < 1.0:
                        wandb.log({'nc1/w_mnorm': nc_dict['w_mnorm'],
                                   'nc1/w_mnorm1': nc_dict['w_mnorm1'],
                                   'nc1/w_mnorm2': nc_dict['w_mnorm2'],
                                   'nc1/h_mnorm': nc_dict['h_mnorm'],
                                   'nc1/h_mnorm1': nc_dict['h_mnorm1'],
                                   'nc1/h_mnorm2': nc_dict['h_mnorm2'],
                                   'nc1/w_cos1': nc_dict['w_cos1'],
                                   'nc1/w_cos2': nc_dict['w_cos2'],
                                   'nc1/w_cos3': nc_dict['w_cos3'],
                                   'nc1/h_cos1': nc_dict['h_cos1'],
                                   'nc1/h_cos2': nc_dict['h_cos2'],
                                   'nc1/h_cos3': nc_dict['h_cos3']},
                                  step=epoch + 1)
                    if (epoch + 1) % (self.args.debug * 5) == 0:
                        fig = plot_nc(nc_dict)
                        wandb.log({"chart": fig}, step=epoch + 1)

                        filename = os.path.join(self.args.root_model, self.args.store_name, 'analysis{}.pkl'.format(epoch))
                        with open(filename, 'wb') as f:
                            pickle.dump(nc_dict, f)
                        self.log.info('-- Has saved the NC analysis result/epoch{} to {}'.format(epoch + 1, filename))

        self.log.info('Best Testing Prec@1: {:.3f}\n'.format(best_acc1))
        # Store NC statistics
        filename = os.path.join(self.args.root_model, self.args.store_name, 'train_nc.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(train_nc, f)
        self.log.info('-- Has saved Train NC analysis result to {}'.format(filename))

    def validate(self, epoch=None):
        # switch to evaluate mode
        self.model.eval()
        all_logits, all_targets = [], []

        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                input, target = input.to(self.device), target.to(self.device)

                output = self.model(input, target, ret='o')
                all_logits.append(output)
                all_targets.append(target)
            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)
            all_preds = all_logits.argmax(1)
            # measure accuracy
            acc1, acc5 = accuracy(all_logits, all_targets, topk=(1, 5))
            cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(all_targets.cpu().numpy(), all_preds.cpu().numpy())
            self.log.info(
                '---->EPOCH{} Val: Prec@1 {:.3f} Prec@5 {:.3f}'.format(epoch, acc1, acc5))
            self.log.info("many acc {:.2f}, med acc {:.2f}, few acc {:.2f}".format(many_acc, medium_acc, few_acc))

            wandb.log({'val/val_acc1': acc1,
                       'val/val_acc5': acc5,
                       'val/val_many': many_acc,
                       'val/val_medium': medium_acc,
                       'val/val_few': few_acc},
                      step=epoch + 1)

        return acc1

    def validate_knn(self, epoch=None):
        # switch to evaluate mode
        self.model.eval()
        all_logits, all_targets = [], []
        cfeats = self.get_knncentroids()
        self.knn_classifier = KNNClassifier(feat_dim=self.model.out_dim, num_classes=self.args.num_classes, feat_type='cl2n', dist_type='l2')
        self.knn_classifier.update(cfeats)

        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                input, target = input.to(self.device), target.to(self.device)
                _, feats = self.model(input, target, ret='of')  # pred from fc_bc
                logit = self.knn_classifier(feats)
                all_logits.append(logit)
                all_targets.append(target)
            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)
            all_preds = all_logits.argmax(1)

            # measure accuracy
            acc1, acc5 = accuracy(all_logits, all_targets, topk=(1, 5))
            cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(all_targets.cpu().numpy(), all_preds.cpu().numpy())
            self.log.info(
                '---->EPOCH{} Val: Prec@1 {:.3f} Prec@5 {:.3f}'.format(epoch, acc1, acc5))
            self.log.info("many acc {:.2f}, med acc {:.2f}, few acc {:.2f}".format(many_acc, medium_acc, few_acc))

            wandb.log({'knn_val/val_acc1': acc1,
                       'knn_val/val_acc5': acc5,
                       'knn_val/val_many': many_acc,
                       'knn_val/val_medium': medium_acc,
                       'knn_val/val_few': few_acc},
                      step=epoch + 1)

        return acc1

    def calculate_acc(self, targets, preds):
        eps = np.finfo(np.float64).eps
        cf = confusion_matrix(targets, preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        many_shot = self.cls_num_list > 100
        medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
        few_shot = self.cls_num_list <= 20

        many_acc = float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps))
        medium_acc = float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps))
        few_acc = float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))

        return cls_acc, many_acc, medium_acc, few_acc

    def SimSiamLoss(self,p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def paco_adjust_learning_rate(self,optimizer, epoch, args):
        warmup_epochs = 10
        lr = self.args.lr
        if epoch <= warmup_epochs:
            lr = self.args.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.args.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_knncentroids(self):
        # print('===> Calculating KNN centroids.')

        torch.cuda.empty_cache()
        self.model.eval()
        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Calculate Features of each training data
                _, feats = self.model(inputs, labels, ret='of')
                feats_all.append(feats.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all)
        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids = []
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_ == i], axis=0))
            return np.stack(centroids)

        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)

        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,
                'cl2ncs': cl2n_centers}

    def train(self):
        best_acc1 = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            alpha = 1 - (epoch / self.args.epochs) ** 2  # balance loss terms
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            # switch to train mode
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            for i, (inputs, targets) in enumerate(self.train_loader):

                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                try:
                    input_invs, target_invs = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs = next(weighted_train_loader)

                input_invs_1 = input_invs[0][:input_org_1.size()[0]]
                input_invs_2 = input_invs[1][:input_org_2.size()[0]]

                one_hot_org = torch.zeros(target_org.size(0), self.num_classes).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1),
                                                                                           1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = self.per_cls_weights.cpu() * one_hot_invs

                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # Data augmentation
                lam = np.random.beta(self.beta, self.beta)

                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1,
                                                                                         org2=input_org_2,
                                                                                         invs1=input_invs_1,
                                                                                         invs2=input_invs_2,
                                                                                         label_org=one_hot_org,
                                                                                         label_invs=one_hot_invs,
                                                                                         label_org_w=one_hot_org_w,
                                                                                         label_invs_w=one_hot_invs_w)

                output_1, output_cb_1, z1, p1, _ = self.model(mix_x, ret='all')
                output_2, output_cb_2, z2, p2, _ = self.model(cut_x, ret='all')
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w

                loss = alpha * balance_loss + (1 - alpha) * rebalance_loss + self.contrast_weight * contrastive_loss

                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.args.epochs, i, len(self.train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))  # TODO
                    print(output)

            # measure NC
            if self.args.debug > 0:
                if (epoch + 1) % self.args.debug == 0:
                    nc_dict = analysis(self.model, self.train_loader, self.args)
                    self.log.info('Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f},\nWnorm:{}\nHnorm:{}\nWcos:{}\nWHcos:{}'.format(
                        nc_dict['loss'], nc_dict['acc'], nc_dict['nc1'],
                        np.array2string(nc_dict['w_norm'], separator=',',
                                        formatter={'float_kind': lambda x: "%.3f" % x}),
                        np.array2string(nc_dict['h_norm'], separator=',',
                                        formatter={'float_kind': lambda x: "%.3f" % x}),
                        np.array2string(nc_dict['w_cos_avg'], separator=',',
                                        formatter={'float_kind': lambda x: "%.3f" % x}),
                        np.array2string(nc_dict['wh_cos'], separator=',',
                                        formatter={'float_kind': lambda x: "%.3f" % x})
                    ))
                if (epoch + 1) % (5 * self.args.debug) == 0:
                    filename = os.path.join(self.args.root_model, self.args.store_name, 'analysis{}.pkl'.format(epoch))
                    import pickle
                    with open(filename, 'wb') as f:
                        pickle.dump(nc_dict, f)
                    self.log.info('-- Has saved the NC analysis result to {}'.format(filename))

            # evaluate on validation set
            acc1 = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, epoch + 1)



