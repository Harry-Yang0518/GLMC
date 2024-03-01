import sys
import math
import time
import wandb
import torch
import pickle
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

from utils import util
from utils.util import *
# from utils.plot import plot_nc
from utils.measure_nc import analysis
from model.KNN_classifier import KNNClassifier
from model.loss import CrossEntropyLabelSmooth, CDTLoss, LDTLoss

import pdb

class Trainer(object):
    def __init__(self, args, model=None,train_loader=None, val_loader=None,weighted_train_loader=None,per_class_num=[],log=None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.label_weighting = args.label_weighting
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader
        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr, weight_decay=args.weight_decay)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.log = log
        self.beta = args.beta
        self.update_weight()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # for writing summary
        path = os.path.join(args.root_model, args.store_name, 'log')
        self.writer = SummaryWriter(path)

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)

    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            alpha = 1 - (epoch / self.epochs) ** 2     # balance loss terms
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
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1), 1)
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

                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1, org2=input_org_2,
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
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))  # TODO
                    print(output)
                    
            # measure NC
            if self.args.debug>0:
                if (epoch+1) % self.args.debug == 0:
                    nc_dict = analysis(self.model, self.train_loader, self.args)
                    self.log.info('Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f},\nWnorm:{}\nHnorm:{}\nWcos:{}\nWHcos:{}'.format(
                        nc_dict['loss'], nc_dict['acc'], nc_dict['nc1'],
                        np.array2string(nc_dict['w_norm'], separator=',', formatter={'float_kind': lambda x: "%.3f" % x}),
                        np.array2string(nc_dict['h_norm'], separator=',', formatter={'float_kind': lambda x: "%.3f" % x}),
                        np.array2string(nc_dict['w_cos_avg'], separator=',', formatter={'float_kind': lambda x: "%.3f" % x}),
                        np.array2string(nc_dict['wh_cos'], separator=',', formatter={'float_kind': lambda x: "%.3f" % x})
                    ))
                if (epoch+1) % (5*self.args.debug) == 0:
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
            best_acc1 = max(acc1,  best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)

    def train_one_epoch(self):

        # switch to train mode
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')
        # pdb.set_trace()
        if self.args.resample_weighting > 0:
            train_loader = self.weighted_train_loader
        else:
            train_loader = self.train_loader

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device) # [B, 3, 32, 32]
            targets = targets.to(self.device) # [B]
            if self.args.aug == 'mixup':
                if self.args.mixup >= 0:
                    output_cb, reweighted_targets, h = self.model.forward_mixup(inputs, targets, mixup=self.args.mixup,
                                                                                mixup_alpha=self.args.mixup_alpha)
                # output_cb: [B, C], reweighted_targets: [B, C], h: [B, D]
            elif self.args.aug == 'cutmix':
                cutmix = v2.CutMix(num_classes=self.args.num_classes, alpha=self.args.mixup_alpha)
                inputs, reweighted_targets = cutmix(inputs, targets)
                # output_cb, reweighted_targets, h = self.model.forward_cutmix(inputs, targets, cutmix_alpha=self.args.mixup_alpha)
                output, output_cb, z, p, h = self.model(inputs, ret='all')
            else:
                output, output_cb, z, p, h = self.model(inputs, ret='all')

            # ==== update loss and acc
            train_acc.update(torch.sum(output_cb.argmax(dim=-1) == targets).item() / targets.size(0),
                             targets.size(0)
                             ) # train_acc.val to get the current loss
            loss = self.criterion(output_cb, reweighted_targets if (self.args.aug == "mixup" or self.args.aug == "cutmix") else targets)
            losses.update(loss.item(), targets.size(0))

            # ==== gradient update
            if self.args.loss != 'hce':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            elif self.args.loss == 'hce':

                # gradient of L wrt. b
                beta = self.per_cls_weights[targets]  # [B]
                P = nn.Softmax(dim=-1)(output_cb.detach())  # [B, K]
                Y = torch.eye(self.args.num_classes, device=targets.device)[targets]  # [B, K]
                b_grad = beta.unsqueeze(1) * (P - Y)  # [B, K]
                b_grad = torch.sum(b_grad, dim=0) / len(b_grad)

                # gradient of L wrt. W
                weighted_P_Y = (P.detach() - Y) * beta.unsqueeze(1)  # [B, K]
                W_grad = torch.einsum('db, bk->dk', h.detach().T, weighted_P_Y) / len(output_cb)  # [D, K]
                W_grad = W_grad.T  # [K, D]

                self.optimizer.zero_grad()
                loss.backward()
                self.model.fc_cb.bias.grad = b_grad
                self.model.fc_cb.weight.grad = W_grad
                self.optimizer.step()
        return losses, train_acc

    def train_base(self):
        best_acc1 = 0

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

        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        train_nc = Graph_Vars()

        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            losses, train_acc = self.train_one_epoch()
            epoch_time = time.time() - start_time

            self.log.info('====>EPOCH{epoch}Train{iters}, Epoch Time:{epoch_time:.3f}, Loss:{loss:.4f}, Acc:{acc:.4f}'.format(
                epoch=epoch+1, iters=len(self.train_loader), epoch_time=epoch_time, loss=losses.avg, acc=train_acc.avg
            ))
            wandb.log({'train/train_loss': losses.avg,
                       'train/train_acc': train_acc.avg,
                       'lr': self.optimizer.param_groups[0]['lr']},
                       step=epoch+1)

            # measure NC
            if self.args.debug > 0:
                # if (epoch + 1) % self.args.debug == 0:
                if (epoch + 1) % 1 == 0:
                    # pdb.set_trace()
                    nc_dict = analysis(self.model, self.train_loader, self.args)
                    self.log.info('Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f}, NC2h:{:.3f}, NC2W:{:.3f}, NC3:{:.3f}'.format(
                        nc_dict['loss'], nc_dict['acc'], nc_dict['nc1'], nc_dict['nc2_h'], nc_dict['nc2_w'], nc_dict['nc3'],
                        # np.array2string(nc_dict['w_norm'], separator=',', formatter={'float_kind': lambda x: "%.3f" % x}),
                        # np.array2string(nc_dict['h_norm'], separator=',', formatter={'float_kind': lambda x: "%.3f" % x}),
                        # np.array2string(nc_dict['wh_cos'], separator=',', formatter={'float_kind': lambda x: "%.3f" % x})
                    ))
                    train_nc.load_dt(nc_dict, epoch=epoch+1, lr=self.optimizer.param_groups[0]['lr'])
                    wandb.log({'nc/loss': nc_dict['loss'],
                               'nc/acc':  nc_dict['acc'],
                               'nc/nc1':  nc_dict['nc1'],
                               'nc/nc2h': nc_dict['nc2_h'],
                               'nc/nc2w': nc_dict['nc2_w'],
                               'nc/nc3':  nc_dict['nc3']},
                              step=epoch+1)
                    if (epoch+1) % (self.args.debug*5) ==0:
                        # fig = plot_nc(nc_dict)
                        # wandb.log({"chart": fig}, step=epoch+1)

                        filename = os.path.join(self.args.root_model, self.args.store_name, 'analysis{}.pkl'.format(epoch))
                        with open(filename, 'wb') as f:
                            pickle.dump(nc_dict, f)
                        self.log.info('-- Has saved the NC analysis result/epoch{} to {}'.format(epoch+1, filename))

            # evaluate on validation set
            if self.args.knn:
                acc1 = self.validate_knn(epoch=epoch)
            else:
                acc1 = self.validate(epoch=epoch)
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
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)

        self.log.info('Best Testing Prec@1: {:.3f}\n'.format(best_acc1))

        # Store NC statistics
        filename = os.path.join(self.args.root_model, self.args.store_name, 'train_nc.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(train_nc, f)
        self.log.info('-- Has saved Train NC analysis result to {}'.format(filename))

    def validate(self,epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.to(self.device)
                target = target.to(self.device)

                # compute output
                output = self.model(input, ret='o')      # pred from fc_bc

                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}], '
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)

            cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(all_targets, all_preds)
            self.log.info('---->EPOCH{epoch} Val: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1, top1=top1, top5=top5))
            self.log.info("many acc {:.2f}, med acc {:.2f}, few acc {:.2f}".format(many_acc, medium_acc, few_acc))
            # out_cls_acc = '%s Class Accuracy: %s' % ('val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
            # self.log.info(out_cls_acc)

            wandb.log({'val/val_acc1': top1.avg,
                       'val/val_acc5': top5.avg,
                       'val/val_many': many_acc,
                       'val/val_medium': medium_acc,
                       'val/val_few': few_acc},
                      step=epoch+1)

        return top1.avg


    def validate_knn(self,epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []
        cfeats = self.get_knncentroids()
        self.knn_classifier = KNNClassifier(feat_dim=self.model.out_dim, num_classes=self.args.num_classes, feat_type='cl2n', dist_type='l2')
        self.knn_classifier.update(cfeats)

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input, target = input.to(self.device), target.to(self.device)
                _, feats = self.model(input, ret='of')      # pred from fc_bc
                logit = self.knn_classifier(feats)

                # measure accuracy
                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(logit, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}], '
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)

            cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(all_targets, all_preds)
            self.log.info('---->EPOCH{epoch} Val: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1, top1=top1, top5=top5))
            self.log.info("many acc {:.2f}, med acc {:.2f}, few acc {:.2f}".format(many_acc, medium_acc, few_acc))

            # out_cls_acc = '%s Class Accuracy: %s' % ('val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
            # self.log.info(out_cls_acc)

            wandb.log({'val/val_acc1': top1.avg,
                       'val/val_acc5': top5.avg,
                       'val/val_many': many_acc,
                       'val/val_medium': medium_acc,
                       'val/val_few': few_acc},
                      step=epoch + 1)

        return top1.avg

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
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
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
                _, feats = self.model(inputs, ret='of')
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



