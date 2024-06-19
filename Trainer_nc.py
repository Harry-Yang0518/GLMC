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
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

from utils import util
from utils.util import *
from utils.plot import plot_nc
from utils.measure_nc import analysis
from model.KNN_classifier import KNNClassifier
from model.loss import CrossEntropyLabelSmooth, CDTLoss, LDTLoss


def _get_polynomial_decay(lr, end_lr, decay_epochs, from_epoch=0, power=1.0):
    # Note: epochs are zero indexed by pytorch
    end_epoch = float(from_epoch + decay_epochs)

    def lr_lambda(epoch):
        if epoch < from_epoch:
            return 1.0
        epoch = min(epoch, end_epoch)
        new_lr = ((lr - end_lr) * (1. - epoch / end_epoch) ** power + end_lr)
        return new_lr / lr  # LambdaLR expects returning a factor

    return lr_lambda


def get_scheduler(args, optimizer, n_batches):
    """cosine will change learning rate every iteration, others change learning rate every epoch"""

    lr_lambda = _get_polynomial_decay(args.lr, args.end_lr,
                                      decay_epochs=args.decay_epochs,
                                      from_epoch=0, power=args.power)
    SCHEDULERS = {
        'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 10, gamma=args.lr_decay),
        'multi_step': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 350], gamma=0.1),
        'ms': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 350], gamma=0.1),
        'poly': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    }
    return SCHEDULERS[args.scheduler]


class Trainer(object):
    def __init__(self, args, model=None, train_loader=None, val_loader=None, log=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch

        self.train_loader = train_loader
        self.val_loader = val_loader
        if args.imbalance_type == 'step' or args.imbalance_type == 'exp':
            self.cls_num_list = np.array(train_loader.dataset.per_class_num)
        else:
            self.cls_num_list = np.array([500] * args.num_classes)

        self.num_classes = args.num_classes
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr,
                                         weight_decay=args.weight_decay)
        self.lr_scheduler = get_scheduler(args, self.optimizer, n_batches=len(train_loader))

        self.log = log

    def train_one_epoch(self):
        # switch to train mode
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')

        for i, (inputs, targets) in enumerate(self.train_loader):

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            output, h = self.model(inputs, ret='of')

            # ==== update loss and acc
            train_acc.update(torch.sum(output.argmax(dim=-1) == targets).item() / targets.size(0),
                             targets.size(0)
                             )
            loss = self.criterion(output, targets)
            losses.update(loss.item(), targets.size(0))

            # ==== gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ==== LR decay
            if self.args.scheduler == 'cosine':
                self.lr_scheduler.step()

        return losses, train_acc

    def train_base(self):
        best_acc1 = 0

        if self.args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')  # train fc_bc
        elif self.args.loss == 'ls':
            self.criterion = CrossEntropyLabelSmooth(self.args.num_classes, epsilon=self.args.eps)

        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.criterion, log="all", log_freq=20)
        train_nc = Graph_Vars()

        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            losses, train_acc = self.train_one_epoch()
            epoch_time = time.time() - start_time

            # ===== Log training metrics
            self.log.info(
                '====>EPOCH_{epoch}_Iters_{iters}, Epoch Time:{epoch_time:.3f}, Loss:{loss:.4f}, Acc:{acc:.4f}'.format(
                    epoch=epoch + 1, iters=len(self.train_loader), epoch_time=epoch_time, loss=losses.avg,
                    acc=train_acc.avg
                ))
            if epoch % 10 == 0 and self.args.bias:
                bias_values = self.model.fc.bias.data
                self.log.info('--Epoch_{epoch}, Bias: {bias_str}'.format(
                    epoch=epoch + 1, bias_str=', '.join([f'{bias_value:.4f}' for bias_value in bias_values])))
            wandb.log({'train/train_loss': losses.avg,
                       'train/train_acc': train_acc.avg,
                       'lr': self.optimizer.param_groups[0]['lr']},
                      step=epoch + 1)

            # ===== evaluate on validation set
            acc1, acc5, cls_acc, many_acc, few_acc = self.validate(epoch=epoch)
            wandb.log({'val/val_acc1': acc1,
                       'val/val_acc5': acc5,
                       'val/many_acc': many_acc,
                       'val/few_acc': few_acc},
                      step=epoch + 1)

            # ===== measure NC
            if self.args.debug > 0:
                if (epoch + 1) % self.args.debug == 0:
                    nc_dict = analysis(self.model, self.train_loader, self.args)
                    nc_dict['test_acc'] = acc1

                    self.log.info(
                        '>>>>Epoch:{}, Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f}, NC2h:{:.3f}, NC2W:{:.3f}, NC3:{:.3f}, TestAcc:{:.2f}'.format(
                            epoch + 1, nc_dict['loss'], nc_dict['acc'], nc_dict['nc1'], nc_dict['nc2_h'],
                            nc_dict['nc2_w'],
                            nc_dict['nc3'], nc_dict['test_acc']))
                    train_nc.load_dt(nc_dict, epoch=epoch + 1, lr=self.optimizer.param_groups[0]['lr'])
                    wandb.log({'nc/loss': nc_dict['loss'],
                               'nc/acc': nc_dict['acc'],
                               'nc/nc1': nc_dict['nc1'],
                               'nc/nc2h': nc_dict['nc2_h'],
                               'nc/nc2w': nc_dict['nc2_w'],
                               'nc/nc3': nc_dict['nc3'],
                               'nc/w_norm': nc_dict['w_mnorm'],
                               'nc/h_norm': nc_dict['h_mnorm'],
                               'nc/nc3_d': nc_dict['nc3_d']
                               }, step=epoch + 1)
                    if (epoch + 1) % (self.args.debug * 5) == 0:
                        fig = plot_nc(nc_dict)
                        wandb.log({"chart": fig}, step=epoch + 1)

                        # filename = os.path.join(self.args.root_model, self.args.store_name,
                        #                         'analysis{}.pkl'.format(epoch))
                        # with open(filename, 'wb') as f:
                        #     pickle.dump(nc_dict, f)
                        # self.log.info('-- Has saved the NC analysis result/epoch{} to {}'.format(epoch + 1, filename))

            if self.args.scheduler in ['step', 'ms', 'multi_step', 'poly']:
                self.lr_scheduler.step()
            self.model.train()

        self.log.info('Best Testing Prec@1: {:.3f}\n'.format(best_acc1))

        # Store NC statistics
        filename = os.path.join(self.args.root_model, self.args.store_name, 'train_nc.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(train_nc, f)
        self.log.info('-- Has saved Train NC analysis result to {}'.format(filename))

    def validate(self, epoch=None, ret_fine=False):
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
                output = self.model(input, ret='o')

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

            self.log.info(
                '---->EPOCH_{epoch} Val: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1, top1=top1, top5=top5))
            # out_cls_acc = '%s Class Accuracy: %s' % ('val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
            # self.log.info(out_cls_acc)

        cls_acc, many_acc, few_acc = self.calculate_acc(all_targets, all_preds)
        return top1.avg, top5.avg, cls_acc, many_acc, few_acc

    def calculate_acc(self, targets, preds):
        eps = np.finfo(np.float64).eps
        cf = confusion_matrix(targets, preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        many_shot = self.cls_num_list >= np.max(self.cls_num_list)
        few_shot = self.cls_num_list <= np.min(self.cls_num_list)

        many_acc = float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps))
        few_acc = float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))

        return cls_acc, many_acc, few_acc