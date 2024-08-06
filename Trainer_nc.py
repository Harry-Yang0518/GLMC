from imbalance_data.cifar100_coarse2fine import fine_id_coarse_id
import wandb
import torch.nn as nn
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix

from utils.util import *
from utils.plot import plot_nc
from utils.measure_nc import analysis
from model.loss import CrossEntropyLabelSmooth


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


class Trainer(object):
    def __init__(self, args, model=None, train_loader=None, val_loader=None, train_loader_base=None, log=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch

        self.model = model
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_base = train_loader_base

        if args.imbalance_type == 'step' or args.imbalance_type == 'exp':
            self.cls_num_list = np.array(train_loader.dataset.per_class_num)
        else:
            self.cls_num_list = np.array([500] * args.num_classes)

        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr, weight_decay=args.weight_decay)
        self.set_scheduler()
        self.log = log

    def set_scheduler(self,):
        if self.args.scheduler == 'cos':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        elif self.args.scheduler in ['ms', 'multi_step']:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[70, 140], gamma=0.1)

    def train_one_epoch(self):
        # switch to train mode
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')

        for i, (inputs, targets) in enumerate(self.train_loader):

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if self.args.aug == 'cm' or self.args.aug == 'cutmix':  # cutmix augmentation within the mini-batch
                cutmix = v2.CutMix(num_classes=self.args.num_classes)
                inputs, reweighted_targets = cutmix(inputs, targets)  # reweighted target will be [B, K]
                
            if self.args.mixup >= 0:
                output, reweighted_targets, h = self.model.forward_mixup(inputs, targets, mixup=self.args.mixup, mixup_alpha=self.args.mixup_alpha)
            else:
                output, h = self.model(inputs, ret='of')

            # ==== update loss and acc
            loss = self.criterion(output,
                                  reweighted_targets if self.args.mixup >= 0 or self.args.aug in ['cm', 'cutmix'] else targets)
            losses.update(loss.item(), targets.size(0))
            train_acc.update(torch.sum(output.argmax(dim=-1) == targets).item() / targets.size(0),
                             targets.size(0)
                             )

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

        for epoch in range(self.start_epoch, self.epochs):

            # ========= Train one epoch =========
            losses, train_acc = self.train_one_epoch()

            self.log.info('====>EPOCH_{epoch}, Loss:{loss:.4f}, Acc:{acc:.4f}'.format(epoch=epoch, loss=losses.avg, acc=train_acc.avg))
        
            wandb.log({'train/train_loss': losses.avg,
                       'train/train_acc': train_acc.avg,
                       'lr': self.optimizer.param_groups[0]['lr']},
                      step=epoch)

            # ========= evaluate on validation set =========
            val_acc = self.validate(epoch=epoch)
            wandb.log({'val/val_acc1': val_acc},step=epoch)

            # ========= measure NC =========
            if (epoch + 1) % self.args.debug == 0 and self.args.debug > 0:
                train_nc = analysis(self.model, self.train_loader_base, self.args)
                self.log.info('>>>>Epoch:{}, Train Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f}, NC3:{:.3f}'.format(
                    epoch, train_nc['loss'], train_nc['acc'], train_nc['nc1'], train_nc['nc3']))
                wandb.log({
                    'train_nc/nc1': train_nc['nc1'],
                    'train_nc/nc2h': train_nc['nc2_h'],
                    'train_nc/nc2w': train_nc['nc2_w'],
                    'train_nc/nc3': train_nc['nc3'],
                    'train_nc/nc3_d': train_nc['nc3_d'],
                    'train_nc2/nc21_h': train_nc['nc21_h'],
                    'train_nc2/nc22_h': train_nc['nc22_h'],
                    'train_nc2/nc21_w': train_nc['nc21_w'],
                    'train_nc2/nc22_w': train_nc['nc22_w'],
                }, step=epoch)

                test_nc = analysis(self.model, self.val_loader, self.args)
                wandb.log({
                    'test_nc/nc1': test_nc['nc1'],
                    'test_nc/nc2h': test_nc['nc2_h'],
                    'test_nc/nc2w': test_nc['nc2_w'],
                    'test_nc/nc3': test_nc['nc3'],
                    'test_nc/nc3_d': test_nc['nc3_d'],
                    'test_nc2/nc21_h': test_nc['nc21_h'],
                    'test_nc2/nc22_h': test_nc['nc22_h'],
                    'test_nc2/nc21_w': test_nc['nc21_w'],
                    'test_nc2/nc22_w': test_nc['nc22_w'],
                }, step=epoch)

                if (epoch + 1) % (self.args.debug * 5) == 0:
                    fig = plot_nc(train_nc)
                    wandb.log({"chart": fig}, step=epoch + 1)

            if self.args.scheduler in ['step', 'ms', 'multi_step', 'poly']:
                self.lr_scheduler.step()
            self.model.train()

        self.log.info('Best Testing Prec@1: {:.3f}\n'.format(best_acc1))

    def validate(self, epoch=None):
        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_loader):
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input, ret='o')
                _, pred = torch.max(output, 1)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        if self.args.coarse == 't':
            vectorized_map = np.vectorize(fine_id_coarse_id.get)
            preds = vectorized_map(np.array(all_preds))
        else:
            preds = np.array(all_preds)
        targets = np.array(all_targets)

        acc = np.sum(preds == targets)/len(targets)
        return acc

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