import pdb
import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from model import ResNet_norm
from imbalance_data import dataset_lt_data, cifar100Imbanlance, cifar10Imbanlance
import logging
import datetime
import wandb
from sklearn.metrics import confusion_matrix
from Trainer_bn import Trainer_bn

best_acc1 = 0

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        print("=> creating model '{}'".format('resnext50_32x4d'))
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        return net
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'mresnet32':
            net = ResNet_norm.mresnet32(args=args)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(num_class=args.num_classes)
        return net

def get_dataset(args):
    transform_train,transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbalance(transform=util.TwoCropTransform(transform_train),imbalance_rate=args.imbalance_rate, train=True,file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbalance(imbalance_rate=args.imbalance_rate, train=False, transform=transform_val,file_path=args.root)
        print("load cifar10")
        return trainset,testset

    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbalance(transform=util.TwoCropTransform(transform_train),imbalance_rate=args.imbalance_rate,
                                                         train=True,file_path=os.path.join(args.root,'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbalance(imbalance_rate=args.imbalance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/'))
        print("load cifar100")
        return trainset,testset

    if args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt,util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        return trainset,testset

    if args.dataset == 'iNaturelist2018':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt,util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        return trainset,testset

def main(args):
    
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        
    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  #"dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='bn_'+args.dataset,
               name= args.store_name.split('/')[-1]
               )
    wandb.config.update(args)
    main_worker(wandb.config)

def main_worker(args):

    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    # create model
    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if torch.cuda.is_available():
        model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M')
    fh = logging.FileHandler(os.path.join(args.root_model + args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)
    logging.info(args)

    # Data loading code
    train_dataset,val_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == args.num_classes

    cls_num_list = train_dataset.get_per_class_num()
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),num_workers=args.workers, persistent_workers=True,pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, persistent_workers=True,pin_memory=True)

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)

    # ==== weighted_loader
    cls_weight = 1.0 / (np.array(cls_num_list) ** args.resample_weighting)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.workers, persistent_workers=True,pin_memory=True,sampler=weighted_sampler)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    start_time = time.time()
    print("Training started!")
    trainer = Trainer_bn(args, model=model,train_loader=train_loader, val_loader=val_loader,weighted_train_loader=weighted_train_loader, per_class_num=train_cls_num_list,log=logging)
    trainer.train()
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))

if __name__ == '__main__':
    # train set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
    parser.add_argument('--root', type=str, default='../dataset/', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',choices=('resnet18', 'resnet34', 'resnet32', 'resnet50', 'mresnet32', 'resnext50_32x4d'))
    parser.add_argument('--branch2', default=False, action='store_true')                 # turn on
    parser.add_argument('--contrast', default=False, action='store_true')                # turn on
    parser.add_argument('--etf_cls', default=False, action='store_true')
    parser.add_argument('--feat', type=str, default='null')
    parser.add_argument('--norm', default=False, action='store_true')  # none|nn1|nn2
    parser.add_argument('--bn_type', type=str, default='bn')  # cbn: class balanced bn
    parser.add_argument('--bias', default=False, action='store_true')  # none|nn1|nn2    # turn on
    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh|hinge

    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument('--imbalance_type', default='exp', type=str, help='imbalance type')
    parser.add_argument('--imbalance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--beta', type=float, default=0.5, help="augment mixture")
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W',help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')
    parser.add_argument('--resample_weighting', default=0.2, type=float,help='weighted for sampling probability (q(1,k))')
    parser.add_argument('--label_weighting', default=1.0, type=float, help='weighted for Loss')
    parser.add_argument('--contrast_weight', default=10,type=int,help='Mixture Consistency  Weights')
    # etc.
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
    parser.add_argument('-p', '--print_freq', default=1000, type=int, metavar='N',help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
    parser.add_argument('--root_log', type=str, default='./output/log/')
    parser.add_argument('--root_model', type=str, default='./output/')
    parser.add_argument('--store_name', type=str, default='cifar')
    
    parser.add_argument('--debug',  type=int, default=10)
    args = parser.parse_args()
    
    if args.dataset == 'cifar10' or args.dataset == 'fmnist':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'ImageNet-LT':
        args.num_classes = 1000
    elif args.dataset == 'iNaturelist2018':
        args.num_classes = 8142
    elif args.dataset == 'tinyi':
        args.num_classes = 200
    if args.imbalance_rate < 1.0:
        args.knn = True
    
    file_name = args.store_name
    args.store_name = '{}_{}/{}/{}'.format(
        args.dataset, args.arch,
        args.imbalance_type + '_' + str(args.imbalance_rate),
        file_name
    )
    print(args)
    
    main(args)