import shutil
from torch.utils import data
import copy
import os
from imbalance_data.cifar100Imbanlance import *
from imbalance_data.cifar10Imbanlance import *
from imbalance_data.dataset_lt_data import *
import utils.moco_loader as moco_loader

import torch
from torchvision.transforms import v2
import numpy as np

import pdb


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # pdb.set_trace()
        h = img.shape[1]
        w = img.shape[2]

        mask = torch.ones((h, w))

        for n in range(self.n_holes):
            y = torch.randint(high=h, size=(1,)).item()
            x = torch.randint(high=w, size=(1,)).item()

            y1 = (y - self.length // 2)
            y2 = (y + self.length // 2)
            x1 = (x - self.length // 2)
            x2 = (x + self.length // 2)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img
    
# only cifar10 and cifar100 are supported
def get_transform(args, dataset):
    if dataset == "cifar10":
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)
    
    elif dataset == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    if args.aug == 'none':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif args.aug == 'crop':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif args.aug == 'cutout':
        if args.cutoutholes and args.cutoutlength:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=args.cutoutholes, length=args.cutoutlength),
                transforms.Normalize(mean, std),
            ])
    elif args.aug == 'colorjitter':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform_train,transform_val


    # elif dataset == "fmnist":
    #     fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root="data").train_data.float()
    #     transform_train = transforms.Compose([transforms.Resize((32, 32)),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize((fashion_mnist.mean() / 255,), (fashion_mnist.std() / 255,))
    #                                         ])
    #     transform_val = transforms.Compose([transforms.Resize((32, 32)),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((fashion_mnist.mean() / 255,), (fashion_mnist.std() / 255,))
    #                                         ])
    #     return transform_train, transform_val

    # elif dataset == 'tinyi':  # image_size:64 x 64
    #     transform_train = transforms.Compose([transforms.ToTensor(),
    #                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                                           ])
    #     tranform_val = transforms.Compose([transforms.ToTensor(),
    #                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                                        ])
    #     return transform_train, tranform_val

    # elif dataset == "ImageNet-LT":
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     rgb_mean = (0.485, 0.456, 0.406)
    #     ra_params = dict(translate_const=int(224 * 0.45),img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    #     augmentation_sim = [
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
    #         ], p=1.0),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ]
    #     transform_val = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize])

    #     transform_train = transforms.Compose(augmentation_sim)

    #     return transform_train, transform_val

    # elif dataset == "iNaturelist2018":
    #     normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
    #     rgb_mean = (0.485, 0.456, 0.406)
    #     ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

    #     augmentation_sim = [
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
    #         ], p=1.0),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ]
    #     transform_val = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize])

    #     transform_train = transforms.Compose(augmentation_sim)

    #     return transform_train, transform_val