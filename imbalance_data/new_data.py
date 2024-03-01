import os
import torchvision
from . import cifar10Imbanlance, cifar100Imbanlance, dataset_lt_data, new_cifar100Imbanlance
from .new_transform import get_transform, TwoCropTransform
from torchvision import datasets, transforms

data_folder = '/' # for greene,  '../dataset' for local

def get_dataset(args):
    transform_train, transform_val = get_transform(args, args.dataset)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(transform=transform_train, imbanlance_rate=args.imbanlance_rate,
                                                       train=True, file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                      transform=transform_val, file_path=args.root)
        print("load cifar10")
        return trainset, testset

    elif args.dataset == 'cifar100':
        trainset = new_cifar100Imbanlance.Cifar100Imbanlance(transform=transform_train,
                                                         imbanlance_rate=args.imbanlance_rate, train=True,
                                                         file_path=os.path.join(args.root, 'cifar-100-python/'))
        testset = new_cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                        transform=transform_val,
                                                        file_path=os.path.join(args.root, 'cifar-100-python/'))
        print("load cifar100")
        return trainset, testset

    elif args.dataset == 'fmnist':
        fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root="data").train_data.float()
        trainset = datasets.FashionMNIST("data", download=True, train=True, transform=transform_train)
        testset = datasets.FashionMNIST("data", download=True, train=False, transform=transform_val)
        return trainset, testset

    elif args.dataset == 'tinyi':  # image_size:64 x 64
        trainset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val'), transform_val)
        return trainset, testset

    elif args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt, TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return trainset, testset

    elif args.dataset == 'iNaturelist2018':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt, TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return trainset, testset