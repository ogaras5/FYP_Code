"""
Script to train resnet model on tiny imagenet
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import model.resnet_cifar as models

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision.datasets import ImageFolder

from utils.progress import MonitorProgress
from utils.helpers import *
from utils.data_helpers import *

import argparse
import time
import os
import random

# Arguments for Data set
parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Arguments for Optimization options
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='epoch to start training on (must have checkpoint saved)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='batch size for training (default: 128)')
parser.add_argument('--valid-batch', default=100, type=int, metavar='N',
                    help='batch size for testing (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122],
                    help='Decrease learning rate at these epochs (default: [81 122])')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Arguments for model architecture
parser.add_argument('--depth', type=int, default=56, help='Model depth (default: 29)')
# Arguments for miscellaneous
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# If there are GPUs available to use
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate a random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    print("------Setup Validation Data------")
    print("Create seperate folders for each class and place validation data in "
          "correct classes")
    create_val_folder('./data/tiny-imagenet-200')
    print("Completed")
    # transforms for the training data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        ])

    # transform for the validation data
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_set = ImageFolder('./data/tiny-imagenet-200/train',
                              transform=train_transform)
    valid_set = ImageFolder('./data/tiny-imagenet-200/val/images',
                              transform=valid_transform)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.train_batch,
                              shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch,
                              shuffle=False, num_workers=args.workers)

    # Get list of class names, e.g. n01443537, n01629819, etc... and create dictionary
    class_names = train_set.classes
    num_classes = len(class_names)
    tiny_class = class_extractor(class_names, './data/tiny-imagenet-200')

if __name__ == '__main__':
    main()
