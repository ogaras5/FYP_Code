from __future__ import print_function, division

import numpy as np
import torch

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from utils.data_helpers import *

import matplotlib.image as image

import os
import argparse
import random

parser = argparse.ArgumentParser(description='Visual data')
parser.add_argument('-d', '--dataset', default='cifar_10', type=str,
                    help='dataset to visual (default: cifar_10)')
parser.add_argument('-f', '--filename', type=str, default='/data/sarah',
                    help='parent location to load/save dataset (default: /data/sarah)')
parser.add_argument('--sample-size', type=int, metavar='N', required=True,
                    help='number of samples per class')
parser.add_argument('--manualSeed', type=int, default=12345, help='manual seed (default: 12345)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# If there are GPUs available to use
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Generate a random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

print('PyTorch Version:', torch.__version__)

# Validate dataset is cifar10 or cifar 100
assert args.dataset == 'cifar_10' or args.dataset == 'cifar_100', 'Dataset can only be cifar10 or cifar100'

# Create dictionary to ensure all images given unique label inside folde
val_img_dict = {}
if args.dataset == 'cifar_10':
    num_classes = range(10)
else:
    num_classes = range(100)
for i in num_classes:
    val_img_dict[i] = 0

# Function to save the images as a jpeg in the correct file
def imsave(inp, label, meanVal, stdVal):
    """
    Imsave for Tensor.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(meanVal)
    std = np.array(stdVal)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    val_img_dict[label] += 1
    filepath = '{}/{}/{}_samples/{}'.format(args.filename, args.dataset,
            args.sample_size, label)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if val_img_dict[label] <= args.sample_size:
        image.imsave('{}/{:04d}.png'.format(filepath, val_img_dict[label]), inp)

def run():
    # Normalization of images
    if args.dataset == 'cifar_10' or args.dataset == 'cifar_100':
        meanVal = [0.4914, 0.4822, 0.4465]
        stdVal =  [0.2023, 0.1994, 0.2010]
        normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])

    # transform for the training data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer,
    ])

    if args.dataset == 'cifar_10':
        train_set = CIFAR10('{}/cifar_10'.format(args.filename), train=True,
                            download=True, transform=train_transform)
    if args.dataset == 'cifar_100':
        train_set = CIFAR100('{}/cifar_100'.format(args.filename), train=True,
                            download=True, transform=train_transform)

    print('Training set Size: ', len(train_set))

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=100, num_workers=4, shuffle=True)

    # Loop through dataset once to create all required folders and images
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        for j in range(inputs.size()[0]):
            imsave(inputs.cpu().data[j], labels.cpu().data[j].item(), meanVal, stdVal)

if __name__ == '__main__':
    run()
