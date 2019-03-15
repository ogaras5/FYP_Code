from __future__ import print_function, division

import numpy as np
import pandas as pd
import argparse
import os
from utils.helpers import *

# Arguments for Plotting loss
parser = argparse.ArgumentParser(description='Compare loss plot for several models')
parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                    help='Dataset results to plot. All models must belong to\
                        same dataset (default: cifar10)')
parser.add_argument('-f', '--filepath', default='./losses', type=str,
                    help='Path to csv files (default: ./losses)')
parser.add_argument('-a', '--augmentations', default=['benchmark'], type=str, nargs='+',
                    help='Augementations to grab results for. Augementation name\
                    based on filename, e.g. rotation2, rotation30, etc (default: benchmark)')
parser.add_argument('-p' '--pretrained', dest='pretrained', action='store_true',
                    help='Results contain pretrained models (default: false)')
parser.add_argument('--accuracy', dest='accuracy', action='store_true',
                    help='Plot accuracy curve for cifar datasets')

args = parser.parse_args()

train_losses = []
valid_losses = []
train_acc = []
valid_acc = []

for augment in args.augmentations:
    # Setup path to csv file
    filepath = '{}/{}-{}.csv'.format(args.filepath, augment, args.dataset)
    if args.pretrained and not os.path.exists(filepath):
        filepath = '{}/{}-{}-pretrain.csv'.format(args.filepath, augment, args.dataset)
    # Load in csv file into pandas array
    df = pd.read_csv(filepath)
    train_losses.append(df['train'])
    valid_losses.append(df['valid'])
    if args.accuracy:
        train_acc.append(df['train_acc'])
        valid_acc.append(df['valid_acc'])

# Plot the learning curve, and accuracy if using imagenet
plotLosses(train_losses, valid_losses, args.augmentations)
plotLosses(train_acc, valid_acc, args.augmentations)

# TODO: Fix below to visualize accuracy for ImageNet
# if args.dataset == 'imagenet':
    # plotAccuracy(df['train_top1'], df['valid_top1'], title=args.augmentation)
    # plotAccuracy(df['train_top5'], df['valid_top5'], top=5, title=args.augmentation)
