from __future__ import print_function, division

import numpy as np
import pandas as pd
import argparse
from utils.helpers import *

# Arguments for Plotting loss
parser = argparse.ArgumentParser(description='Model loss plot')
parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                    help='Dataset results to plot (default: cifar10)')
parser.add_argument('-f', '--filepath', default='./losses', type=str,
                    help='Path to csv file (default: ./losses)')
parser.add_argument('-a', '--augmentation', default='benchmark', type=str,
                    help='Augementation to grab result for (default: benchmark)')
parser.add_argument('-p' '--pretrained', dest='pretrained', action='store_true',
                    help='Use pretrained model results (default: false)')
parser.add_argument('--accuracy', dest='accuraccy', action='storee_true',
                    help='Plot accuracy curve for cifar datasets')

args = parser.parse_args()

# Setup path to csv file
filepath = '{}/{}-{}.csv'.format(args.filepath, args.augmentation, args.dataset)
if args.pretrained and args.augmentation != 'benchmark':
    filepath = '{}/{}-{}-pretrain.csv'.format(args.filepath, args.augmentation, args.dataset)
# Load in csv file into pandas array
df = pd.read_csv(filepath)

# Plot the learning curve, and accuracy if using imagenet
plotLoss(df['train'], df['valid'], args.augmentation)
if ''.join(filter(lambda x: x.isalpha(), args.dataset)) == 'cifar' and args.accuracy:
    plotAccuracy(df['train_acc'], df['valid_acc'], title=args.augmentation)
if args.dataset == 'imagenet':
    plotAccuracy(df['train_top1'], df['valid_top1'], title=args.augmentation)
    plotAccuracy(df['train_top5'], df['valid_top5'], top=5, title=args.augmentation)
