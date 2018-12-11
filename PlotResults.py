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

args = parser.parse_args()

filepath = '{}/{}-{}.csv'.format(args.filepath, args.augmentation, args.dataset)
df = pd.read_csv(filepath)

plotLoss(df['train'], df['valid'])
