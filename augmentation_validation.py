from __future__ import print_function, division

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import Augmentor
import model.resnet_cifar as models

from torch.utils.data import DataLoader, ConcatDataset
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
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Validation')
parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                    help='dataset to use for training, either cifar10 or cifar100 (default: cifar10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Arguments for Optimization options
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', default=31, type=int, metavar='N',
                    help='epoch to start training on (must have checkpoint saved)')
parser.add_argument('--valid-batch', default=100, type=int, metavar='N',
                    help='batch size for testing (default: 100)')
# Arguments for model architecture
parser.add_argument('--depth', type=int, default=56, help='Model depth (default: 56)')
# Arguments for augmentation
parser.add_argument('--augmentation', type=str, default='rotation',
                    help='Type of augmentation to apply to the dataset')
parser.add_argument('--model-augmentation', type=str, default='rotation',
                    help='Type of augmentation used to train model')
# Arguments for miscellaneous
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

def main():
    # transform for the validation data
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                            [0.2023, 0.1994, 0.2010]),
    ])

    # Validate dataset is cifar10 or cifar 100
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100'

    # Load datasets
    if args.dataset == 'cifar10':
        valid_set = ImageFolder('/data/sarah/cifar_10/{}'.format(args.augmentation),
                                transform=valid_transform)
        num_classes = 10
    else:
        valid_set = ImageFolder('/data/sarah/cifar_100/{}'.format(ags.augmentation),
                                transform=valid_transform)
        num_classes = 100

    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch,
                                shuffle=False, num_workers=args.workers)

    # Evaluation of model
    def evaluate_model(model, criterion, optimizer, num_epochs):
        valid_losses = []
        valid_acc = []
        best_acc = 0.0

        for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
            cur_epoch = load_checkpoint(optimizer, model,
                            './checkpoints/{}/{}-{}-{:03d}.pkl'
                            .format(args.dataset, args.model_augmentation, args.dataset, epoch))

            print('Epoch: {}/{}'.format(epoch, args.start_epoch + num_epochs - 1))

            # Validation Phase
            model.eval()

            # Create progress bar
            progress = MonitorProgress(total=len(valid_set))
            valid_loss = RunningAverage()

            # Keep track of predictions
            y_pred = []
            y_true = []

            # We don't need gradients for validation, so wrap in no_grad to save memory
            with torch.no_grad():
                for batch, targets in valid_loader:
                    #Move the validation batch to GPU
                    batch = batch.to(device)
                    targets = targets.to(device)

                    # Forward Propagation
                    predictions = model(batch)

                    # Calculate Loss
                    loss = criterion(predictions, targets)

                    # Update running loss value
                    valid_loss.update(loss)

                    # Save predictions
                    y_true.extend(targets.cpu().numpy())
                    print(len(y_true))
                    y_pred.extend(predictions.argmax(dim=1).cpu().numpy())
                    print(len(y_pred))

                    # Update progress bar
                    progress.update(batch.shape[0], valid_loss)

            print('Validation Loss: ', valid_loss)
            valid_losses.append(valid_loss.value)

            # Calculate validation accuracy and see if it is the best accuracy
            y_true = torch.tensor(y_true, dtype=torch.int64)
            y_pred = torch.tensor(y_pred, dtype=torch.int64)
            accuracy = torch.mean((y_pred == y_true).float())
            valid_acc.append(accuracy)
            print('Validation accuracy: {:4f}%'.format(float(accuracy)*100))
            if accuracy > best_acc:
                best_acc = accuracy

            df = pd.DataFrame({
                'epoch': range(args.start_epoch, len(train_losses) + args.start_epoch),
                'valid_acc': valid_acc,
                'valid': valid_losses
            })
            df.set_index('epoch', inplace=True)
            # Save to tmp csv file
            df.to_csv("./losses/{}-{}-{}-tmp.csv".format(args.model_augmentation, args.augmentation, args.dataset))
            fp = open('./losses/{}-{}-details-tmp.txt'.format(args.dataset, args.model_augmentation), 'w+')
            fp.write('\nResults for validating {} model:\n Start epoch {}, End epoch {}, Augmentation {}, Best Validation accuracy {:4f}%'.format(args.model_augmentation,
                    args.start_epoch, epoch,
    	            args.augmentation, float(best_acc)*100))
            fp.close()

        print('Best value Accuracy: {:4f}%'.format(float(best_acc)*100))
        fp = open('./losses/{}-{}-details.txt'.format(args.dataset, args.model_augmentation), 'a+')
        fp.write('\nResults for validating {} model:\n Start epoch {}, End epoch {}, Augmentation {}, Best Validation accuracy {:4f}%'.format(args.model_augmentation,
                    args.start_epoch, args.start_epoch + args.epochs - 1,
    	            args.augmentation, float(best_acc)*100))
        fp.close()
        return valid_losses, y_pred, valid_acc, best_acc

    # Model
    print('Creating model...')
    model_res = models.resnet(num_classes=num_classes, depth=args.depth)
    if torch.cuda.device_count() > 1:
        model_res = DataParallel(model_res).cuda()
        cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model_res.parameters())/1000000.0))
    model_res.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Using Sochastic Gradient Descent
    optimizer = optim.SGD(model_res.parameters(), lr=0.1,
                             momentum=0.9, weight_decay=1e-4)

    # Start evaluating the model
    valid_losses, y_pred, valid_acc, best_acc = evaluate_model(model_res, criterion, optimizer, args.epochs)

    # Save taining loss, and validation loss to a csv
    df = pd.DataFrame({
        'epoch': range(args.start_epoch, len(valid_losses) + args.start_epoch),
        'valid_acc': valid_acc,
        'valid': valid_losses
    })
    df.set_index('epoch', inplace=True)

    # If starting from later epoch grab results already in csv file and make new dataframe
    if args.start_epoch != 1:
        old_df = pd.read_csv('./losses/{}-{}-{}.csv'.format(args.model_augmentation, args.augmentation, args.dataset))
        old_df.set_index('epoch', inplace=True)
        df = old_df.join(df, on='epoch', how='outer', lsuffix='_df1', rsuffix='_df2')
        df.loc[df['valid_acc_df2'].notnull(), 'valid_acc_df1'] = df.loc[df['valid_acc_df2'].notnull(), 'valid_acc_df2']
        df.loc[df['valid_df2'].notnull(), 'valid_df1'] = df.loc[df['valid_df2'].notnull(), 'valid_df2']
        df.drop(['valid_acc_df2', 'valid_df2'], axis=1, inplace=True)
        df.rename(columns={'valid_acc_df1': 'valid_acc', 'valid_df1': 'valid'}, inplace=True)

    # Save to csv file
    df.to_csv("./losses/{}-{}-{}.csv".format(args.model_augmentation, args.augmentation, args.dataset))

    # Save accuracy to another csv for heatmap
    df2 = pd.DataFrame({
        'model_augmentation':args.model_augmentation,
        args.augmentation:best_acc,
    })
    df2.set_index('model_augmentation', inplace=True)
    df2.to_csv("./losses/validate-{}-{}-{}.csv".format(args.model_augmentation, args.augmentation, args.dataset))    
