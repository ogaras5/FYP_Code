"""
Script to train sampled version of CIFAR-10/CIFAR-100 Dataset.
Can be used to train both benchmark dataset and
dataset doubled with single augmentation.
"""
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
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from utils.progress import MonitorProgress
from utils.helpers import *
from utils.data_helpers import *

import argparse
import time
import os
import random

# Arguments for Data set
parser = argparse.ArgumentParser(description='PyTorch Sampled CIFAR10/CIFAR100 Training')
parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                    help='dataset to use for training, either cifar10 or cifar100 (default: cifar10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-s', '--sample-size', default=200, type=int, metavar='N',
                    help='number of samples per class (defalt: 200)')
# Arguments for Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N',
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
parser.add_argument('--depth', type=int, default=56, help='Model depth (default: 56)')
# Arguments for augmentation
parser.add_argument('--augmentation', type=str, default='benchmark',
                    help='Type of augmentation to apply to the dataset (default: benchmark)')
parser.add_argument('--value', type=int, default=25,
                    help='Value to use in augmentation')
parser.add_argument('--magnitude', type=float, default=0.5,
                    help='Magnitude of augmentation')
parser.add_argument('--probability', type=float, default=1.0,
                    help='Probability that the augmentation is applied to the image')
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
    # transforms for the training data
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
    ])

    # transform for the validation data
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                            [0.2023, 0.1994, 0.2010]),
    ])

    # Add augmentation to pipeline if wanted
    if ''.join(filter(lambda x: x.isalpha(), args.augmentation)) != "benchmark":
        # Add augmentation to pipeline (see utils/data_helpers.py for code)
        p = create_augmentation_pipeline(args.augmentation, args.probability, args.value, args.value, args.magnitude)

        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            p.torch_transform(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010]),
        ])

    # Validate dataset is cifar10 or cifar 100
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100'

    # Load datasets
    if args.dataset == 'cifar10':
        train_set = ImageFolder('/data/sarah/cifar_10/{}_samples'.format(args.sample_size),
                                transform=train_transform)
        valid_set = CIFAR10('/data/sarah/cifar_10', train=False,
                            download=True, transform=valid_transform)
        if ''.join(filter(lambda x: x.isalpha(), args.augmentation)) != "benchmark":
            augmented_set = ImageFolder('/data/sarah/cifar_10/{}_samples'.format(args.sample_size),
                                    transform=augment_transform)
            train_set = ConcatDataset((train_set, augmented_set))
        num_classes = 10
    else:
        train_set = ImageFolder('/data/sarah/cifar_100/{}_samples'.format(args.sample_size),
                                transform=train_transform)
        valid_set = CIFAR100('/data/sarah/cifar_100', train=False,
                             download=True, transform=valid_transform)
        if ''.join(filter(lambda x: x.isalpha(), args.augmentation)) != "benchmark":
            augmented_set = ImageFolder('/data/sarah/cifar_100/{}_samples'.format(args.sample_size),
                                        transform=augment_transform)
            train_set = ConcatDataset((train_set, augmented_set))
        num_classes = 100

    print('Total Images in Training set: {}'.format(len(train_set)))
    print('Total Images in Validation set: {}'.format(len(valid_set)))

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.train_batch,
                              shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch,
                              shuffle=False, num_workers=args.workers)

    # Model training
    def train_model(model, criterion, optimizer, num_epochs=10):
        since = time.time()

        train_losses = []
        valid_losses = []

        best_valid_acc = 0.0
        best_train_acc = 0.0

        for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
            print('Epoch: {}/{}'.format(epoch, args.start_epoch + num_epochs - 1))

            # Each epoch has a training and a validation phase
            # Train Phase
            adjust_learning_rate(optimizer, epoch)
            print('Learning Rate: {:6f}'.format(state['lr']))
            model.train()

            # Create progress bar
            progress = MonitorProgress(total=len(train_set))

            # Save Training data
            train_loss = MovingAverage()

            # Save training predictions and true labels
            y_pred = []
            y_true = []

            for batch, targets in train_loader:
                # Move the training data to the CPU
                batch = batch.to(device)
                targets = targets.to(device)

                # Clear the previous gradient computation
                optimizer.zero_grad()

                # Forward propagate
                predictions = model(batch)

                # Calculate the loss
                loss = criterion(predictions, targets)

                # backpropagation to compute gradients
                loss.backward()

                # Update model weights
                optimizer.step()

                # Update Average Loss
                train_loss.update(loss)

                # Update progress bar
                progress.update(batch.shape[0], train_loss)

                # Save predictions and actual labels
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predictions.argmax(dim=1).cpu().numpy())

            print('Training Loss: ', train_loss)
            train_losses.append(train_loss.value)

            # Calculate validation accuracy and see if it is the best accuracy
            y_true = torch.tensor(y_true, dtype=torch.int64)
            y_pred = torch.tensor(y_pred, dtype=torch.int64)
            accuracy = torch.mean((y_pred == y_true).float())
            print('Training accuracy: {:4f}%'.format(float(accuracy)*100))
            if accuracy > best_train_acc:
                best_train_acc = accuracy

            # Validation Phase
            model.eval()

            valid_loss = RunningAverage()

            # Keep track of predictions
            y_pred = []

            # We don't need gradients for validation, so wrap in no_grad to save memory
            with torch.no_grad():
                for batch, targets in valid_loader:
                    #Move the validation batch to CPU
                    batch = batch.to(device)
                    targets = targets.to(device)

                    # Forward Propagation
                    predictions = model(batch)

                    # Calculate Loss
                    loss = criterion(predictions, targets)

                    # Update running loss value
                    valid_loss.update(loss)

                    # Save predictions
                    y_pred.extend(predictions.argmax(dim=1).cpu().numpy())

            print('Validation Loss: ', valid_loss)
            valid_losses.append(valid_loss.value)

            # Calculate validation accuracy and see if it is the best accuracy
            y_true = torch.tensor(valid_set.test_labels, dtype=torch.int64)
            y_pred = torch.tensor(y_pred, dtype=torch.int64)
            accuracy = torch.mean((y_pred == y_true).float())
            print('Validation accuracy: {:4f}%'.format(float(accuracy)*100))
            if accuracy > best_valid_acc:
                best_valid_acc = accuracy

            # Save checkpoint
            checkpoint_filename = './checkpoints/{}/{}_sample/{}/{}-{}-{:03d}.pkl'.format(args.dataset,
                                    args.sample_size, ''.join(filter(lambda x: x.isalpha(), args.augmentation)),
                                    args.augmentation, args.dataset, epoch)
            save_checkpoint(optimizer, model, epoch, checkpoint_filename)

            # Save taining loss, and validation loss to a csv
            df = pd.DataFrame({
                'epoch': range(args.start_epoch, len(train_losses) + args.start_epoch),
                'train': train_losses,
                'valid': valid_losses
            })
            df.set_index('epoch', inplace=True)
            # Save to tmp csv file
            df.to_csv("./losses/{}/{}_sample/{}-{}-tmp.csv".format(args.dataset, args.sample_size, args.augmentation, args.dataset))

            # Save details about time of training to tmp file
            time_elapsed = time.time() - since
            fp = open('./losses/{}/{}_sample/{}-details-tmp.txt'.format(args.dataset, args.sample_size, args.dataset), 'w+')
            fp.write('\nResults for training {}:\n Start epoch {}, End epoch {}, Training time {:.0f}m {:.0f}s, Best Validation accuracy {:4f}%, Best Training accuracy {:4f}%'.format(args.augmentation,
                    args.start_epoch, epoch,
    	            time_elapsed // 60, time_elapsed % 60, float(best_valid_acc)*100, float(best_train_acc)*100))
            fp.close()

        # Give some details about how long the training took
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
               time_elapsed // 60, time_elapsed % 60))
        print('Best Validation Accuracy: {:4f}%'.format(float(best_valid_acc)*100))
        print('Best Training Accuracy: {:4f}%'.format(float(best_train_acc)*100)) 
        fp = open('./losses/{}/{}_sample/{}-details.txt'.format(args.dataset, args.sample_size, args.dataset), 'a+')
        fp.write('\nResults for training {}:\n Start epoch {}, End epoch {}, Training time {:.0f}m {:.0f}s, Best Validation accuracy {:4f}%, Best Training accuracy {:4f}%'.format(args.augmentation,
                    args.start_epoch, args.start_epoch + args.epochs - 1,
    	            time_elapsed // 60, time_elapsed % 60, float(best_valid_acc)*100, float(best_train_acc)*100))
        fp.close()
        return train_losses, valid_losses, y_pred

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
    optimizer = optim.SGD(model_res.parameters(), lr=args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)

    # Load model if starting from checkpoint
    if args.start_epoch != 1:
        epoch = load_checkpoint(optimizer, model_res,
                            './checkpoints/{}/{}_sample/{}/{}-{}-{:03d}.pkl'.format(args.dataset,
                            args.sample_size, ''.join(filter(lambda x: x.isalpha(), args.augmentation)),
                            args.augmentation, args.dataset, args.start_epoch-1))
        print('Resuming training from epoch', epoch)

    # Train model
    print('\nTraining and validating model for {} epoch{}...'
            .format(args.epochs, "s"[args.epochs==1:]))
    train_losses, valid_losses, y_pred = train_model(model_res, criterion,
                                                     optimizer, args.epochs)

    # Save taining loss, and validation loss to a csv
    df = pd.DataFrame({
        'epoch': range(args.start_epoch, len(train_losses) + args.start_epoch),
        'train': train_losses,
        'valid': valid_losses
    })
    df.set_index('epoch', inplace=True)

    # If starting from later epoch grab results already in csv file and make new dataframe
    if args.start_epoch != 1:
        old_df = pd.read_csv('./losses/{}/{}_sample/{}-{}.csv'.format(args.dataset, args.sample_size, args.augmentation, args.dataset))
        old_df.set_index('epoch', inplace=True)
        df = old_df.join(df, on='epoch', how='outer', lsuffix='_df1', rsuffix='_df2')
        df.loc[df['train_df2'].notnull(), 'train_df1'] = df.loc[df['train_df2'].notnull(), 'train_df2']
        df.loc[df['valid_df2'].notnull(), 'valid_df1'] = df.loc[df['valid_df2'].notnull(), 'valid_df2']
        df.drop(['train_df2', 'valid_df2'], axis=1, inplace=True)
        df.rename(columns={'train_df1': 'train', 'valid_df1': 'valid'}, inplace=True)

    # Save to csv file
    df.to_csv('./losses/{}/{}_sample/{}-{}.csv'.format(args.dataset, args.sample_size, args.augmentation, args.dataset))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
