"""
Script to train resnet model on imagenet with image size of 32x32 pixels.
Script can be used to train resnet model using no augmentation, i.e. benchmark for imagenet
dataset, or for a single augmentation. Please ensure that the file structure is correct before 
continue, otherwise checkpoints and results will not be saved and cause errors!
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import model.resnet_cifar as models

from torch.utils.data import DataLoader, ConcatDataset
from torch.nn import DataParallel

from utils.progress import MonitorProgress
from utils.helpers import *
from utils.data_helpers import *
from utils.ImageNet32_dataset import ImageNet32

import argparse
import time
import os
import random

# Arguments for Data set
parser = argparse.ArgumentParser(description='PyTorch 32x32 ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Arguments for Optimization options
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of epochs to train (default: 25)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='epoch to start training on (must have checkpoint saved)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='batch size for training (default: 32)')
parser.add_argument('--valid-batch', default=100, type=int, metavar='N',
                    help='batch size for testing (default: 25)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                    help='Decrease learning rate at these epochs (default: [30 60])')
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
                    help='Type of augmentation to apply to the dataset (default: none)')
parser.add_argument('--value', type=int, default=25,
                    help='Value to use in augmentation (default: 25)')
parser.add_argument('--magnitude', type=float, default=0.5,
                    help='Magnitude of augmentation (default: 0.5)')
parser.add_argument('--probability', type=float, default=1.0,
                    help='Probability that the augmentation is applied to the image (default: 1.0)')
# Arguments for miscellaneous
parser.add_argument('--manualSeed', type=int, default=12345, help='manual seed')
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
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # transform for the validation data
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_set = ImageNet32('/data/sarah/imagenet-32', train=True,
                              transform=train_transform)
    valid_set = ImageNet32('/data/sarah/imagenet-32', train=False,
                              transform=valid_transform)

    # Add augmentation to pipeline if wanted
    if args.augmentation != "benchmark":
        # Add augmentation to pipeline (see utils/data_helpers.py for code)
        p = create_augmentation_pipeline(args.augmentation, args.probability, args.value, args.value, args.magnitude)
        # transforms for the training data
        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            p.torch_transform(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        augment_set = ImageNet32('/data/sarah/imagenet-32', train=True,
                                    transform=augment_tansform)
        train_set = ConcatDataset((train_set, augment_set))

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.train_batch,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    # Model training
    def train_model(model, criterion, optimizer, num_epochs=10):
        since = time.time()

        train_losses = []
        train_top1s = []
        train_top5s = []
        valid_losses = []
        valid_top1s = []
        valid_top5s = []

        best_acc = 0.0

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
            train_top1 = MovingAverage()
            train_top5 = MovingAverage()

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
                prec1, prec5 = accuracy(predictions.data, targets.data, topk=(1,5))
                train_top1.update(prec1)
                train_top5.update(prec5)

                # Update progress bar
                progress.update(batch.shape[0], train_loss)

            print('Training Loss: ', train_loss)
            print('Training Top1: ', train_top1)
            print('Training Top5: ', train_top5)
            train_losses.append(train_loss.value)
            train_top1s.append(train_top1.value)
            train_top5s.append(train_top5.value)

            # Validation Phase
            model.eval()

            valid_loss = RunningAverage()
            valid_top1 = RunningAverage()
            valid_top5 = RunningAverage()

            # Keep track of predictions and actual labels
            y_true = []
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
                    prec1, prec5 = accuracy(predictions.data, targets.data, topk=(1,5))
                    valid_top1.update(prec1)
                    valid_top5.update(prec5)

                    # Save predictions and actual labels
                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(predictions.argmax(dim=1).cpu().numpy())

            print('Validation Loss: ', valid_loss)
            print('Validation Top1: ', valid_top1)
            print('Validation Top5: ', valid_top5)
            valid_losses.append(valid_loss.value)
            valid_top1s.append(valid_top1.value)
            valid_top5s.append(valid_top5.value)

            # Calculate validation accuracy and see if it is the best accuracy
            y_true = torch.tensor(y_true, dtype=torch.int64)
            y_pred = torch.tensor(y_pred, dtype=torch.int64)
            acc = torch.mean((y_pred == y_true).float())
            print('Validation accuracy: {:4f}%'.format(float(acc)*100))
            if acc > best_acc:
                best_acc = acc

            # Save checkpoint
            checkpoint_filename = './checkpoints/imagenet-32/{}-imagenet-32-{:03d}.pkl'.format(args.augmentation, epoch)
            save_checkpoint(optimizer, model, epoch, checkpoint_filename)

            # Save taining loss, and validation loss to a csv
            df = pd.DataFrame({
                'epoch': range(args.start_epoch, len(train_losses) + args.start_epoch),
                'train': train_losses,
                'train_top1': train_top1s,
                'train_top5': train_top5s,
                'valid': valid_losses,
                'valid_top1': valid_top1s,
                'valid_top5': valid_top5s
            })
            df.set_index('epoch', inplace=True)
            # Save to tmp csv file
            df.to_csv("./losses/imagenet-32/{}-imagenet-tmp.csv".format(args.augmentation))

            #Save details about time of training to tmp file
            time_elapsed = time.time() - since
            fp = open('./losses/imagenet-32-details-tmp.txt')
            fp.write('\nResults for training {}:\n Start epoch {}, End epoch {}, Training time {:.0f}m {:.0f}s, Best Validation accuracy {:4f}%'.format(args.augmentation,
                        args.start_epoch, epoch,
                        time_elapsed // 60, time_elapsed % 60, float(best_acc)*100))
            fp.close()

        # Give some details about how long the training took
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
               time_elapsed // 60, time_elapsed % 60))
        print('Best value Accuracy: {:4f}%'.format(float(best_acc)*100))
        fp = open('./losses/imagenet-32-details.txt', 'a+')
        fp.write('\nResults for training {}:\n Start epoch {}, End epoch {}, Training time {:.0f}m {:.0f}s, Best Validation accuracy {:4f}%'.format(args.augmentation,
                    args.start_epoch, args.start_epoch + args.epochs - 1,
                    time_elapsed // 60, time_elapsed % 60, float(best_acc)*100))
        fp.close()
        return train_losses, train_top1s, train_top5s, valid_losses, valid_top1s, valid_top5s, y_pred

    # Model with 1000 class output
    print('Creating model...')
    model_res = models.resnet(depth=args.depth)
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
    if args.start_epoch != 1 and not args.evaluate:
        epoch = load_checkpoint(optimizer, model_res,
                            './checkpoints/imagenet-32/{}-imagenet-32-{:03d}.pkl'
                            .format(args.augmentation, args.start_epoch-1))
        print('Resuming training from epoch', epoch)

    # Train model
    print('\nTraining and validating model for {} epoch{}...'
            .format(args.epochs, "s"[args.epochs==1:]))
    train_losses, train_top1s, train_top5s, valid_losses, valid_top1s, valid_top5s, y_pred = train_model(model_res, criterion,
                                                     optimizer, args.epochs)

    # Save taining loss, and validation loss to a csv
    df = pd.DataFrame({
        'epoch': range(args.start_epoch, len(train_losses) + args.start_epoch),
        'train': train_losses,
        'train_top1': train_top1s,
        'train_top5': train_top5s,
        'valid': valid_losses,
        'valid_top1': valid_top1s,
        'valid_top5': valid_top5s
    })
    df.set_index('epoch', inplace=True)

    # If starting from later epoch grab results already in csv file and make new dataframe
    if args.start_epoch != 1:
        old_df = pd.read_csv('./losses/imagenet-32/{}-imagenet.csv'.format(args.augmentation))
        old_df.set_index('epoch', inplace=True)
        df = old_df.join(df, on='epoch', how='outer', lsuffix='_df1', rsuffix='_df2')
        df.loc[df['train_df2'].notnull(), 'train_df1'] = df.loc[df['train_df2'].notnull(), 'train_df2']
        df.loc[df['valid_df2'].notnull(), 'valid_df1'] = df.loc[df['valid_df2'].notnull(), 'valid_df2']
        df.loc[df['train_top1_df2'].notnull(), 'train_top1_df1'] = df.loc[df['train_top1_df2'].notnull(), 'train_top1_df2']
        df.loc[df['valid_top1_df2'].notnull(), 'valid_top1_df1'] = df.loc[df['valid_top1_df2'].notnull(), 'valid_top1_df2']
        df.loc[df['train_top5_df2'].notnull(), 'train_top5_df1'] = df.loc[df['train_top5_df2'].notnull(), 'train_top5_df2']
        df.loc[df['valid_top5_df2'].notnull(), 'valid_top5_df1'] = df.loc[df['valid_top5_df2'].notnull(), 'valid_top5_df2']
        df.drop(['train_df2', 'train_top1_df2', 'train_top5_df2', 'valid_df2', 'valid_top1_df2', 'valid_top5_df2'], axis=1, inplace=True)
        df.rename(columns={'train_df1': 'train', 'train_top1_df1': 'train_top1', 'train_top5_df1': 'train_top5', 'valid_df1': 'valid', 'valid_top1_df1': 'valid_top1', 'valid_top5_df1': 'valid_top5'}, inplace=True)

    # Save to csv file
    df.to_csv("./losses/imagenet-32/{}-imagenet.csv".format(args.augmentation))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
