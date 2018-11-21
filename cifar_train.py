from __future__ import print_function, division

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
from torchvision.datasets import CIFAR10, CIFAR100

from utils.progress import MonitorProgress
from utils.helpers import *

import argparse
import time
import os
import random

# Arguments for Data set
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                    help='dataset to use for training, either cifar10 or cifar100 (default: cifar10)')
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

state = {k: v for k, v in args._get_kwargs()}

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

    # Validate dataset is cifar10 or cifar 100
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100'

    # Load datasets
    if args.dataset == 'cifar10':
        train_set = CIFAR10('./data/cifar_10', train=True,
                            download=True, transform=train_transform)
        valid_set = CIFAR10('./data/cifar_10', train=False,
                            download=True, transform=valid_transform)
        num_classes = 10
    else:
        train_set = CIFAR100('./data/cifar_100', train=True,
                             download=True, transform=train_transform)
        valid_set = CIFAR100('./data/cifar_100', train=False,
                             download=True, transform=valid_transform)
        num_classes = 100

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

        best_acc = 0.0

        for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
            print('Epoch: {}/{}'.format(epoch, num_epochs))

            # Each epoch has a training and a validation phase
            # Train Phase
            adjust_learning_rate(optimizer, epoch)
            print('Learning Rate: {:6f}'.format(state['lr']))
            model.train()

            # Create progress bar
            progress = MonitorProgress(total=len(train_set))

            # Save Training data
            train_loss = MovingAverage()

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

            print('Training Loss: ', train_loss)
            train_losses.append(train_loss.value)

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
            if accuracy > best_acc:
                best_acc = accuracy

            # Save checkpoint
            checkpoint_filename = './checkpoints/benchmark-{}-{:03d}.pkl'.format(args.dataset, epoch)
            save_checkpoint(optimizer, model, epoch, checkpoint_filename)

        # Give some details about how long the training took
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
               time_elapsed // 60, time_elapsed % 60))
        print('Best value Accuracy: {:4f}%'.format(float(best_acc)*100))
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
        model_res = epoch = load_checkpoint(optimizer, model,
                            './checkpoints/benchmark-{}-{:03d}.pkl'
                            .format(args.dataset, args.start_epoch))
        print('Resuming training from epoch', epoch)

    # Check if the model is just being evaluted
    if args.evaluate:
        print('\nEvaluation only for epoch {}'.format(args.start_epoch))
        # TODO: Create evaluation function
        #valid_losses, y_pred = test_model(model, criterion, args.epochs)
        return

    # Train model
    print('\nTraining and validating model for {} epoch{}...'
            .format(args.epochs, "s"[args.epochs==1:]))
    train_losses, valid_losses, y_pred = train_model(model_res, criterion,
                                                     optimizer, args.epochs)

    # Visualize the training loss
    plotLoss(train_losses, valid_losses)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
