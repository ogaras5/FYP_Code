from __future__ import print_function, division

import numpy as np
import pandas as pd
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
from utils.data_helpers import *
import matplotlib.pyplot as plt

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
parser.add_argument('--valid-batch', default=100, type=int, metavar='N',
                    help='batch size for testing (default: 100)')
# Arguments for model architecture
parser.add_argument('--depth', type=int, default=56, help='Model depth (default: 56)')
# Arguments for augmentation
parser.add_argument('--model-name', type=str, default='benchmark',
                    help='Trained model to demo e.g. erase-200')
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
        valid_set = CIFAR10('./data/cifar_10', train=False,
                            download=True, transform=valid_transform)
        num_classes = 10
        classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    else:
        valid_set = CIFAR100('./data/cifar_100', train=False,
                            download=True, transform=valid_transform)
        num_classes = 100

    valid_loader = DataLoader(valid_set, batch_size=args.valid_batch,
                                shuffle=False, num_workers=args.workers)

    # Evaluation of model
    def evaluate_model(model, criterion, optimizer):
        valid_losses = []
        valid_acc = []

        # Load the demo model
        epoch = load_checkpoint(optimizer, model,
                        './trained_models/{}-{}.pkl'
                        .format(args.model_name, args.dataset))

        print('{} model best accuracy from epoch: {}'.format(args.model_name, epoch))

        # Validation Phase
        model.eval()

        # Create progress bar
        progress = MonitorProgress(total=len(valid_set))
        valid_loss = RunningAverage()

        # Keep track of predictions
        y_pred = []

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
                y_pred.extend(predictions.argmax(dim=1).cpu().numpy())

                # Update progress bar
                progress.update(batch.shape[0], valid_loss)

        print('Validation Loss: ', valid_loss)
        valid_losses.append(valid_loss.value)

        # Calculate validation accuracy and see if it is the best accuracy
        y_true = torch.tensor(valid_set.test_labels, dtype=torch.int64)
        y_pred = torch.tensor(y_pred, dtype=torch.int64)
        accuracy = torch.mean((y_pred == y_true).float())
        valid_acc.append(float(accuracy)*100)
        print('Validation accuracy: {:4f}%'.format(float(accuracy)*100))
        return valid_losses, y_pred, valid_acc

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
    valid_losses, y_pred, valid_acc = evaluate_model(model_res, criterion, optimizer)

    # Show some facts about about the Evaluation!
    y_true = torch.tensor(valid_set.test_labels, dtype=torch.int64)
    num_errors = torch.sum((y_pred != y_true).float())
    print('Validation errors {} (out of {})'.format(int(num_errors), len(valid_set)))

    # Pull out examples of mistakes in the valid set
    error_indicator = y_pred != y_true
    error_examples = []
    for i in range(10000):
        if error_indicator[i] == 1:
            error_examples.append(valid_set.test_data[i])

    # Show what label they were given and the actual label
    sample = error_examples[:25]
    y_true_25 = y_true[error_indicator][:25].numpy()
    y_pred_25 = y_pred[error_indicator][:25].numpy()
    print('y_true:', y_true_25)
    print('y_pred:', y_pred_25)

    # Start plotting...
    fig = plt.figure(figsize=(14,14))

    for i in range(len(sample)):
        ax = plt.subplot(len(sample)//5, 5, i+1)
        plt.imshow(sample[i])
        plt.xticks([])
        plt.title('Predict:{}'.format(classes[y_pred_25[i]]))
        plt.xlabel('True:{}'.format(classes[y_true_25[i]]))
        plt.yticks([])
        plt.grid(False)
    plt.show()

    class_true = [classes[y_true[i]] for i in range(10000)]
    class_pred = [classes[y_pred[i]] for i in range(10000)]
    d = {'y_true': y_true, 'y_pred': y_pred, 'class_true': class_true, 'class_pred': class_pred}
    df = pd.DataFrame(data=d)
    error_count = df[df['class_true'] != df['class_pred']].groupby(['class_true', 'class_pred']).count()
    error_plot = pd.pivot_table(error_count.reset_index(), index='class_true', columns='class_pred', values='y_true')
    error_plot.plot.bar(figsize=(14,10))

    input('Press [enter] to exit.')

if __name__ == '__main__':
    main()
