"""
Simple Convolutional Neuraal Network test for MNIST dataset
Program is to test the functionality of helper methods utilised in
cifar model
"""
import sys
print(sys.version)
device = 'cpu'

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import time
import argparse

from utils.progress import MonitorProgress
from utils.helpers import *

parser = argparse.ArgumentParser(description='PyTorch MNIST training')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='Number of epochs to train')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='Epoch to start training on (must have checkpoint saved)')
args = parser.parse_args()

print('PyTorch version: ', torch.__version__)

torch.manual_seed(271828)
np.random.seed(271728)

class SimpleCNN(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(14*14*32, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = self.pool1(X)
        X = self.drop1(X)
        X = X.reshape(-1, 14*14*32)
        X = F.relu(self.fc1(X))
        X = self.drop2(X)
        X = self.fc2(X)
        return X

#transforms for the training set
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

# Use same transform for validation set
valid_transform = train_transform

# Load the datasets, downloading if needed
train_set = MNIST('./data/mnist', train=True, download=True,
                    transform=train_transform)
valid_set = MNIST('./data/mnist', train=False, download=True,
                    transform=valid_transform)

print(train_set.train_data.shape)
print(valid_set.test_data.shape)

plt.figure(figsize=(10,10))

sample = train_set.train_data[:64]
# shape (64, 28, 28)
sample = sample.reshape(8, 8, 28, 28)
# shape (8, 8, 28, 28)
sample = sample.permute(0, 2, 1, 3)
# shape (8, 28, 8, 28)
sample = sample.reshape(8*28, 8*28)
# shape (8*28, 8*28)
plt.imshow(sample)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.title('First 64 MNIST digits in training set')
plt.show()

print('Labels:', train_set.train_labels[:64].numpy())

train_loader = DataLoader(train_set, batch_size=256, num_workers=0, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=512, num_workers=0, shuffle=False)

# Create model
model = SimpleCNN()
model.to(device)

# Stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# Model training
def train(optimizer, model, num_epochs=10, first_epoch=1):
    since = time.time()

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    best_acc = 0.0

    for epoch in range(first_epoch, first_epoch + num_epochs):
        print('Epoch', epoch)

        # train phase
        model.train()

        # create a progress bar
        progress = MonitorProgress(total=len(train_set))

        # save training data
        train_loss = MovingAverage()

        for batch, targets in train_loader:
            # Move the training data to the GPU
            batch = batch.to(device)
            targets = targets.to(device)

            # clear previous gradient computation
            optimizer.zero_grad()

            # forward propagation
            predictions = model(batch)

            # calculate the loss
            loss = criterion(predictions, targets)

            # backpropagate to compute gradients
            loss.backward()

            # update model weights
            optimizer.step()

            # update average loss
            train_loss.update(loss)

            # update progress bar
            progress.update(batch.shape[0], train_loss)

        print('Training loss:', train_loss)
        train_losses.append(train_loss.value)

        # validation phase
        model.eval()

        valid_loss = RunningAverage()

        # keep track of predictions
        y_pred = []

        # We don't need gradients for validation, so wrap in
        # no_grad to save memory
        with torch.no_grad():

            for batch, targets in valid_loader:
                # Move the training batch to the GPU
                batch = batch.to(device)
                targets = targets.to(device)

                # forward propagation
                predictions = model(batch)

                # calculate the loss
                loss = criterion(predictions, targets)

                # update running loss value
                valid_loss.update(loss)

                # save predictions
                y_pred.extend(predictions.argmax(dim=1).cpu().numpy())

        print('Validation loss:', valid_loss)
        valid_losses.append(valid_loss.value)

        # Calculate validation accuracy
        y_pred = torch.tensor(y_pred, dtype=torch.int64)
        accuracy = torch.mean((y_pred == valid_set.test_labels).float())
        print('Validation accuracy: {:.4f}%'.format(float(accuracy) * 100))
        if accuracy > best_acc:
            best_acc = accuracy

        # Save a checkpoint
        checkpoint_filename = './checkpoints/mnist-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    print('Best value Accuracy: {:4f}%'.format(float(best_acc)*100))

    return train_losses, valid_losses, y_pred

if args.start_epoch != 1:
    epoch = load_checkpoint(optimizer, model, './checkpoints/mnist-{:03d}.pkl'
            .format(args.start_epoch))
    print('Resuming training from epoch', epoch)
train_losses, valid_losses, y_pred = train(optimizer, model, num_epochs=args.epochs,
                                            first_epoch=args.start_epoch)

# Save taining loss, and validation loss to a csv
df = pd.DataFrame({
    'epoch': range(1, len(train_losses) + 1),
    'train': train_losses,
    'valid': valid_losses
})

# Save to csv file
df.to_csv("./losses/MNIST.csv")

plotLoss(train_losses, valid_losses)
num_errors = torch.sum((y_pred != valid_set.test_labels).float())
print('Validation errors {} (out of {})'.format(int(num_errors), len(valid_set)))

# pull out examples of mistakes in the valid set
error_indicator = y_pred != valid_set.test_labels
error_examples = valid_set.test_data[error_indicator, :, :]

plt.figure(figsize=(10,10))

sample = error_examples[:64]
# shape (64, 28, 28)
sample = sample.reshape(8,8,28,28)
# shape (8, 8, 28, 28)
sample = sample.permute(0,2,1,3)
# shape (8, 28, 8, 28)
sample = sample.reshape(8*28,8*28)
# shape (8*28, 8*28)
plt.imshow(sample)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.title('Error analysis (validation set)')
plt.show()

print('y_true:', valid_set.test_labels[error_indicator][:64].numpy())
print('y_pred:', y_pred[error_indicator][:64].numpy())
