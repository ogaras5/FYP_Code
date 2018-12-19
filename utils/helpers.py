"""
Helper functions for visualization, saving,
and monitoring
"""
from __future__ import absolute_import, print_function
import torch
import matplotlib.pyplot as plt

__all__ = ['AverageBase', 'RunningAverage', 'MovingAverage',
           'plotLoss', 'load_checkpoint', 'save_checkpoint',
           'accuracy']

class AverageBase(object):
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None

    def __str__(self):
        return str(round(self.value, 4))

    def __repr__(self):
        return self.value

    def __format__(self, fmt):
        return self.value.__format__(fmt)

    def __float__(self):
        return self.value

class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA)
    """
    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count

    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value

class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA)
    """
    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value

def load_checkpoint(optimizer, model, filename):
    """
    Function to load saved checkpoints
    """
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

def save_checkpoint(optimizer, model, epoch, filename):
    """
    Function to save checkpoints while Training
    """
    checkpoint_dict = {
        'optimizer' : optimizer.state_dict(),
        'model' : model.state_dict(),
        'epoch' : epoch
    }
    torch.save(checkpoint_dict, filename)

def accuracy(output, target, topk=(1,)):
    """
    Computes the prescision k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def plotLoss(train_losses,valid_losses):
    # Visualize the Learning Curve
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_losses, '-o', label='Training loss')
    plt.plot(epochs, valid_losses, '-o', label='Validation loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.show()
