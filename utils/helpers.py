"""
Helper functions for visualization, saving,
and monitoring
"""
from __future__ import print_function, division

import torch
import os
import time

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
            self.value = flaot(value)
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
