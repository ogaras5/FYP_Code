from __future__ import print_function
import sys

__all__ = ['MonitorProgress']

class MonitorProgress(object):
    """
    Function to show the progress of the training epoch.
    Displays the current loss, number of images trained and a progress bar.
    Please ensure that the cmd window length is wider than the progress bar for
    correct functionality of overwriting progress bar
    @params:
    batch     - current training batch (images already trained)
    total     - total training images
    prefix    - prefix string (use for Loss)
    suffix    - suffix string
    decimals  - number of decimal places in percentage progress
    length    - length of bar
    fill      - character to fill bar with
    """
    def __init__(self, total, suffix='', decimals=1, length=50, fill='#'):
        self.total = total
        self.batch = 0
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.update(self.batch, 0)

    def update(self, batch, loss):
        self.batch += batch
        self.prefix = 'Loss: {:.04f}'.format(loss)
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.batch / float(self.total)))
        filled_length = int(self.length * self.batch // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print('{} |{}| {}% {}'.format(self.prefix, bar, percent, self.suffix),
              end='\r'),

        # Print New Line when complete
        if self.batch == self.total:
            print()

        sys.stdout.flush()
