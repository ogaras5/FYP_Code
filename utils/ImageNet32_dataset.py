import os
import sys
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


_base_folder = 'imagenet-32-batches-py'
_train_list = ['train_data_batch_1',
               'train_data_batch_2',
               'train_data_batch_3',
               'train_data_batch_4',
               'train_data_batch_5',
               'train_data_batch_6',
               'train_data_batch_7',
               'train_data_batch_8',
               'train_data_batch_9',
               'train_data_batch_10']
_val_list = ['val_data']
_label_file = 'map_clsloc.txt'


class ImageNet32(Dataset):
    """`ImageNet32 <https://patrykchrabaszcz.github.io/Imagenet32/>`_ dataset.
    Warning: this will load the whole dataset into memory! Please ensure that
    4 GB of memory is available before loading.
    Refer to ``map_clsloc.txt`` for label information.
    The integer labels in this dataset are offset by -1 from ``map_clsloc.txt``
    to make indexing start from 0.
    Args:
        root (string): Root directory of dataset where directory
            ``imagenet-32-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        exclude (list, optional): List of class indices to omit from dataset.
        remap_labels (bool, optional): If True and exclude is not None, remaps
            remaining class labels so it is contiguous.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        # Now load the picked numpy arrays
        if self.train:
            self.data = []
            self.labels = []
            for f in _train_list:
                file = os.path.join(self.root, _base_folder, f)
                with open(file, 'rb') as fo:
                    if sys.version.split(".")[0] == "3":
                        entry = pickle.load(fo, encoding='latin1')
                    else:
                        entry = pickle.load(fo)
                    self.data.append(entry['data'])
                    self.labels += entry['labels']
            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((-1, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.labels = np.array(self.labels) - 1
        else:
            f = _val_list[0]
            file = os.path.join(self.root, _base_folder, f)
            with open(file, 'rb') as fo:
                if sys.version.split(".")[0] == "3":
                    entry = pickle.load(fo, encoding='latin1')
                else:
                    entry = pickle.load(fo)
                self.data = entry['data']
                self.labels = entry['labels']
            self.data = self.data.reshape((-1, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.labels = np.array(self.labels) - 1

        self.labels = self.labels.tolist()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self):
        return len(self.data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'val'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_imagenet32_labels(root):
    file = os.path.join(root, _base_folder, _label_file)
    return np.loadtxt(file, dtype=str)[:, 2].tolist()


def remap(old_array, mapping):
    new_array = np.copy(old_array)
    for k, v in mapping.items():
        new_array[old_array == k] = v
    return new_array
