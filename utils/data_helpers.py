"""
Functions to help with the loading and labelling of tiny-imagenet-200
"""
import os
import shutil
import json

__all__ = ['create_val_folder', 'class_extractor', 'create_200_class_imagenet']

def create_val_folder(data_path):
    """
    Creates the correct file structure to use pyTorch ImageFolder to retrieve data
    """
    path = os.path.join(data_path, 'val/images')
    filename = os.path.join(data_path, 'val/val_annotations.txt')
    fp = open(filename, 'r')
    data = fp.readlines()

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if it is not already present and move image into the folder
    for img, folder in val_img_dict.items():
        newpath = os.path.join(path, folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path,img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))

def create_200_class_imagenet(data_path, new_path):
    """
    Creates a version of ImageNet containing the 200 classes in tiny-imagenet-200
    """
    valid_path = os.path.join(data_path, 'val')
    train_path = os.path.join(data_path, 'train')
    categories = os.path.join(data_path, 'meta/categories.json')
    classes = os.path.join(data_path, 'meta/tiny_class.json')

    # Load json dictionary for tiny-imagenet classes and imagenet classes
    with open(categories, 'r') as fp:
        all_classes = json.load(fp)
    with open(classes, 'r') as fp:
        used_classes = json.load(fp)

    new_categories = []
    for classes in all_classes:
        if classes['id'] in used_classes:
            print('Id: {} in list!'.format(classes['id']))
            new_categories.append(classes)
            old_valid_path = os.path.join(valid_path, classes['index'])
            old_train_path = os.path.join(train_path, classes['index'])
            new_valid_path = os.path.join(new_path, 'val/' + classes['index'])
            new_train_path = os.path.join(new_path, 'train/' + classes['index'])
            if not os.path.exists(new_valid_path):
                os.makedirs(new_valid_path)
            with open(old_valid_path) as fp:
                for img in os.listdir(fp):
                    if not os.path.exists(os.join.path(new_valid_path), img):
                        shutil.copy(os.join.path(old_valid_path, img), os.join.path(new_valid_path, img))
            if not os.path.exists(new_train_path):
                os.makedirs(new_train_path)
            with open(old_train_path) as fp:
                for img in os.listdir(fp):
                    if not os.path.exists(os.join.path(new_train_path), img):
                        shutil.copy(os.join.path(old_train_path, img), os.join.path(new_train_path, img))

    meta = os.path.join(new_path, 'meta/categories.json')
    with open(meta, 'w') as fp:
        json.dump(new_categories, fp, sort_keys=True, indent=4)

def class_extractor(class_list, data_path):
    """
    Create a dictionary of the labels from the words.txt. This file contains
    all labels for full ImageNet dataset, so want to return only those associated
    with tiny ImageNet
    """
    filename = os.path.join(data_path, 'words.txt')
    fp = open(filename, 'r')
    data = fp.readlines()

    # Create a dictionary with numerical class names as keys and label string as values
    large_class_dict = {}
    for line in data:
        words = line.split('\t')
        super_label = words[1].split(',')
        large_class_dict[words[0]] = super_label[0].rstrip()
    fp.close()

    # Create a small dictionary with only 200 classes for Tiny ImageNet
    tiny_class_dict = {}
    for small_label in class_list:
        for k, v in large_class_dict.items():
            if small_label == k:
                tiny_class_dict[k] = v
                continue

    return tiny_class_dict
