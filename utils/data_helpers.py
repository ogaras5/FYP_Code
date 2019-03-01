"""
Functions to help with the loading and labelling of tiny-imagenet-200
Also includes any functions relating to the loading of data
"""
import os
import shutil
import json
import Augmentor

__all__ = ['create_val_folder', 'class_extractor', 'create_200_class_imagenet',
           'create_augmentation_pipeline']

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
            old_valid_path = os.path.join(valid_path, str(classes['index']))
            old_train_path = os.path.join(train_path, str(classes['index']))
            new_valid_path = os.path.join(new_path, 'val/' + str(classes['index']))
            new_train_path = os.path.join(new_path, 'train/' + str(classes['index']))
            if not os.path.exists(new_valid_path):
                os.makedirs(new_valid_path)
            for img in os.listdir(old_valid_path):
                if not os.path.exists(os.path.join(new_valid_path, img)):
                    shutil.copy(os.path.join(old_valid_path, img), new_valid_path)
            if not os.path.exists(new_train_path):
                os.makedirs(new_train_path)
            for img in os.listdir(old_train_path):
                if not os.path.exists(os.path.join(new_train_path, img)):
                    shutil.copy(os.path.join(old_train_path, img), new_train_path)

    meta_path = os.path.join(new_path, 'meta')
    if not os.path.exists(meta_path):
        os.makedirs(meta_path)
    meta_path = os.path.join(meta_path, 'categories.json')
    with open(meta_path, 'w') as fp:
        json.dump(new_categories, fp, indent=4)

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

def create_augmentation_pipeline(augmentations, prob=1, max_right=25,
                                max_left=25, mag=0.5):
    """
    Create a pipeline to add given augmentation with given parameters
    """
    # Create empty pipeline
    p = Augmentor.Pipeline()
    # Check if single augmentation or list of augmentation passed to function
    if isinstance(augmentations, (list, tuple)):
        for augment in augmentations:
            add_augmentation(p, augment, prob=prob, max_right=max_right,
                            max_left=max_left, mag=mag)
    else:
        add_augmentation(p, augmentations, prob=prob, max_right=max_right,
                        max_left=max_left, mag=mag)
    return p

def add_augmentation(p, augment, **kwargs):
    """
    Function to add a given augmentation to the pipeline with the passed arguments
    """
    # Grab list of optional arguments for augmentations
    args = {k: v for k, v in kwargs.items()}
    # List of all allowed augmentations
    possible_augments = ['rotation', 'erase', 'skew', 'shear', 'distortion',
                        'gaussianDistortion']
    # Filter out any numbers from the augmentation name passed to ensure it can
    # be found
    augment = ''.join(filter(lambda x: x.isalpha(), augment))
    if augment in possible_augments:
        if augment == 'rotation':
            print('Adding rotation to pipeline')
            p.rotate(probability=args['prob'], max_left_rotation=args['max_left'],
                    max_right_rotation=args['max_right'])
        if augment == 'skew':
            print('Adding skew tilt to pipeline')
            p.skew_tilt(probability=args['prob'], magnitude=args['mag'])
        if augment == 'shear':
            print('Adding shear to pipeline')
            p.shear(probability=args['prob'], max_shear_left=args['max_left'],
                    max_shear_right=args['max_right'])
        if augment == 'distortion':
            print('Adding distortion to pipeline')
            p.random_distortion(probability=args['prob'], grid_width=6, grid_height=6, magnitude=5)
        if augment == 'gaussianDistortion':
            print('Adding gaussian distortion to pipeline')
            p.gaussian_distortion(probability=args['prob'], grid_width=6, grid_height=6, magnitude=5,
                                corner="bell", method="in", mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)
        if augment == 'erase':
            print('Adding random erasing to pipeline')
            p.random_erasing(probability=args['prob'], rectangle_area=args['mag'])
    else:
        raise ValueError('The augmentation {} is invalid. Possible augmentations\
 are (augmentations can be followed by digits):\n{}'
        .format(augment, possible_augments))
    return p
