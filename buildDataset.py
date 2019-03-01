import numpy as np
import os
import shutil
import json
import argparse

from utils.data_helpers import *

parser = argparse.ArgumentParser("Create 200 class ImageNet")
parser.add_argument('-d', '--destination', default='/data/sarah/200class-imagenet-256', type=str,
            help='Destination file to store new dataset')
parser.add_argument('-s', '--source', default='/data/sarah/imagenet-256', type=str,
            help='Source file of full imagenet dataset')
args = parser.parse_args()

def main():
    print("-----Make Dataset-----")
    if not os.path.exists(args.destination):
        print("Creating destination folder for new dataset")
        os.makedirs(args.destination)
    create_200_class_imagenet(args.source, args.destination)
    print("Complete")

if __name__ == '__main__':
    main()
