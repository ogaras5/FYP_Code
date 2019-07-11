# FYP_Code
All code relating to my final year project: "Exploring data augmentation strategies for deep learning".
Note trained models can be found in './trained_models' with a list of the accuracy achieved by the model for the CIFAR10 validation dataset available in './trained_models/modelAccuracies.md'.

## Main Training Scripts
A brief description of the main scripts utilised throughout the project to train models is given below. Each script is linked to a set of experiments carried out over the course of the project, which can be linked back to the FYP report. Amy scripts relating to trainng a resnet model usign the CIFAR-10/CIFAR-100 dataset, require the use of the custom model in './model/resnet_cifar.py'. All scripts use an absolute path to the dataset folder, which must be changed to allow use of the scripts based on User's dataset location.
### cifar_train.py
Script to train resnet model on CIFAR-10/CIFAR-100 dataset. This generates benchmark results and cannot be used to train with augmentations.

### imagenet_train.py
Script to train resnet model on 200 class ImageNet dataset. This generates benchmark results and cannot be used to train with augmentations.

### cifar_train_augmentation.py
Script to train resnet model on CIFAR-10/CIFAR-100 dataset. A single augmentation can be applied to the dataset which is one of the following: rotation, shear, skew, erase, distortion, and gaussianDistortion. 

### augmentation_validation.py
Script to validate on augmented validation datasets for CIFAR-10/CIFAR-100. Can specify the type of augmentation that is applied to the validation dataset and the type of augmentation applied to the model. The validation dataset must already exist (created using augmented_dataset_creator.py), and the model trained with the desired augmentation must exist.

### cifar_train_augmentation_pretrain.py
Script to train resnet model on CIFAR-10/CIFAR-100 dataset using pretrained model generated from cifar_train.py. A single augmentation can be applied to the dataset, which is one of the following: rotation, shear, skew, erase, distortion, gaussianDistortion. The benchmark checkpoint that training must be resumed from, must exist else training will not commence.

### pair_augmentation_cifar_train.py
Script to apply one/more augmentations to the CIFAR-10/CIFAR-100 dataset to train resnet model. The dataset is doubled in size, with all augmentations listed being applied with a probability of 1 to the original dataset.

### sampled_cifar_train.py
Script to train resnet model on sampled versions of CIFAR-10/CIFAR-100 dataset. Script can be used to generate benchmark model, or model trained using single augmentation dataset.

### multi_single_augmentation_train.py
Script to apply one/more augmentations to the CIFAR-10/CIFAR-100 dataset to train resnet model. Each augmentation listed is applied to the dataset individually, with the final training dataset being the concatenation of all augmented datasets created.

### imageNet32_train.py (Unused)
Script to train resnet model on ImageNet dataset with image resolution of 32x32 pixels. Script can be used to train the model using no augmentation, i.e. benchmark testing, or for single augmentations. The file uses the custom dataset loader ImageNet32 which is contained within './utils/ImageNet32_dataset.py'. Unfortunately this model was not trained due to difficulties arising with python3.7 on the GPU server.

## Dataset Creation Scripts
The scripts listed below allow for the creation of datasets based on the ImageNet and CIFAR-10/CIFAR-100 datasets. The scripts require the original datsets to be available to alloww for the creation of the altered datasets.

### buildDataset.py
Creates 200 class imagenet dataset using helper scripts available in './utils'. A json file containing the 200 classes in Tiny-ImageNet-200 and a json file containing the classes in ImageNet must be available for this script to work. 

### augmentation_dataset_creator.py
Creates validation dataset for CIFAR-10/CIFAR-100 dataset. Can apply a given augmentaiton to the created dataset.

### sample_cifar10_dataset_creator.py
Creates a smapled version of the CIFAR-10/CIFAR-100 training dataset. Must specify the number of images per class. The script stops creating the dataset once all classes have the specified number of images.

## Helper Methods in './utils'
### helpers.py
Contains the helper methods:
* AverageBase - Parent class for RunningAverage and MovingAverage
* RunningAverage - Keeps track of a cumulative moving average
* MovingAverage - Keeps track of an exponentially decaying moving average
* accuracy - Calculates the top-1 and top-5 accuracy
* load_checkpoint - Loads a saved checkpoint's optimizer and model for a given epoch
* save_checkpoint - Saves the optimizer and model for a given epoch
* plotLoss - Visualize the learning curve
* plotAccuracy - Visualize the accuracy learning curve
* plotLosses - Plot several training losses and validaation losses on seperate graphs

### data_helpers.py
Contains th helper methods:
* create_val_folder - Creates the correct file structure to use PyTorch ImageFolder to retrueve data for Tiny-ImageNet-200 dataset
* create_200_class_imagenet - Creates ImageNet dataset containing 200 classes instead of 1000 classes
* class_extractor - Creates a dictionary of the labels from the words.txt file in the Tiny-ImageNet-200 download. This file contains all labels for full ImageNet dataset, so want to return only those associated with Tiny-ImageNet-200.
* create_augmentation_pipeline - Creates an Augmentor Pipeline for the given augmentations
* add_augmentation - Adds the desired augmentation to the dataset pipeline

### progress.py
Creates a progress bar in the coomand line to show the current progress for processing the dataset for the current epoch.

