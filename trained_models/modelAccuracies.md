# Accuracy for each Model
Below is a list of the accuracies achieved by each of the available pretrained models. All accuracy were 
found for the CIFAR-10 Validation dataset.
## Single Augmentations
| Augmentation        | CIFAR10 Accuracy | CIFAR10 200 Sample Accuracy | CIFAR10 1000 Sample Accuracy |
| ------------------- |:----------------:|:---------------------------:|:----------------------------:|
| Benchmark           | 93.59 %          | 66.54 %                     | 86.04 %                      |  
| Rotation            | 94.66 %          | 71.21 %                     | 84.85 %                      |
| Random Erase        | 95.00 %          | 74.46 %                     | 87.44 %                      |
| Skew                | 94.15 %          | 69.95 %                     | 87.14 %                      |
| Shear               | 94.85 %          | 70.66 %                     | 86.54 %                      |
| Random Distortion   | 93.19 %          | 64.73 %                     | 84.03 %                      |
| Gaussian Distortion | 93.54 %          | 69.18 %                     | 86.46 %                      |

## Pair Augmentations
| Augmentation            | CIFAR10 Accuracy | CIFAR10 200 Sample Accuracy |
| ----------------------- |:----------------:|:---------------------------:|
| Rotation - Random Erase | 94.97 %          | 71.39 %                     |
| Random Erase - Rotation | 95.19 %          | 71.75 %                     |
| Skew - Random Erase     | 95.69 %          | 75.36 %                     |
| Random Erase - Skew     | 94.91 %          | 72.07 %                     |
| Shear - Random Erase    | 94.63 %          | 73.01 %                     |
| Random Erase - Shear    | 95.34 %          | 73.25 %                     |

## Multi-Single Augmentations
| Augmentation                           | CIFAR10 Accuracy | CIFAR10 200 Sample Accuracy |
| -------------------------------------- |:----------------:|:---------------------------:|
| Random Erase - Rotation - Skew - Shear | 95.86 %          | 76.09 %                     |

## Augmentation Introduced at Epoch 30
| Augmentation        | CIFAR10 Accuracy |
| ------------------- |:----------------:|
| Skew                | 94.97 %          |
| Shear               | 94.87 %          |
| Random Distortion   | 93.71 %          |
| Gaussian Distortion | 94.30 %          |
