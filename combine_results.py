import pandas as pd
import argparse

# Arguments for pulling results
parser = argparse.ArgumentParser(description='Script to concatenate results for augmentation validation')
parser.add_argument('-a', '--augmentation', type=str, nargs='+', default=['rotation'], help='Augmentation models to pull results for')
parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='Dataset to pull results for')

args = parser.parse_args()

# Types of augmentations applied to the dataset
augmentations = ['benchmark', 'rotation', 'shear', 'skew', 'erase', 'distortion', 'gaussianDistortion']

# Pandas array to hold results for all dataset and models
final_csv = pd.DataFrame()

# Loop through each augmentation supplied for model 
for j, augmentation in enumerate(args.augmentation):
    result = []
    # Loop thorugh each augmentation of the dataset
    for i, augment in enumerate(augmentations):
        df = pd.read_csv('./losses/validate-{}Model-{}-{}.csv'.format(augmentation, augment, args.dataset))
        df.set_index('model_augmentation', inplace=True)
	if i == 0:
            result = df
        else:
            result = pd.concat([result, df], axis=1, sort=False)
    final_csv = final_csv.append(result)

print(final_csv)
final_csv.to_csv('./losses/validation-results-{}.csv'.format(args.dataset)) 
