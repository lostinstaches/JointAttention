import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Load data from files
input_data = pd.read_csv('../data/input.csv', sep=" ", header=None)
images = np.load('../data/images.npy')
labels = pd.read_csv('../data/labels.csv')

print("Shape of input data is:" + str(input_data.shape))
print("Shape of images data is:" + str(images.shape))
print("Shape of labels data is:" + str(labels.shape))

# Verify that all datasets have the same number of entries
assert len(input_data) == len(images) == len(labels), "Datasets must have the same number of entries"

# Splitting parameters: Hyperparameters
test_size = 0.2  # 20% for testing
val_size = 0.1  # 10% for validation (of the remaining 80%)

# Split into train+val and test sets
train_val_indices, test_indices = train_test_split(np.arange(len(input_data)), test_size=test_size, random_state=42)

# Split train+val into train and val sets
train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size/(1-test_size), random_state=42)

# Function to extract subsets based on indices
def extract_subset(indices, input_df, image_arr, label_df):
    return input_df.iloc[indices], image_arr[indices], label_df.iloc[indices]

# Extract subsets
train_input, train_images, train_labels = extract_subset(train_indices, input_data, images, labels)
val_input, val_images, val_labels = extract_subset(val_indices, input_data, images, labels)
test_input, test_images, test_labels = extract_subset(test_indices, input_data, images, labels)

# Save subsets to pickle files
def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

save_to_pickle((train_input, train_images, train_labels), '../data/pickles/train.pkl')
save_to_pickle((val_input, val_images, val_labels), '../data/pickles/val.pkl')
save_to_pickle((test_input, test_images, test_labels), '../data/pickles/test.pkl')