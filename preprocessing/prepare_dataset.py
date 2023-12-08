import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime

# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a new directory for the timestamped data
pickle_directory = f'../data/pickles/{timestamp}'
os.makedirs(pickle_directory, exist_ok=True)

# Load data from files
input_data = pd.read_csv('../data/input.csv', sep=" ", header=None)
images = np.load('../data/images.npy')
labels = pd.read_csv('../data/labels.csv')

print("Shape of input data is:" + str(input_data.shape))
print("Shape of images data is:" + str(images.shape))
print("Shape of labels data is:" + str(labels.shape))

# Identify entries with label -1
invalid_label_indices = labels[labels.iloc[:, 0] == -1].index

# Remove these entries from input_data, images, and labels
input_data = input_data.drop(invalid_label_indices).reset_index(drop=True)
images = np.delete(images, invalid_label_indices, axis=0)
labels = labels.drop(invalid_label_indices).reset_index(drop=True)

print("Shape of input data is:" + str(input_data.shape))
print("Shape of images data is:" + str(images.shape))
print("Shape of labels data is:" + str(labels.shape))

print(labels)

# Count the number of entries for each label
label_counts = labels.iloc[:, 0].value_counts()
print("Label counts:")
print(label_counts)

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
    with open(os.path.join(pickle_directory, filename), 'wb') as file:
        pickle.dump(data, file)

save_to_pickle((train_input, train_images, train_labels), 'train.pkl')
save_to_pickle((val_input, val_images, val_labels), 'val.pkl')
save_to_pickle((test_input, test_images, test_labels), 'test.pkl')