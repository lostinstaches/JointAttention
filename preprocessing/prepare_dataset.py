import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime

# Helper function to reconstruct datasets from groups and drop ID column
def reconstruct_data(groups, images, labels):
    indices = np.concatenate([group.index.values for _, group in groups])
    subset_input_data = input_data.loc[indices].drop(columns='ID')  # Drop the ID column
    return subset_input_data, images[indices], labels.loc[indices]

# Function to filter data based on IDs
# def filter_data_by_ids(ids, input_df, image_arr, label_df):
#     indices = input_df[input_df['ID'].isin(ids)].index
#     return input_df.loc[indices].drop(columns='ID'), image_arr[indices], label_df.loc[indices] # New function

# Function to filter data based on IDs and then drop the ID column
def filter_data_by_ids(ids, input_df, image_arr, label_df):
    indices = input_df[input_df.iloc[:, -1].isin(ids)].index
    filtered_input = input_df.loc[indices].drop(input_df.columns[-1], axis=1)  # Drop the last column (ID)
    return filtered_input, image_arr[indices], label_df.loc[indices]

# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a new directory for the timestamped data
pickle_directory = f'../data/pickles/{timestamp}'
os.makedirs(pickle_directory, exist_ok=True)

# Load data from files
input_data = pd.read_csv('../data/dataset/data.csv', sep=" ", header=None)
images = np.load('../data/dataset/images.npy')
labels = pd.read_csv('../data/dataset/labels.csv', sep=" ", header=None)

print("Shape of input data is:" + str(input_data.shape))
print("Shape of images data is:" + str(images.shape))
print("Shape of labels data is:" + str(labels.shape))

# Identify entries with label -1 and remove these entires
invalid_label_indices = labels[labels.iloc[:, 0] == -1].index
input_data = input_data.drop(invalid_label_indices).reset_index(drop=True)
images = np.delete(images, invalid_label_indices, axis=0)
labels = labels.drop(invalid_label_indices).reset_index(drop=True)

print("After removing wrong indices")
print("Shape of input data is:" + str(input_data.shape))
print("Shape of images data is:" + str(images.shape))
print("Shape of labels data is:" + str(labels.shape))

# Get the unique IDS
unique_ids = input_data.iloc[:, -1].unique()

# Splitting parameters: Hyperparameters
test_size = 0.2  # 20% for testing
val_size = 0.2  # 10% for validation (of the remaining 80%)

# Split unique IDs into train+val and test sets
train_val_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=42) # Modified line

# Further split train+val IDs into train and val sets
train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size/(1-test_size), random_state=42) # Modified line

# Filter datasets based on IDs
train_input, train_images, train_labels = filter_data_by_ids(train_ids, input_data, images, labels)
val_input, val_images, val_labels = filter_data_by_ids(val_ids, input_data, images, labels)
test_input, test_images, test_labels = filter_data_by_ids(test_ids, input_data, images, labels)

# Verify that all datasets have the same number of entries
assert len(train_input) == len(train_images) == len(train_labels), "Train datasets must have the same number of entries"
assert len(val_input) == len(val_images) == len(val_labels), "Validation datasets must have the same number of entries"
assert len(test_input) == len(test_images) == len(test_labels), "Test datasets must have the same number of entries"

print('test')
print(train_input.head())

# Save subsets to pickle files
def save_to_pickle(data, filename):
    with open(os.path.join(pickle_directory, filename), 'wb') as file:
        pickle.dump(data, file)

save_to_pickle((train_input, train_images, train_labels), 'train.pkl')
save_to_pickle((val_input, val_images, val_labels), 'val.pkl')
save_to_pickle((test_input, test_images, test_labels), 'test.pkl')

print("Data processing complete.")