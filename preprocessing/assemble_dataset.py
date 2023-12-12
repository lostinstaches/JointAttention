import os
import numpy as np

def load_npy_file(file_path):
    return np.load(file_path, allow_pickle=True)

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

# Base directory
base_dir = '../data/patients'

# Directory to save concatenated files
save_dir = '../data/dataset/'

# Ensure save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize containers for data
all_images = []
all_labels = []
all_gaze_data = []

# Traverse directories
for root, dirs, files in os.walk(base_dir):
    for subfolder in dirs:
        subfolder_path = os.path.join(root, subfolder)

        # print(subfolder_path)

        # Paths to files
        image_path = os.path.join(subfolder_path, 'IMAGE.npy')
        labels_path = os.path.join(subfolder_path, 'labels_ch.txt')
        gaze_path = os.path.join(subfolder_path, 'selected_gaze_ch.txt')

        # Check if files exist
        if os.path.exists(image_path) and os.path.exists(labels_path) and os.path.exists(gaze_path):
            images = load_npy_file(image_path)
            labels = read_txt_file(labels_path)
            gaze_data = read_txt_file(gaze_path)

            # Check if the number of entries are equal
            if len(images) == len(labels) == len(gaze_data):
                all_images.extend(images)
                all_labels.extend(labels)
                all_gaze_data.extend(gaze_data)


# Convert lists to arrays and save
np.save(os.path.join(save_dir, 'images.npy'), np.array(all_images), allow_pickle=True)
with open(os.path.join(save_dir, 'labels.csv'), 'w') as file:
    file.writelines(all_labels)
with open(os.path.join(save_dir, 'data.csv'), 'w') as file:
    file.writelines(all_gaze_data)

print("Data processing complete.")
