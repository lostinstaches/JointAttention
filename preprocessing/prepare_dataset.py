import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF

def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def normalize_input_data(input_data):
    # Exclude the last 5 columns (ID and 4 boolean flags)
    numerical_data = input_data.iloc[:, :-5]
    # Calculate mean and std only for numerical data
    mean = numerical_data.mean(axis=0)
    std = numerical_data.std(axis=0)
    # Normalize the numerical data
    normalized_data = (numerical_data - mean) / std
    # Combine the normalized data with the excluded columns
    normalized_input_data = pd.concat([normalized_data, input_data.iloc[:, -5:]], axis=1)
    return normalized_input_data

color_jitter_transform = transforms.ColorJitter(
    brightness=0.5,  # Example values, adjust as needed
    contrast=0.5,
    saturation=0.5,
    hue=0.2
)

def add_noise_to_image(image, noise_level=0.04):
    # print("Original image statistics:")
    # print(f"  Min: {image.min()}, Max: {image.max()}, Mean: {image.mean()}")
    #
    # print("im shape", image.shape)
    #
    # # Assuming image pixel values range from 0 to 255
    # noise = np.random.normal(0, noise_level, image.shape)
    # print("noise shape", noise.shape)
    # noisy_image = image + noise
    # print("nois im shape", noisy_image.shape)
    # noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values stay within [0, 255]
    #
    # print(image[0,0,0])
    # print(noise[0, 0, 0])
    # print(noisy_image[0, 0, 0])
    # print("Noisy image statistics:")
    # print(f"  Min: {noisy_image.min()}, Max: {noisy_image.max()}, Mean: {noisy_image.mean()}")
    #
    # return noisy_image
    noise = np.random.normal(0, noise_level * 255, image.shape)  # Scale the noise level
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip to valid range
    return noisy_image.astype(np.uint8)


def apply_color_jitter(image, base=0.1, variance=0.1):
    # Randomly choose the parameters within the specified range
    brightness = np.random.uniform(base - variance, base + variance)
    contrast = np.random.uniform(base - variance, base + variance)
    saturation = np.random.uniform(base - variance, base + variance)

    # Ensure hue stays within the [0, 0.5] range
    hue = np.random.uniform(0, min(base + variance, 0.5))

    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image.astype(np.uint8))

    # Define the color jitter transform with random parameters
    transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    # Apply the transform
    jittered_image = transform(image)

    # Convert back to NumPy array
    jittered_image = np.array(jittered_image)

    return jittered_image

def add_jitter_to_input(input_datum, jitter_level=0.1):
    # Assuming input_datum is now a numpy array
    if input_datum.size == 0:
        raise ValueError("Empty input_datum provided to add_jitter_to_input function.")

    # Apply jitter to all elements except the last one (ID)
    jittered_data = input_datum[:-1] + np.random.uniform(-jitter_level, jitter_level, input_datum[:-1].shape)
    # Append the ID back without change
    jittered_with_id = np.concatenate([jittered_data, [input_datum[-1]]])
    return jittered_with_id

# Helper function to reconstruct datasets from groups and drop ID column
def reconstruct_data(groups, images, labels):
    indices = np.concatenate([group.index.values for _, group in groups])
    subset_input_data = input_data.loc[indices].drop(columns='ID')  # Drop the ID column
    return subset_input_data, images[indices], labels.loc[indices]

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

input_data = normalize_input_data(input_data)

# Print label distribution for Train dataset
print("Label Distribution before processing:\n", labels.iloc[:, 0].value_counts())

# Augmentation
underrepresented_classes = [1, 4, 3, 5]
augmented_images = []
augmented_input_data = []
augmented_labels = []

target_count_per_class = 600
current_count_per_class = {label: 0 for label in underrepresented_classes}

for label in labels.iloc[:, 0]:
    if label in underrepresented_classes:
        current_count_per_class[label] += 1

augmentation_counter = 0

# Assuming underrepresented_classes, target_count_per_class, and augmented_count_per_class are defined correctly
print("Underrepresented Classes:", underrepresented_classes)
print("Target Count per Class:", target_count_per_class)
print("Augmented Count per Class:", current_count_per_class)

while any(current_count_per_class[label] < target_count_per_class for label in underrepresented_classes):
    for image, (index, input_datum), label_value in zip(images, input_data.iterrows(), labels.iloc[:, 0]):
        if label_value in underrepresented_classes and current_count_per_class[label_value] < target_count_per_class:
            # Choose augmentation type
            if augmentation_counter % 2 == 0:
                augmented_image = add_noise_to_image(image)
            else:
                augmented_image = apply_color_jitter(image)
            # augmented_image = apply_color_jitter(image)
            # augmented_image = add_noise_to_image(image)
            jittered_input = add_jitter_to_input(input_datum.values)

            # Append augmented data
            augmented_images.append(augmented_image)
            augmented_input_data.append(jittered_input)
            augmented_labels.append(label_value)

            # Update counts
            current_count_per_class[label_value] += 1
            augmentation_counter += 1

            # Break the loop if the target is reached for this label
            if current_count_per_class[label_value] >= target_count_per_class:
                break

# Convert augmented data lists to numpy arrays
augmented_images = np.array(augmented_images)
augmented_input_data_df = pd.DataFrame(augmented_input_data)
augmented_labels_df = pd.DataFrame(augmented_labels)

# Append augmented data to original data
images = np.concatenate((images, augmented_images), axis=0)
images = images.astype(np.float32) / 255.0
input_data = pd.concat([input_data, augmented_input_data_df], ignore_index=True)
labels = pd.concat([labels, augmented_labels_df], ignore_index=True)


display_image(images[-880])

# Get the unique IDS
unique_ids = input_data.iloc[:, -1].unique()

# Splitting parameters: Hyperparameters
test_size = 0.1  # 20% for testing
val_size = 0.1  # 10% for validation (of the remaining 80%)

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

# Save subsets to pickle files
def save_to_pickle(data, filename):
    with open(os.path.join(pickle_directory, filename), 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    print(os.path.join(pickle_directory, filename))
    with open(os.path.join(pickle_directory, filename), 'rb') as file:
        return pickle.load(file)


save_to_pickle((train_input, train_images, train_labels), 'train.pkl')
save_to_pickle((val_input, val_images, val_labels), 'val.pkl')
save_to_pickle((test_input, test_images, test_labels), 'test.pkl')

# Print label distribution for Train dataset
print("Loaded Label Distribution:\n", labels.iloc[:, 0].value_counts())

# print("Data processing complete.")
#
# # Print label distribution for Train dataset
# print("Train Label Distribution:\n", train_labels.iloc[:, 0].value_counts())
#
# # Print label distribution for Validation dataset
# print("Validation Label Distribution:\n", val_labels.iloc[:, 0].value_counts())
#
# # Print label distribution for Test dataset
# print("Test Label Distribution:\n", test_labels.iloc[:, 0].value_counts())



