import numpy as np
import os

# Parent directory containing all patient folders
parent_directory = '/Users/lostinstaches/Desktop/Patients'

# List to store arrays from each IMAGE.npy file
all_arrays = []


# Iterate through each subdirectory in the parent directory
for patient_folder in os.listdir(parent_directory):
    patient_path = os.path.join(parent_directory, patient_folder)

    # Skip if it's not a directory
    if not os.path.isdir(patient_path):
        continue

    # Path to the IMAGE.npy file
    npy_file_path = os.path.join(patient_path, 'IMAGE.npy')

    # Check if IMAGE.npy exists in the folder
    if os.path.exists(npy_file_path):
        # Load the array and add it to the list
        array = np.load(npy_file_path)
        all_arrays.append(array)
    else:
        print(f"IMAGE.npy not found in {patient_folder}")

# Concatenate all arrays if there are any
if all_arrays:
    concatenated_array = np.concatenate(all_arrays, axis=0)

    # Path for the output concatenated file
    output_file = os.path.join(parent_directory, 'Concatenated_IMAGE.npy')

    # Save the concatenated array to a new .npy file
    np.save(output_file, concatenated_array)

    print(f"Concatenated array saved to {output_file}")
else:
    print("No arrays to concatenate.")