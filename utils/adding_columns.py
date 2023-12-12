import os
import random

def process_gaze_file(file_path, flags, experiment_id):
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify each line by appending flags and experiment ID
    modified_lines = [line.strip() + ' ' + ' '.join(map(str, flags)) + ' ' + str(experiment_id) + '\n' for line in lines]

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)


def update_gaze_file_with_new_id(file_path):
    # Generate a new random ID for this file
    new_id = random.randint(1000, 9999)

    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify each line by replacing the last number with the new ID
    modified_lines = [' '.join(line.strip().split(' ')[:-1]) + ' ' + str(new_id) + '\n' for line in lines]

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

    return new_id

# Base directory
base_dir = '../data/patients'

# Random experiment ID
experiment_id = random.randint(1000, 99999)

# Folder to flags mapping
folder_flags = {
    'robot_patient': [1, 0, 0, 0],
    'robot_therapist': [0, 1, 0, 0],
    'no_robot_patient': [0, 0, 1, 0],
    'no_robot_therapist': [0, 0, 0, 1]
}

# Traverse directories
for folder in folder_flags:
    folder_path = os.path.join(base_dir, folder)

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        if os.path.isdir(subfolder_path):
            gaze_files = [f for f in os.listdir(subfolder_path) if 'selected_gaze' in f]

            for gaze_file in gaze_files:
                gaze_file_path = os.path.join(subfolder_path, gaze_file)
                new_id = update_gaze_file_with_new_id(gaze_file_path)
                print(f"Process completed with experiment ID: {new_id}")
                # process_gaze_file(gaze_file_path, folder_flags[folder], experiment_id)

# print(f"Process completed with experiment ID: {experiment_id}")
