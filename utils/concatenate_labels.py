import os
import pandas as pd

# Parent directory containing all patient folders
parent_directory = '/Users/lostinstaches/Desktop/Patients'

# Initialize an empty DataFrame to store all data
merged_data = pd.DataFrame()

# Iterate through each subdirectory in the parent directory
for patient_folder in os.listdir(parent_directory):
    patient_path = os.path.join(parent_directory, patient_folder)

    # Skip if it's not a directory
    if not os.path.isdir(patient_path):
        continue

    # Find a file that contains 'selected_gaze' in the filename
    gaze_files = [f for f in os.listdir(patient_path) if 'labels' in f]
    for gaze_file in gaze_files:
        gaze_file_path = os.path.join(patient_path, gaze_file)

        # Read the .txt file into a DataFrame
        data = pd.read_csv(gaze_file_path, sep=' ', header=None)

        # Append the data to the merged DataFrame
        merged_data = merged_data._append(data, ignore_index=True)

# Save the merged data to a CSV file
output_csv_file = os.path.join(parent_directory, 'merged_labels.csv')
merged_data.to_csv(output_csv_file, index=False, header=False, sep=' ')

print(f"Merged data saved to {output_csv_file}")
