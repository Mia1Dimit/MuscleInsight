import os
import csv
import pandas as pd

csv.field_size_limit(100000000)

def process_csv_files(folder_path, sampling_frequency=1000):
    # Dictionary to hold file names as rows and ArrayIDs as columns
    data = {}
    durations = {}  # Dictionary to hold the duration for each file

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            data[file_name] = {}  # Initialize row for this file
            
            total_samples = 0  # Initialize total samples for the file
            
            with open(file_path, 'r', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                
                for row in reader:
                    array_id = row["ArrayID"]
                    array = row["Array"]
                    array_length = len(array.strip('"').split(","))
                    data[file_name][array_id] = array_length
                    total_samples += array_length  # Accumulate the total samples

            # Calculate duration for the file
            durations[file_name] = total_samples / sampling_frequency

    # Convert to DataFrame for matrix representation
    df = pd.DataFrame.from_dict(data, orient='index').fillna(0).astype(int)
    df.index.name = "Filename"

    # Add the durations as a separate column
    df['Duration (seconds)'] = pd.Series(durations)

    return df

# Folder containing the CSV files
folder_path = r'C:\Dimitris\Thesis\final_data'

# Process the files and display the resulting matrix with durations
result_matrix = process_csv_files(folder_path)
print(result_matrix)

# Save the matrix to a CSV file (optional)
result_matrix.to_csv("output_matrix_with_durations.csv")
