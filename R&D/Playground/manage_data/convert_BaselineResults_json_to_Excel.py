import os
import json
import pandas as pd
import re

def extract_statistical_data(json_file_path):
    """
    Extract statistical data in the specified format.
    
    Args:
        json_file_path (str): Path to the JSON file
    
    Returns:
        list: List of dictionaries with extracted statistical data
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract participant ID from filename
    participant_match = re.search(r'([\w-]+)\.json', os.path.basename(json_file_path))
    participant_id = participant_match.group(1) if participant_match else 'Unknown'
    
    # Extract phase from path (assuming it's the directory name before the filename)
    phase_match = re.search(r'class_segments/(\w+)/', json_file_path)
    phase = phase_match.group(1) if phase_match else 'Unknown'
    
    # Prepare output data
    output_data = []
    
    for metric, stats in data['statistical_analysis'].items():
        # Skip metrics with all null values
        if all(value is None for value in stats.values()):
            continue
        
        # Prepare a row for each metric
        row = {
            'File Path': json_file_path,
            'Participant ID': participant_id,
            'Phase': phase,
            'Metric': metric,
            'Mean': round(stats.get('mean', 0), 4) if stats.get('mean') is not None else 0,
            'Std Dev': round(stats.get('std', 0), 4) if stats.get('std') is not None else 0,
            '80th Percentile': round(stats.get('percentile_80', 0), 4) if stats.get('percentile_80') is not None else 0,
            'Max': round(stats.get('max', 0), 4) if stats.get('max') is not None else 0,
            'Min': round(stats.get('min', 0), 4) if stats.get('min') is not None else 0,
            'Range': round(stats.get('range', 0), 4) if stats.get('range') is not None else 0,
            'Percent Below 80th': round(stats.get('percent_below_80th', 0), 4) if stats.get('percent_below_80th') is not None else 0
        }
        output_data.append(row)
    
    return output_data

def convert_json_files_to_excel(input_folder, output_file):
    """
    Convert all JSON files in a folder to an Excel spreadsheet.
    
    Args:
        input_folder (str): Path to the folder containing JSON files
        output_file (str): Path for the output Excel file
    """
    # Find all JSON files recursively
    json_files = []
    for root, dirs, files in os.walk(input_folder):
        json_files.extend([os.path.join(root, f) for f in files if f.endswith('.json')])
    
    # Extract data from all files
    all_data = []
    for file in json_files:
        try:
            file_data = extract_statistical_data(file)
            all_data.extend(file_data)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Data exported to {output_file}")

# Example usage
if __name__ == "__main__":
    input_folder = r"C:\\Dimitris\\MuscleInsight\\Data_Acquisition\\"  
    output_file = r"C:\\Dimitris\\MuscleInsight\\Data_Acquisition\\.xlsx"  
    
    convert_json_files_to_excel(input_folder, output_file)