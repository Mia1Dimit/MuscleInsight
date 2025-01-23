import pandas as pd
import json

# Load JSON data from file
input_file = "C:\Dimitris\MuscleInsight\Data_Acquisition\Baseline_search_analysis_output.json"  # Replace with your file path
output_file = "C:\Dimitris\MuscleInsight\Data_Acquisition\Baseline_Search_analysis.xlsx"    # Desired Excel output file name

with open(input_file, "r") as file:
    data = json.load(file)

# Prepare data for the Excel file
rows = []

for entry in data['files']:
    participant_id = entry.get('participant_id')
    phase = entry.get('phase')
    metrics = entry.get('metrics', {})
    
    for metric, values in metrics.items():
        if isinstance(values, dict):  # Check if the value is a dictionary
            row = {
                "File Path": entry.get('file_path'),
                "Participant ID": participant_id,
                "Phase": phase,
                "Metric": metric,
                "Mean": values.get('mean'),
                "Std Dev": values.get('std'),
                "80th Percentile": values.get('percentile_80'),
                "Max": values.get('max'),
                "Min": values.get('min'),
                "Range": values.get('range'),
                "Percent Below 80th": values.get('percent_below_80th'),
            }
            rows.append(row)

# Create DataFrame and save to Excel
df = pd.DataFrame(rows)
df.to_excel(output_file, index=False)

print(f"Data successfully saved to {output_file}")
