import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import json


csv.field_size_limit(100000000)

def select_files():
    """Opens a dialog window to select files and returns their paths as a list."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return list(file_paths)

def import_csv_mhts(file_paths):
    dicts = []
    
    for path in file_paths:
        data = {}
        # Read the CSV file
        filename = path.split('/')[-1]
        data['person'] = filename.split('_')[0]
        data['ID'] = filename.split('.')[-2][-1]
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            i = 0
            for row in csv_reader:
                if i==0:
                    i+=1
                    continue
                # Extract key and values, and convert values to a NumPy array of floats
                key = row[0]
                values = [float(x) for x in row[1].strip('"').split(",")]
                data[key] = values
        dicts.append(data)
    for k in data.keys():
        print(k)
    return dicts



def save_dict_as_json(dictionary, file_path, file_name):
    """
    Saves a dictionary as a JSON file at the specified path with the given name.
    
    Parameters:
        dictionary (dict): The dictionary to save.
        file_path (str): The path to save the JSON file.
        file_name (str): The name of the JSON file (with or without `.json` extension).
    """
    # Ensure the file name ends with .json
    if not file_name.endswith(".json"):
        file_name += ".json"
    
    # Create the full file path
    full_path = file_path+file_name
    
    # Write the dictionary to the JSON file
    with open(full_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)
    print(f"Dictionary saved as JSON at: {full_path}")


if __name__ == "__main__":
    pathhhh = "C:\\Users\\John Stivaros\\Documents\\PersonalProjects\\MuscleInsight\\raw_data\\"
    # File path to the CSV file
    file_paths = select_files()
    data_dicts = import_csv_mhts(file_paths)
    
    for dict in data_dicts:
        save_dict_as_json(dict, pathhhh, dict['person']+'_id'+dict['ID'])
        


