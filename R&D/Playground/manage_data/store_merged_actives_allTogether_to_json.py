import json
import tkinter as tk
from tkinter import filedialog

output_path = "C:\\Users\\jstivaros\\Documents\\MuscleInsight\\MuscleInsight\\Data_Acquisition\\merged_active_segments\\"

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select JSON files", filetypes=[("JSON files", "*.json")])
    return file_paths

def merge_json_files(file_paths):
    merged_data = None
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            if merged_data is None:
                merged_data = data.copy()
                merged_data['signal'] = data['signal']
            else:
                merged_data['signal'].extend(data['signal'])
    
    return merged_data

def save_merged_json(merged_data):
    if merged_data:
        output_filename = merged_data['person']+".json"
        with open(output_path+output_filename, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Merged data saved to {output_filename}")
    else:
        print("No data to save.")

def main():
    while(1):
        file_paths = select_files()
        if not file_paths:
            print("No files selected.")
            return
        
        merged_data = merge_json_files(file_paths)
        save_merged_json(merged_data)
    
if __name__ == "__main__":
    main()
