import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import json
from matplotlib.widgets import SpanSelector, Button
import os


def select_files():
    """Opens a dialog window to select files and returns their paths as a list."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return list(file_paths)

class InteractiveSignalPlot:
    def __init__(self, dataDict):
        """
        Initializes the interactive plot.

        Parameters:
            data (dict): The signal data struct to plot.
        """
        
        self.person = dataDict['person']
        self.ID = dataDict['ID']
        self.amplitudes = dataDict['rms_values']
        self.title = dataDict['person']+'_ID'+dataDict['ID']
        self.selected_region = None  # To store the selected region
        self.cl4ss = "activation"
        
        # Initialize the plot
        self.fig, self.ax = plt.subplots(figsize=(16, 8))  # Adjust width and height as needed
        plt.subplots_adjust(bottom=0.2)  # Make space for buttons

        self.ax.set_title(self.title)
        self.line, = self.ax.plot(self.amplitudes, label=self.ID)
        self.ax.legend()

        # Initialize SpanSelector for selecting horizontal region
        self.selector = SpanSelector(self.ax, self.on_select, 'horizontal', props={'facecolor': 'blue', 'alpha': 0.3})

        # Add Save Button
        self.save_ax = plt.axes([0.7, 0.05, 0.1, 0.075])  # [left, bottom, width, height]
        self.save_button = Button(self.save_ax, 'Save Segment')
        self.save_button.on_clicked(self.save_segment)

    def on_select(self, xmin, xmax):
        """
        Callback for SpanSelector. Stores the selected region.

        Parameters:
            xmin (float): The starting x-coordinate of the selection.
            xmax (float): The ending x-coordinate of the selection.
        """
        self.selected_region = (xmin, xmax)
        print(f"Selected region: {self.selected_region}")

    def save_segment(self, event):
        """
        Callback for Save Button. Saves the selected signal segment as a JSON file.

        Parameters:
            event: The event that triggered the callback.
        """
        if self.selected_region is None:
            print("No region selected. Please select a region before saving.")
            return

        # Extract the selected segment
        xmin, xmax = self.selected_region
        # Assuming 'amplitudes' corresponds to x-axis indices
        start_idx = int(np.clip(xmin, 0, len(self.amplitudes)-1))
        end_idx = int(np.clip(xmax, 0, len(self.amplitudes)-1))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx  # Swap to ensure start <= end
        segment = self.amplitudes[start_idx:end_idx+1]

        # Create a dictionary to save
        data_to_save = {
            "person": self.person,
            "ID": self.ID,
            "class": self.cl4ss,
            "signal": segment
        }

        # Open a dialog to choose the folder
        folder_selected = self.choose_folder()
        if not folder_selected:
            print("No folder selected. Save operation canceled.")
            return

        # Define the file name
        file_name = f"{self.cl4ss}_{self.person}_ID{self.ID}.json"
        full_path = os.path.join(folder_selected, file_name)

        # Save the dictionary as JSON
        try:
            with open(full_path, 'w') as json_file:
                json.dump(data_to_save, json_file, indent=4)
            print(f"Segment saved as JSON at: {full_path}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
        plt.close(self.fig)

    def choose_folder(self):
        """
        Opens a dialog to choose a folder.

        Returns:
            str: The path to the selected folder, or None if canceled.
        """
        return "C:\\Dimitris\\MuscleInsight\\Data_Acquisition\\class_segments_rms"
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        folder_selected = filedialog.askdirectory(title="Select Folder to Save JSON")
        root.destroy()
        if folder_selected:
            return folder_selected
        else:
            return None

    def show(self):
        """
        Displays the interactive plot.
        """
        plt.show()



if __name__ == "__main__":
    
    # File path to the CSV file
    file_paths = select_files()
    
    for path in file_paths:
        
        with open(path, "r") as json_file:
            data = json.load(json_file)
            
        interactive_plot = InteractiveSignalPlot(dataDict=data)
        interactive_plot.show()

        


