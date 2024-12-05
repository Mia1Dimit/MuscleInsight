import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import json
import itertools


def select_files():
    """Opens a dialog window to select files and returns their paths as a list."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return list(file_paths)


if __name__ == "__main__":
    # File path to the CSV file
    file_paths = select_files()
    all_lists = []
    for path in file_paths:
        with open(path, "r") as json_file:
            data = json.load(json_file)
        all_lists.append((path, data['signal']))  # Include file path and signal data

    long_1D_list = [item for _, sublist in all_lists for item in sublist]
    x = np.arange(len(long_1D_list))
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])  # Cycles through colors
    start_idx = 0

    # Plot each array in a different color
    for file_path, arr in all_lists:
        end_idx = start_idx + len(arr)
        print(f"File: {file_path}, Length of array being plotted: {len(arr)}, corresponds to samples: {len(arr)/800}")  # Log the file name and length
        plt.plot(x[start_idx:end_idx], arr, color=next(colors))
        start_idx = end_idx

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Rest')

    # Show the plot
    plt.show()
