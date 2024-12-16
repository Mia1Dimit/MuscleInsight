import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib import pyplot as plt
import json
import itertools
import os
from matplotlib.lines import Line2D


def select_files():
    """Opens a dialog window to select files and returns their paths as a list."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return list(file_paths)


def calculate_rms(signal, window_size, overlap):
    """
    Calculates RMS for a signal with a specified window size and overlap.

    :param signal: List or array of signal values.
    :param window_size: Number of samples in each window.
    :param overlap: Number of overlapping samples between consecutive windows.
    :return: List of RMS values.
    """
    step = window_size - overlap
    rms_values = []
    for start in range(0, len(signal) - window_size + 1, step):
        segment = signal[start:start + window_size]
        rms = np.sqrt(np.mean(np.square(segment)))
        rms_values.append(rms)
    return rms_values


def ask_save_plot():
    """Asks the user if they want to save the plot as a PNG file."""
    return messagebox.askyesno("Save Plot", "Do you want to save the plot as a PNG file?")


def on_pick(event):
    legend_line = event.artist
    is_dimmed = legend_line.get_alpha() == 0.2  # Check if the legend line is dimmed
    new_alpha = 1.0 if is_dimmed else 0.2  # Toggle between dimmed and fully visible
    legend_line.set_alpha(new_alpha)  # Set the alpha for the legend line
    
    # Toggle the visibility of the corresponding plot line
    for line in plt.gca().get_lines():
        if line.get_label() == legend_line.get_label():
            line.set_visible(is_dimmed)  # Show if it was dimmed, hide if it was visible
    
    plt.gcf().canvas.draw_idle()  # Redraw the canvas


def generate_filename(file_details, metric, window_size, overlap, folder_path):
    """
    Generates a descriptive filename for the plot and ensures the folder exists.

    :param file_details: List of dictionaries with details about the selected files.
    :param metric: The metric used (e.g., RMS).
    :param window_size: Window size used in the calculation.
    :param overlap: Overlap size used in the calculation.
    :param folder_path: Path to the folder where the file will be saved.
    :return: Full path to the file with the descriptive filename.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist

    file_names = "_".join([os.path.basename(detail['path']).split(".")[0] for detail in file_details])
    classes = "_".join(set([detail['class'] for detail in file_details]))
    filename = f"{metric}_Plot_{window_size}Window_{overlap}Overlap_{classes}.png"
    return os.path.join(folder_path, filename)


if __name__ == "__main__":
    # Select files
    file_paths = select_files()
    
    # Ask for RMS calculation parameters
    root = tk.Tk()
    root.withdraw()
    window_size = simpledialog.askinteger("Input", "Enter window size (number of samples):")
    overlap = simpledialog.askinteger("Input", "Enter overlap (number of samples):")
    
    all_rms_results = []
    file_details = []
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])  # Colors for plotting

    plt.figure(figsize=(10, 6))  # Prepare the figure for plotting
    for path in file_paths:
        with open(path, "r") as json_file:
            data = json.load(json_file)
        
        signal = data['signal']
        rms_values = calculate_rms(signal, window_size, overlap)
        all_rms_results.append((path, rms_values))
        file_details.append({"path": path, "class": data["class"], "ID": data["ID"]})
        
        # Plot the RMS values
        x = np.arange(len(rms_values))  # Index for RMS values
        plt.plot(x, rms_values, label=f"{data['person']} {data['ID']} {data['class']}", color=next(colors))
    
    # Add labels and legend
    plt.xlabel('Segment Index')
    plt.ylabel('RMS Value')
    plt.title('RMS Calculation')
    legend = plt.legend()
    plt.grid(True)

    # Ask if the user wants to save the plot
    # if ask_save_plot():
    #     folder_path = r"C:\Dimitris\MuscleInsight\Plots\plots_rms"
    #     filename = generate_filename(file_details, "RMS", window_size, overlap, folder_path)
    #     plt.savefig(filename, bbox_inches='tight')  # Save the figure correctly
    #     print(f"Plot saved at {filename}")
    # else:
    #     print("Plot not saved.")
    
    for legend_line, original_line in zip(legend.get_lines(), plt.gca().get_lines()):
        legend_line.set_picker(True)  # Enable picking on legend lines
        legend_line.set_pickradius(5)  # Set a pick radius for easier clicking

   
    plt.gcf().canvas.mpl_connect("pick_event", on_pick)

    # Show the plot after saving
    plt.show()

