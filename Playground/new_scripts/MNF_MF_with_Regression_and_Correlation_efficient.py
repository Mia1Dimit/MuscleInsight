# Trying to follow the part F of Muscle Signal Processing of the paper located in 
# https://github.com/Mia1Dimit/Thesis/blob/main/Bibliografia/Freq%20analysis%20of%20sEMG.pdf

import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib import pyplot as plt
from scipy.signal import welch
import json
import os


def select_files():
    """Opens a dialog window to select files and returns their paths as a list."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return list(file_paths)


def calculate_median_frequency(psd, freqs):
    """Calculate the Median Frequency (MF)."""
    cumulative_power = np.cumsum(psd)
    half_total_power = cumulative_power[-1] / 2
    mf = freqs[np.where(cumulative_power >= half_total_power)[0][0]]
    return mf


def calculate_mean_frequency(psd, freqs):
    """Calculate the Mean Frequency (MNF)."""
    mnf = np.sum(freqs * psd) / np.sum(psd)
    return mnf


def calculate_correlation_coefficient(x, y):
    """Calculate the correlation coefficient (R) using the formula from the reference."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    r = numerator / denominator
    return r


def process_signal(signal, segment_size, overlap):
    """Segment the signal and calculate MF and MNF for each segment."""
    step = segment_size - overlap
    mfs, mnfs = [], []
    for start in range(0, len(signal) - segment_size + 1, step):
        segment = signal[start:start + segment_size]
        freqs, psd = welch(segment, fs=1000)  # Assuming a sampling rate of 1000 Hz
        mf = calculate_median_frequency(psd, freqs)
        mnf = calculate_mean_frequency(psd, freqs)
        mfs.append(mf)
        mnfs.append(mnf)
    return mfs, mnfs


def linear_regression_and_correlation(x, y):
    """
    Perform linear regression to calculate slope and intercept,
    and calculate the correlation coefficient using the formula provided.
    """
    slope, intercept = np.polyfit(x, y, 1)  # Linear regression
    r = calculate_correlation_coefficient(x, y)  # Using provided formula for R
    return slope, intercept, r


def ask_save_plot():
    """Asks the user if they want to save the plot as a PNG file."""
    return messagebox.askyesno("Save Plot", "Do you want to save the plot as a PNG file?")


def generate_filename(file_details, metric, segment_size, overlap, folder_path):
    """Generates a descriptive filename and ensures the folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_names = "_".join([os.path.basename(detail['path']).split(".")[0] for detail in file_details])
    classes = "_".join(set([detail['class'] for detail in file_details]))
    filename = f"{metric}_Plot_{segment_size}Segment_{overlap}Overlap_{classes}.png"
    return os.path.join(folder_path, filename)


if __name__ == "__main__":
    # Select files
    file_paths = select_files()

    # Ask for parameters
    root = tk.Tk()
    root.withdraw()
    segment_size = simpledialog.askinteger("Input", "Enter segment size (number of samples):")
    overlap = simpledialog.askinteger("Input", "Enter overlap size (number of samples):")

    file_details = []
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Colors for plotting

    plt.figure(figsize=(12, 8))  # Prepare the figure for plotting
    for idx, path in enumerate(file_paths):
        with open(path, "r") as json_file:
            data = json.load(json_file)

        signal = data['signal']
        mfs, mnfs = process_signal(signal, segment_size, overlap)
        file_details.append({"path": path, "class": data["class"], "ID": data["ID"]})

        # Segment indices
        x = np.arange(len(mfs))

        # Perform Linear Regression and Correlation for MF
        mf_slope, mf_intercept, mf_r = linear_regression_and_correlation(x, mfs)
        mf_fit = mf_slope * x + mf_intercept

        # Perform Linear Regression and Correlation for MNF
        mnf_slope, mnf_intercept, mnf_r = linear_regression_and_correlation(x, mnfs)
        mnf_fit = mnf_slope * x + mnf_intercept

        # Plot MF and MNF
        plt.plot(x, mfs, label=f"MF {data['class']} (ID: {data['ID']})", linestyle='-', color=colors[idx % len(colors)])
        plt.plot(x, mf_fit, linestyle='--', color=colors[idx % len(colors)],
                 label=f"MF Fit: A={mf_slope:.2f}, R={mf_r:.2f}")

        plt.plot(x, mnfs, label=f"MNF {data['class']} (ID: {data['ID']})", linestyle='-', color=colors[(idx + 1) % len(colors)])
        plt.plot(x, mnf_fit, linestyle='--', color=colors[(idx + 1) % len(colors)],
                 label=f"MNF Fit: A={mnf_slope:.2f}, R={mnf_r:.2f}")

        # Print Results
        print(f"File: {path}")
        print(f"MF: Slope (A) = {mf_slope:.4f}, Intercept (b) = {mf_intercept:.4f}, Correlation (R) = {mf_r:.4f}")
        print(f"MNF: Slope (A) = {mnf_slope:.4f}, Intercept (b) = {mnf_intercept:.4f}, Correlation (R) = {mnf_r:.4f}")
        print()

    # Add labels, legend, and grid
    plt.xlabel('Segment Index')
    plt.ylabel('Frequency (Hz)')
    plt.title('Median and Mean Frequency with Linear Regression')
    plt.legend()
    plt.grid(True)

    # Ask if the user wants to save the plot
    if ask_save_plot():
        folder_path = r"C:\Dimitris\MuscleInsight\Plots\plots_frequency"
        filename = generate_filename(file_details, "Frequency", segment_size, overlap, folder_path)
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved at {filename}")
    else:
        print("Plot not saved.")

    # Show the plot
    plt.show()
