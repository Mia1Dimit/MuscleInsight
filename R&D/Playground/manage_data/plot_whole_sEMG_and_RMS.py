import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

def calculate_rms_series(amplitudes, window_size):
    return [
        np.sqrt(np.mean(np.square(amplitudes[i:i + window_size])))
        for i in range(0, len(amplitudes), window_size)
    ]

def select_and_plot_json():
    # Create a file dialog to select a JSON file
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

    if not file_path:
        print("No file selected. Exiting...")
        return

    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract data from the JSON file
    try:
        amplitudes = data["amplitudes"]
        sampling_rate = 800  # Hz
        time = np.arange(len(amplitudes)) / sampling_rate

        # Plot the amplitude data
        plt.figure(figsize=(10, 5))
        plt.plot(time, amplitudes, label=f"Person: {data.get('person', 'Unknown')} | ID: {data.get('ID', 'N/A')}")
        plt.title("Amplitude vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate RMS series and plot it
        window_size = 200  # Number of samples per window
        rms_series = calculate_rms_series(amplitudes, window_size)
        rms_time = np.arange(len(rms_series)) * (window_size / sampling_rate)

        plt.figure(figsize=(10, 5))
        plt.plot(rms_time, rms_series, label="RMS Values", color="orange")
        plt.title("RMS vs. Time")
        plt.xlabel("Time (s)")
        plt.ylabel("RMS Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

    except KeyError as e:
        print(f"Missing key in JSON file: {e}")

if __name__ == "__main__":
    select_and_plot_json()
