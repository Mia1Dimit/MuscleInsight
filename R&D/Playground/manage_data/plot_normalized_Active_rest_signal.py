import tkinter as tk
from tkinter import filedialog
import json
import numpy as np
import matplotlib.pyplot as plt

def load_json_file():
    """Open file dialog to select a JSON file"""
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select JSON file")
    return filepath

def process_signal_files():
    # Load rest file first
    print("Select REST signal file:")
    rest_filepath = load_json_file()
   
    # Load active file
    print("Select ACTIVE signal file:")
    active_filepath = load_json_file()
   
    # Read rest signal
    with open(rest_filepath, 'r') as rest_file:
        rest_data = json.load(rest_file)
   
    # Read active signal
    with open(active_filepath, 'r') as active_file:
        active_data = json.load(active_file)
   
    # Calculate mean of rest signal
    rest_signal = np.array(rest_data['signal'])
    mean_rest = np.mean(rest_signal)
   
    # Normalize active signal by dividing by mean rest
    active_signal = np.array(active_data['signal'])
    normalized_signal = active_signal // mean_rest
    
    num_samples_active = len(active_signal)
    time_active = np.arange(num_samples_active) / 800  # Time array for plotting

    num_samples = len(rest_signal)
    time_rest = np.arange(num_samples) / 800  # Time array for plotting
   
    # Plot rest, active, and normalized signals
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_rest, rest_signal)
    plt.title('Rest Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_active, active_signal)
    plt.title('Active Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_active, normalized_signal)
    plt.title('Active Signal Normalized by Mean Rest Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
   
    # Print statistics
    print(f"Mean of Rest Signal: {mean_rest}")
    print(f"Original Active Signal Mean: {np.mean(active_signal)}")
    print(f"Normalized Active Signal Mean: {np.mean(normalized_signal)}")

if __name__ == "__main__":
    process_signal_files()