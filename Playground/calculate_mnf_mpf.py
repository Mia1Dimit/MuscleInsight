import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

def calculate_linear_regression_slope(amplitudes):
    sampling_freq = 800

    # Read amplitudes after 6000 samples
    start_index = 0
    amplitudes = amplitudes[start_index:]

    # Calculate Mean Frequency (MNF) and Median Power Frequency (MPF) using a moving window of 800 samples with a step of 400 samples
    mean_frequencies = []
    median_power_frequencies = []

    for i in range(0, len(amplitudes) - 800, 400):
        subset = amplitudes[i:i+800]
        fft_values = fft(subset)
        freq = fftfreq(len(subset), 1 / sampling_freq)
        
        # Select only positive frequencies and corresponding FFT values
        positive_freq_mask = (freq >= 0) & (freq <= 350)
        positive_freq = freq[positive_freq_mask]
        positive_fft_values = np.abs(fft_values)[positive_freq_mask]
        
        # Calculate mean frequency
        mean_frequency = np.sum(positive_freq * np.abs(positive_fft_values)**2) / np.sum(np.abs(positive_fft_values)**2)
        mean_frequencies.append(mean_frequency)
        
        # Calculate median power frequency
        power_spectrum = np.abs(positive_fft_values)**2
        total_power = np.sum(power_spectrum)
        normalized_cumulative_power = np.cumsum(power_spectrum) / total_power
        mpf_index = np.argmax(normalized_cumulative_power >= 0.5)
        median_power_frequencies.append(positive_freq[mpf_index])

    # Calculate linear regression for Mean Frequency (MNF) and Median Power Frequency (MPF)
    mean_frequencies_linear_fit = np.polyfit(range(len(mean_frequencies)), mean_frequencies, 1)
    median_power_frequencies_linear_fit = np.polyfit(range(len(median_power_frequencies)), median_power_frequencies, 1)

    # Extract slopes
    mean_frequencies_slope = mean_frequencies_linear_fit[0]
    median_power_frequencies_slope = median_power_frequencies_linear_fit[0]

    return mean_frequencies_slope, median_power_frequencies_slope

csv.field_size_limit(100000000)
folder_path = r'C:\Users\DeLL\Documents\Arduino\thesis_BioAmp\Thesis-2\final_data'
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

for file_path in file_paths:
    data = {}
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            array_name = row[0]
            array_values = list(map(float, row[1].split(',')))
            data[array_name] = array_values

    amplitudes = data["amplitudes"]

    mean_frequencies_slope, median_power_frequencies_slope = calculate_linear_regression_slope(amplitudes)

    print("File:", file_path)
    print("Slope of Mean Frequency (MNF):", mean_frequencies_slope)
    print("Slope of Median Power Frequency (MPF):", median_power_frequencies_slope)
    print()
