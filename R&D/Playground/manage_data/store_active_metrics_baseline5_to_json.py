import numpy as np
from scipy.fft import fftfreq
import tkinter as tk
from tkinter import filedialog
import json
from scipy.signal import welch, hilbert, detrend
from scipy.stats import entropy
from math import log2
from PyEMD import EMD, EEMD
import matplotlib.pyplot as plt



#############################
# Configuration

output_path = "C:\\Users\\jstivaros\\Documents\\MuscleInsight\\MuscleInsight\\Data_Acquisition\\merged_active_segments\\metrics\\"

INITIAL_RATE = 800.0
window_size = 800
step_size = 200

#############################


def main():

    filepaths = select_files()

    for filepath in filepaths:
        with open(filepath, "r") as json_file:
            newdata = json.load(json_file)
            data = newdata["signal"]

        Fs              = INITIAL_RATE
        input_signal    = np.array(data)
        num_samples     = len(input_signal)
        time            = np.arange(num_samples) / Fs  # Time array for plotting

        frequencies = fftfreq(window_size, 1 / Fs)[:window_size // 2]
        index_25 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 25))
        index_80 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 80))
        index_350 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 350))

        m = {
            'rms': [],
            "mnf_arv_ratio": [],
            "ima_diff": [],
            "emd_mdf1": [],
            "emd_mdf2": [],
            "fluct_variance": [],
            "fluct_range_values": [],
            "fluct_mean_diff_values": []
        }

        baseline = np.sqrt(np.mean(input_signal[0:window_size]**2))

        
        for idx in range(0,len(input_signal)-window_size,step_size):
            segment = input_signal[idx:idx+window_size] / baseline
            
            m["rms"].append(            np.sqrt(np.mean(segment**2)))

            m["mnf_arv_ratio"].append(  calc_mnf_arv_ratio(segment, Fs))
            
            m["ima_diff"].append(       calc_ima_diff(segment, index_25, index_80, index_350))
            
            mdf1,mdf2           =       calc_emd_mdf1_2(segment, Fs)
            m["emd_mdf1"].append(mdf1)
            m["emd_mdf2"].append(mdf2)
            
            sc_flct =                   calc_scaled_fluct_metrics(segment)
            m["fluct_variance"].append(sc_flct[0])
            m["fluct_range_values"].append(sc_flct[1])
            m["fluct_mean_diff_values"].append(sc_flct[2])

        
        m['person'] = newdata['person']
        store_to_json(m)

    
    
    



#########################################################

def store_to_json(merged_data):
    if merged_data:
        output_filename = merged_data['person']+"_metrics.json"
        with open(output_path+output_filename, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Merged data saved to {output_filename}")
    else:
        print("No data to save.")

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select JSON files", filetypes=[("JSON files", "*.json")])
    return file_paths


def calc_emd_mdf1_2(signal, sampling_rate):
    emd = EMD()
    emd.FIXE_H = 5  # Maximum number of sifting iterations
    emd.spline_kind = 'cubic'  # Interpolation type for envelope computation
    imfs_emd = emd(signal)    

    mdf_1 = calculate_median_frequency(imfs_emd[0], sampling_rate)
    mdf_2 = calculate_median_frequency(imfs_emd[1], sampling_rate)
    return mdf_1, mdf_2

def calc_scaled_fluct_metrics(signal):
    processed_segment = calculate_scaled_fluctuations(signal)
    variance = calculate_signal_variance(processed_segment)
    range_values = calculate_signal_range(processed_segment)
    mean_diff_values = calculate_mean_differences(processed_segment)
    entropy_values = calculate_signal_entropy(processed_segment)
    return variance, range_values, mean_diff_values, entropy_values

def calculate_scaled_fluctuations(signal, scales=[5, 10, 20, 40]):
    result = np.zeros_like(signal, dtype=np.float64)
    
    for scale in scales:
        segments = len(signal) // scale
        fluctuations = []
        
        for i in range(segments):
            segment = signal[i * scale:(i + 1) * scale]
            trend_removed = detrend(segment, type='linear')
            fluctuation = np.mean(trend_removed**2)
            fluctuations.append(fluctuation)
        
        rescaled_fluctuations = np.repeat(fluctuations, scale)
        if len(rescaled_fluctuations) < len(signal):
            padding = np.zeros(len(signal) - len(rescaled_fluctuations))
            rescaled_fluctuations = np.concatenate([rescaled_fluctuations, padding])
        
        rescaled_fluctuations = rescaled_fluctuations[:len(signal)]
        result += rescaled_fluctuations
    
    return result / len(scales)

def calculate_signal_variance(segment):
    """Calculate variance of the signal segment"""
    return np.var(segment)

def calculate_signal_range(segment):
    """Calculate range of the signal segment"""
    return np.max(segment) - np.min(segment)

def calculate_mean_differences(segment):
    """Calculate mean absolute differences between consecutive points"""
    return np.mean(np.abs(np.diff(segment)))

def calculate_signal_entropy(segment):
    """Calculate Shannon entropy of the signal segment distribution"""
    hist, _ = np.histogram(segment, bins=10, density=True)
    return entropy(hist, base=2)

def calc_mnf_arv_ratio(segment, sampling_rate):
    mnf = calculate_mnf(segment, sampling_rate)
    arv = np.mean(np.abs(segment))
    return mnf/arv

def calculate_median_frequency(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]
    mf = freqs[np.searchsorted(cumulative_energy, total_energy / 2)]
    return mf

def calculate_mnf(signal, sampling_rate):
    """
    Calculate the Mean Frequency (MNF) of a signal.

    Parameters:
        signal (np.ndarray): The input signal (time-domain).
        sampling_rate (float): The sampling rate of the signal in Hz.

    Returns:
        float: The mean frequency of the signal.
    """
    # Compute the Power Spectral Density (PSD) using FFT
    freqs = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    fft_vals = np.fft.rfft(signal)
    psd = np.abs(fft_vals) ** 2

    # Calculate MNF (frequency-weighted average of the power spectrum)
    mnf = np.sum(freqs * psd) / np.sum(psd)
    return mnf

def segment_signal(signal, window_size, step_size):
    segments = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

def get_fft_values(signal):
    """
    Computes the FFT of a given signal window and returns the values from 0 Hz to fs/2.

    Args:
        signal (np.ndarray): The input signal window.
        fs (float): The sampling frequency of the signal.

    Returns:
        np.ndarray: The FFT values from 0 Hz to fs/2.
    """
    n = len(signal)
    fft_values = np.fft.fft(signal)

    # Take only the positive frequencies
    positive_fft_values = (2.0 / n) * np.abs(fft_values[:n//2])

    return positive_fft_values

def calc_ima_diff(segment, index_25, index_80, index_350):
    # Perform FFT and get magnitude (assuming m.get_fft_values exists)
    fft_magnitudes = get_fft_values(segment)
    
    # IMA diff
    return np.mean(fft_magnitudes[index_25:index_80]) - np.mean(fft_magnitudes[index_80:index_350])


if  __name__ == "__main__":
    main()