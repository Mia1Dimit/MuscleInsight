"""
Opens a dialog for selecting multiple json files which contain the signal data.
Merges the signal data from all the selected files and plots it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq
import tkinter as tk
from tkinter import filedialog
import json
from scipy.signal import welch, hilbert, detrend
from scipy.stats import entropy
from math import log2
from PyEMD import EMD, EEMD
import time
import psutil
import os
from scipy import signal

#############################
# Configuration

INITIAL_RATE = 800.0

# Define multiple window sizes and step sizes to test
WINDOW_SIZES = [1600]  # in samples
STEP_SIZES_RATIOS = [1/4]   # as fraction of window size

# Directory to save plots
SAVE_DIR = r"C:\\Dimitris\\MuscleInsight\\R&D\\Plots\\plots_Window_Analysis_Results\\Active_marios_ID1"
os.makedirs(SAVE_DIR, exist_ok=True)

#############################


def main():

    filepaths = open_dialog_and_select_multiple_files()
    data = []

    for filepath in filepaths:
        with open(filepath, "r") as json_file:
            newdata = json.load(json_file)
            data.extend(newdata["signal"])

    Fs              = INITIAL_RATE
    input_signal    = np.array(data)
    num_samples     = len(input_signal)
    time_array      = np.arange(num_samples) / Fs  # Time array for plotting

    all_results = []

    for window_size in WINDOW_SIZES:
        for step_ratio in STEP_SIZES_RATIOS:
            step_size = int(window_size * step_ratio)
            metrics, variance_stats, max_min_stats, max_diff_stats, system_metrics = process_with_parameters(input_signal, window_size, step_size)
            
            result = {
                'window_size': window_size,
                'step_size': step_size,
                'variance_stats': variance_stats,
                'max_min_stats': max_min_stats,
                'max_diff_stats': max_diff_stats,
                'system_metrics': system_metrics.get_average_metrics()
            }
            all_results.append(result)

    # Print analysis results for all parameter combinations
    print("\nParameter Analysis Results:")
    print("=" * 50)
    metrics_list = ['mnf', 'ima', 'emd', 'fluct']

    for result in all_results:
        print(f"\nWindow size: {result['window_size']} samples ({result['window_size']/INITIAL_RATE:.3f} s)")
        print(f"Step size: {result['step_size']} samples ({result['step_size']/INITIAL_RATE:.3f} s)")
        
        # Print variance stats
        print(f"MNF/ARV Variance: {result['variance_stats']['mnf_arv_ratio']:.6f}")
        print(f"IMA Variance: {result['variance_stats']['ima_diff']:.6f}")
        print(f"EMD Variance: {result['variance_stats']['emd']:.6f}")
        print(f"Fluct Variance: {result['variance_stats']['fluct']:.6f}")
        
        # Print max-min stats
        print(f"MNF/ARV Max-Min: {result['max_min_stats']['mnf_arv_ratio']:.6f}")
        print(f"IMA Max-Min: {result['max_min_stats']['ima_diff']:.6f}")
        print(f"EMD Max-Min: {result['max_min_stats']['emd']:.6f}")
        print(f"Fluct Max-Min: {result['max_min_stats']['fluct']:.6f}")
        
        # Print max diff stats
        print(f"MNF/ARV Max Differential: {result['max_diff_stats']['mnf_arv_ratio']:.6f}")
        print(f"IMA Max Differential: {result['max_diff_stats']['ima_diff']:.6f}")
        print(f"EMD Max Differential: {result['max_diff_stats']['emd']:.6f}")
        print(f"Fluct Max Differential: {result['max_diff_stats']['fluct']:.6f}")
        
        # Print system metrics
        print(f"Computation time: {result['system_metrics']['computation_time']:.6f} s")
        print(f"Memory Usage: {result['system_metrics']['avg_memory_mb']:.2f} MB")

        

def process_with_parameters(input_signal, window_size, step_size):
    system_metrics = SystemMetrics()
    
    num_windows = len(range(0, len(input_signal) - window_size, step_size))
    time_points = np.arange(num_windows) * (step_size / INITIAL_RATE)
    
    frequencies = fftfreq(window_size, 1 / INITIAL_RATE)[:window_size // 2]
    index_25 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 25))
    index_80 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 80))
    index_350 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 350))
    
    m = {
        "mnf_arv_ratio": [],
        "ima_diff": [],
        "emd_mdf1": [],
        "emd_mdf2": [],
        "fluct_variance": [],
        "fluct_range_values": [],
        "fluct_mean_diff_values": [],
    }
    
    # Store differentials for max diff calculation
    mnf_arv_diffs = []
    ima_diffs = []
    emd_diffs = []
    fluct_diffs = []
    
    # Isolated computation time and memory tracking
    start_memory = SystemMetrics.calculate_memory_usage()
    
    for idx in range(0, len(input_signal)-window_size, step_size):
        segment = input_signal[idx:idx+window_size]
        
        # MNF/ARV calculation
        start_time = time.time()
        mnf_value = calc_mnf_arv_ratio(segment, INITIAL_RATE)[0]
        mnf_computation_time = time.time() - start_time
        
        # IMA calculation
        start_time = time.time()
        ima_value = calc_ima_diff(segment, index_25, index_80, index_350)[0]
        ima_computation_time = time.time() - start_time
        
        # EMD calculation
        start_time = time.time()
        emd_values = calc_emd_mdf1_2(segment, INITIAL_RATE)[0]
        emd_computation_time = time.time() - start_time
        
        # Fluct calculation
        start_time = time.time()
        fluct_variance, fluct_range_values, fluct_mean_diff_values, _, _ = calc_scaled_fluct_metrics(segment)
        fluct_computation_time = time.time() - start_time
        
        # Store values
        m["mnf_arv_ratio"].append(mnf_value)
        m["ima_diff"].append(ima_value)
        m["emd_mdf1"].append(emd_values[0])
        m["emd_mdf2"].append(emd_values[1])
        m["fluct_variance"].append(fluct_variance)
        m["fluct_range_values"].append(fluct_range_values)
        m["fluct_mean_diff_values"].append(fluct_mean_diff_values)
        
        # Calculate and store differentials if not the first window
        if len(m["mnf_arv_ratio"]) > 1:
            mnf_arv_diffs.append(abs(m["mnf_arv_ratio"][-1] - m["mnf_arv_ratio"][-2]))
            ima_diffs.append(abs(m["ima_diff"][-1] - m["ima_diff"][-2]))
            emd_diffs.append(abs(m["emd_mdf1"][-1] - m["emd_mdf1"][-2]))
            fluct_diffs.append(abs(m["fluct_variance"][-1] - m["fluct_variance"][-2]))
        
        # Calculate memory usage
        memory_usage = SystemMetrics.calculate_memory_usage()
        
        # Track metrics with individual computation times
        system_metrics.add_metrics(
            memory_usage - start_memory,
            (mnf_computation_time + ima_computation_time + emd_computation_time + fluct_computation_time)
        )

    # Calculate variance statistics
    variance_stats = {
        'mnf_arv_ratio': np.var(m["mnf_arv_ratio"]),
        'ima_diff': np.var(m["ima_diff"]),
        'emd': np.var(m["emd_mdf1"]),
        'fluct': np.var(m["fluct_variance"])
    }
    
    # Calculate max-min statistics
    max_min_stats = {
        'mnf_arv_ratio': max(m["mnf_arv_ratio"]) - min(m["mnf_arv_ratio"]),
        'ima_diff': max(m["ima_diff"]) - min(m["ima_diff"]),
        'emd': max(m["emd_mdf1"]) - min(m["emd_mdf1"]),
        'fluct': max(m["fluct_variance"]) - min(m["fluct_variance"])
    }
    
    # Calculate max differential statistics
    max_diff_stats = {
        'mnf_arv_ratio': max(mnf_arv_diffs) if mnf_arv_diffs else 0,
        'ima_diff': max(ima_diffs) if ima_diffs else 0,
        'emd': max(emd_diffs) if emd_diffs else 0,
        'fluct': max(fluct_diffs) if fluct_diffs else 0
    }
    
    return m, variance_stats, max_min_stats, max_diff_stats, system_metrics

   # plt.figure(1)
   # plt.plot(time_points, (m["mnf_arv_ratio"]-min(m["mnf_arv_ratio"]))/max(m["mnf_arv_ratio"]), label="mnf_arv_ratio" )
   # plt.title(f"{"Window size: ", window_size ," Step size: ", step_size}")
   # plt.xlabel('Time (s)')
   # plt.ylabel('mnf_arv_ratio')
   # plt.grid()
   # add_computation_time(plt.gca(), computation_time/num_windows)

   # plt.figure(2)
   # plt.plot(time_points, (m["ima_diff"]-min(m["ima_diff"]))/max(m["ima_diff"]), label="ima_diff")
   # plt.title(f"{"Window size: ", window_size ," Step size: ", step_size}")
   # plt.xlabel('Time (s)')
   # plt.ylabel('ima_difference')
   # plt.grid()
   # add_computation_time(plt.gca(), computation_time/num_windows)

   # plt.figure(3)
   # plt.plot(time_points, (m["emd_mdf1"]-min(m["emd_mdf1"]))/max(m["emd_mdf1"]), label="emd_mdf1")
   # plt.plot(time_points, (m["emd_mdf2"]-min(m["emd_mdf2"]))/max(m["emd_mdf2"]), label="emd_mdf2")
   # plt.title(f"{"Window size: ", window_size ," Step size: ", step_size}")
   # plt.legend()
   # plt.xlabel('Time (s)')
   # plt.ylabel('Emd')
   # plt.grid()
   # add_computation_time(plt.gca(), computation_time/num_windows)

   # plt.figure(4)
   # plt.plot(time_points, (m["fluct_variance"]-min(m["fluct_variance"]))/max(m["fluct_variance"]), label="fluct_variance")
   # plt.plot(time_points, (m["fluct_range_values"]-min(m["fluct_range_values"]))/max(m["fluct_range_values"]),   label="fluct_range_values")
   # plt.plot(time_points, (m["fluct_mean_diff_values"]-min(m["fluct_mean_diff_values"]))/max(m["fluct_mean_diff_values"]), label="fluct_mean_diff_values")
    # plt.plot(time_points, (m["fluct_entropy_values"]-min(m["fluct_entropy_values"]))/max(m["fluct_entropy_values"]) , label="fluct_entropy_values")
   # plt.title(f"{"Window size: ", window_size ," Step size: ", step_size}")
   # plt.legend()
   # plt.xlabel('Time (s)')
   # plt.ylabel('Fluctuation metrics')
   # plt.grid()
   # add_computation_time(plt.gca(), computation_time/num_windows)    

    # plt.figure(5)
    # plt.plot(m["DOM"])
    # plt.plot(m["DFS"])
    # plt.plot(m["SOM"])
    # plt.plot(m["PSE"])
    # plt.title(f"{"Window size: ", window_size ," Step size: ", step_size}")
    # plt.xlabel('Time (s)')
    # plt.ylabel('MFDMA')
    # plt.grid()
    # add_computation_time(plt.gca(), computation_time/num_windows)

   # plt.show()

'''# Save normalized plots instead of showing them
    save_plot(time_points, [(m["mnf_arv_ratio"] - min(m["mnf_arv_ratio"])) / max(m["mnf_arv_ratio"])], ["mnf_arv_ratio"], window_size, step_size)
    save_plot(time_points, [(m["ima_diff"] - min(m["ima_diff"])) / max(m["ima_diff"])], ["ima_diff"], window_size, step_size)
    save_plot(time_points, [
        (m["emd_mdf1"] - min(m["emd_mdf1"])) / max(m["emd_mdf1"]),
        (m["emd_mdf2"] - min(m["emd_mdf2"])) / max(m["emd_mdf2"]),
    ], ["emd_mdf1", "emd_mdf2"], window_size, step_size)
    save_plot(time_points, [
        (m["fluct_variance"] - min(m["fluct_variance"])) / max(m["fluct_variance"]),
        (m["fluct_range_values"] - min(m["fluct_range_values"])) / max(m["fluct_range_values"]),
        (m["fluct_mean_diff_values"] - min(m["fluct_mean_diff_values"])) / max(m["fluct_mean_diff_values"]),
    ], ["fluct_variance", "fluct_range_values", "fluct_mean_diff_values"], window_size, step_size)'''

    


def save_plot(time_points, data_list, labels, window_size, step_size):
    plt.figure()
    for data, label in zip(data_list, labels):
        plt.plot(time_points, data, label=label)
    plt.title(f"Window size: {window_size}, Step size: {step_size}")
    plt.xlabel("Time (s)")
    plt.ylabel(", ".join(labels))
    plt.legend()
    plt.grid()

    # Save the figure
    filename = f"{'_'.join(labels)}_window_{window_size}_step_{step_size}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close()

#########################################################





def calc_emd_mdf1_2(signal, sampling_rate):
    emd = EMD()
    emd.FIXE_H = 5
    emd.spline_kind = 'cubic'
    
    # Calculate IMF separation quality
    signal_std = np.std(signal)
    zero_crossings = np.sum(np.diff(np.signbit(signal)))
    min_required_length = 4 * (sampling_rate / zero_crossings) if zero_crossings > 0 else len(signal)
    reliability = min(1.0, len(signal) / min_required_length)
    
    imfs_emd = emd(signal)
    
    if len(imfs_emd) >= 2:
        imf1_std = np.std(imfs_emd[0])
        imf2_std = np.std(imfs_emd[1])
        separation_quality = min(imf1_std, imf2_std) / signal_std
        reliability *= separation_quality
    
    mdf_1 = calculate_median_frequency(imfs_emd[0], sampling_rate)
    mdf_2 = calculate_median_frequency(imfs_emd[1], sampling_rate)
    
    return (mdf_1, mdf_2), reliability

def calc_mfdma_metrics(signal, q_orders=[-5, -3, -1, 0, 1, 3, 5]):
    
    window_samples = 0 # eprepe na einai window_size, to evala 0 giati den xrhsimopoieitai kai evgaze error
    small_window_size = window_samples // 10

    scales = np.logspace(1, np.log10(len(signal)/4), 20, dtype=int)

    # Calculate moving average
    ma_signal = moving_average(signal, small_window_size)
    
    # Initialize fluctuation functions
    fluct_funcs = np.zeros((len(q_orders), len(scales)))
    
    # Calculate fluctuation functions for each scale and q-order
    for i, scale in enumerate(scales):
        # Divide signal into segments
        n_segments = len(signal) // scale
        fluctuations = []
        
        for j in range(n_segments):
            segment = signal[j*scale:(j+1)*scale]
            ma_segment = ma_signal[j*scale:(j+1)*scale]
            
            # Remove trend (moving average)
            detrended = segment - ma_segment
            
            # Calculate variance
            variance = np.mean(detrended**2)
            fluctuations.append(variance)
        
        # Calculate q-order fluctuation functions
        for q_idx, q in enumerate(q_orders):
            if q == 0:
                fluct_funcs[q_idx, i] = np.exp(0.5 * np.mean(np.log(fluctuations)))
            else:
                fluct_funcs[q_idx, i] = (np.mean(np.array(fluctuations)**(q/2)))**(1/q)
    
    # Calculate generalized Hurst exponents
    hurst = np.zeros(len(q_orders))
    for q_idx in range(len(q_orders)):
        polyfit = np.polyfit(np.log(scales), np.log(fluct_funcs[q_idx]), 1)
        hurst[q_idx] = polyfit[0]
    
    # Calculate multifractal spectrum
    tau = q_orders * hurst - 1
    alpha = np.gradient(tau, q_orders)
    f_alpha = q_orders * alpha - tau
    
    results = {
        'fluctuation_functions': fluct_funcs,
        'scales': scales,
        'q_orders': q_orders,
        'hurst_exponents': hurst,
        'multifractal_spectrum': {
            'alpha': alpha,
            'f_alpha': f_alpha
        }
    }
    
    dom = abs(max(results['hurst_exponents']) - min(results['hurst_exponents']))
    dfs = abs(results['multifractal_spectrum']['f_alpha'][0] - results['multifractal_spectrum']['f_alpha'][-1])
    som = abs(results['multifractal_spectrum']['alpha'][-1] - results['multifractal_spectrum']['alpha'][0])
    pse = results['multifractal_spectrum']['alpha'][0]
    return dom, dfs, som, pse
    
def calc_scaled_fluct_metrics(signal):
    min_scale_separation = np.log2(len(signal) / 4)
    scales = [5, 10, 20, 40]
    scale_separations = np.diff(np.log2(scales))
    scale_reliability = min(1.0, np.min(scale_separations) / min_scale_separation)
    
    processed_segment = calculate_scaled_fluctuations(signal)
    variance = calculate_signal_variance(processed_segment)
    range_values = calculate_signal_range(processed_segment)
    mean_diff_values = calculate_mean_differences(processed_segment)
    entropy_values = calculate_signal_entropy(processed_segment)
    
    return variance, range_values, mean_diff_values, entropy_values, scale_reliability

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

def moving_average(signal, window_size):
    """Calculate centered moving average of the signal"""
    weights = np.ones(window_size) / window_size
    ma = np.convolve(signal, weights, mode='same')
    
    # Handle edges
    half_window = window_size // 2
    ma[:half_window] = ma[half_window]
    ma[-half_window:] = ma[-half_window-1]
    
    return ma

def calc_mnf_arv_ratio(segment, sampling_rate):
    freq_resolution = sampling_rate / len(segment)
    mnf = calculate_mnf(segment, sampling_rate)
    arv = np.mean(np.abs(segment))
    
    # Calculate reliability based on frequency resolution
    target_resolution = 1.0  # Hz (target frequency resolution)
    reliability = min(1.0, target_resolution / freq_resolution)
    
    return (mnf/arv), reliability

def segment_and_calculate_mnf_arv_ratio(signal, time, sampling_rate, window_size, step_size):
    """
    Segment a signal into overlapping windows and calculate the MNF for each segment.

    Parameters:
        signal (np.ndarray): The input signal (time-domain).
        sampling_rate (float): The sampling rate of the signal in Hz.
        window_size (float): The size of each window in seconds.
        step_size (float): The step size between consecutive windows in seconds.

    Returns:
        list: A list of MNF values for each segment.
    """
    # Convert window and step size from seconds to samples
    window_samples = window_size
    step_samples = step_size

    # Ensure valid parameters
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Window size and step size must be greater than 0.")

    if len(signal) < window_samples:
        window_samples = len(signal)

    mnfs, arvs = [], []
    time_values = []

    # Iterate over the signal with the given window and step size
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        segment = signal[start:start + window_samples]
        mnf = calculate_mnf(segment, sampling_rate)
        arv = np.mean(np.abs(segment))
        mnfs.append(mnf)
        arvs.append(arv)
        time_values.append((time[start]+time[start + window_samples]) /2)

    return {'time_points': time_values, 'mnfs': np.array(mnfs), 'arvs': np.array(arvs), 'mnf/arv': np.array(mnfs)/np.array(arvs)}

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

def segment_signal_numpy(signal, window_size, step_size):
    # More efficient signal segmentation using NumPy
    signal_length = len(signal)
    segments = np.lib.stride_tricks.sliding_window_view(
        signal, 
        window_shape=window_size
    )[::step_size]
    return segments

def calculate_median_frequency_vectorized(segment, sampling_rate):
    # Vectorized median frequency calculation
    fft_result = np.fft.rfft(segment)
    frequencies = np.fft.rfftfreq(len(segment), 1/sampling_rate)
    power_spectrum = np.abs(fft_result)**2
    weighted_frequencies = frequencies * power_spectrum
    return np.sum(weighted_frequencies) / np.sum(power_spectrum)

def analyze_vibration_signal(signal, timestamps, sampling_freq, metrics_window_sec):
    """
    Analyze vibration signal using various time-domain metrics.
    Returns both normalized metrics and scaling factors.
    """
    # Calculate samples per window
    samples_per_window = int(metrics_window_sec * sampling_freq)
    n_windows = len(signal) // samples_per_window
    
    # Initialize arrays for metrics
    metrics = {
        'RMS': [],           
        'Zero_Crossings': [],
        "MeanFreq": [],
        "AvgRectifiedValue": [],
        "MNF/ARV": [],
        "MeanPowerFreq": [],
        "MedianFreq": [],
        "SpectralMomentsRatio": [],
        "InstantaneousMeanFreq": [],
        "InstantaneousMediumFreqBand": [],
        'LempelZivComplexity': [],  
        'WaveletEntropy': [],  
        'BandSpectralEntropy': []
    }
    
    scaling_factors = {}  # Store scaling information
    timestamps_windows = []
    
    # Process each window
    for i in range(n_windows):
        start_idx = i * samples_per_window
        end_idx = (i + 1) * samples_per_window
        
        window = signal[start_idx:end_idx]
        timestamps_windows.append(timestamps[start_idx])
        
        # RMS
        metrics['RMS'].append(np.sqrt(np.mean(window**2)))

        # Zero Crossings
        metrics['Zero_Crossings'].append(np.sum(np.diff(np.signbit(window))))

        # Avg Rectified Value (ARV)
        arv = np.mean(np.abs(window))
        metrics['AvgRectifiedValue'].append(arv)

        # FFT and Welch's PSD for frequency domain metrics
        freqs, psd = welch(window, fs=INITIAL_RATE)

        # Mean Frequency
        meanfreq = np.sum(freqs * psd) / np.sum(psd)
        metrics['MeanFreq'].append(meanfreq)

        # Mean Power Frequency
        metrics['MeanPowerFreq'].append(np.sum(psd) / len(psd))

        # Median Frequency
        cumulative_sum = np.cumsum(psd)
        half_power = cumulative_sum[-1] / 2
        metrics['MedianFreq'].append(freqs[np.where(cumulative_sum >= half_power)[0][0]])

        # Spectral Moments Ratio
        moments = [(freqs**i) @ psd for i in range(1, 4)]
        metrics['SpectralMomentsRatio'].append(moments[1] / moments[0] if moments[0] != 0 else 0)

        # Instantaneous Mean Frequency
        analytic_window = hilbert(window)
        instantaneous_phase = np.unwrap(np.angle(analytic_window))
        instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi / INITIAL_RATE)
        metrics['InstantaneousMeanFreq'].append(np.mean(instantaneous_freq))

        # Instantaneous Medium Frequency Band
        metrics['InstantaneousMediumFreqBand'].append(np.median(instantaneous_freq))

        # Lempel-Ziv Complexity
        binary_signal = ''.join(['1' if x > 0 else '0' for x in window])
        def lempel_ziv_complexity(s):
            n = len(s)
            i, l, c = 1, 1, 1
            while i + l <= n:
                if s[i:i+l] not in s[0:i]:
                    c += 1
                    i += l
                    l = 1
                else:
                    l += 1
            return c / log2(n) if n > 0 else 0
        metrics['LempelZivComplexity'].append(lempel_ziv_complexity(binary_signal))

        # Wavelet Entropy
        coeffs = wavedec(window, 'db1', level=4)
        energy = np.array([np.sum(c**2) for c in coeffs])
        total_energy = np.sum(energy)
        normalized_energy = energy / total_energy if total_energy > 0 else energy
        metrics['WaveletEntropy'].append(entropy(normalized_energy, base=2))

        # Band Spectral Entropy
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        metrics['BandSpectralEntropy'].append(entropy(psd_norm, base=2))

        # MNF/ARV ratio
        metrics['MNF/ARV'].append(meanfreq / arv if arv != 0 else 0)

    
    # Normalize all metrics to [0, 1] range and store scaling factors
    normalized_metrics = {}
    for metric in metrics:
        metric_values = np.array(metrics[metric])
        if len(metric_values) > 0:
            min_val = np.min(metric_values)
            max_val = np.max(metric_values)
            scaling_factors[metric] = {
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val
            }
            if max_val != min_val:
                normalized_metrics[metric] = (metric_values - min_val) / (max_val - min_val)
            else:
                normalized_metrics[metric] = np.zeros_like(metric_values)


    return np.array(timestamps_windows), normalized_metrics, scaling_factors

def mean_frequency(signal, sampling_rate):
    # Compute Power Spectral Density (PSD)
    freqs, psd = welch(signal, fs=sampling_rate)
    
    # Calculate the Mean Frequency
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    return mean_freq

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
    freq_resolution = INITIAL_RATE / len(segment)
    fft_magnitudes = get_fft_values(segment)
    
    # Calculate band separations
    low_band_separation = (80 - 25) / freq_resolution
    high_band_separation = (350 - 80) / freq_resolution
    
    min_required_points = 5
    reliability = min(1.0, min(low_band_separation, high_band_separation) / min_required_points)
    
    ima_value = np.mean(fft_magnitudes[index_25:index_80]) - np.mean(fft_magnitudes[index_80:index_350])
    return ima_value, reliability

def open_dialog_and_select_multiple_files():
    """
    Opens a file dialog allowing the user to select multiple files.

    Returns:
        list: A list of file paths selected by the user.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_paths = filedialog.askopenfilenames(title="Select Files")

    # Convert to a list and return
    return list(file_paths)

def add_computation_time(ax, value, label="Computation Time"):
    """
    Add a text box to a plot showing the computation time.

    Parameters:
        ax: matplotlib.axes.Axes
            The axis object to add the text to.
        value: float
            The computation time value to display.
        label: str
            Label to show with the value.
    """
    text = f"{label}: {value:.10f} s"
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )


class SystemMetrics:
    def __init__(self):
        self.memory_usages = []
        self.computation_times = []
        
    def add_metrics(self, memory_usage, computation_time):
        self.memory_usages.append(memory_usage)
        self.computation_times.append(computation_time)
    
    def get_average_metrics(self):
        return {
            'avg_memory_mb': np.mean(self.memory_usages),
            'computation_time': np.mean(self.computation_times)
        }

    def calculate_memory_usage():
        '''Returns memory usage in MB'''
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


if  __name__ == "__main__":
    main()