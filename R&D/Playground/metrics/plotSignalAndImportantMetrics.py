"""
Opens a dialog for selecting multiple json files which contain the signal data.
Merges the signal data from all the selected files and plots it.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog
import json
from scipy.signal import welch, hilbert, periodogram
from scipy.stats import entropy
from math import log2
from PyEMD import EMD, EEMD
from pywt import wavedec
from MFDFA import MFDFA

#############################
# Configuration

INITIAL_RATE = 800.0

fft_window_size = 1024*2
fft_step_size = fft_window_size//2

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
    time            = np.arange(num_samples) / Fs  # Time array for plotting

    # Plot the upsampled signal
    print('Plotting..')

    plot_signal(input_signal, time, filepaths, Fs)
    semg_mnf_arv_ratio(input_signal, Fs, fft_window_size, overlap=0.5)
    plot_IMA_diff(input_signal, Fs, filepaths)
    # plot_emd([1,1,0], input_signal, Fs, fft_window_size, fft_step_size)
    # plot_emd_optimized(input_signal, Fs, fft_window_size, fft_step_size)
    # plot_mfdma_features(input_signal)

    window_sizes = [320,640,1280,2560]
    q_vals = [-5,-2,0,2,5] 
    mfdma_with_segments(input_signal, window_sizes, q_vals, poly_degree=2, plot_segments=['first', 'last'])

    plt.show()


#########################################################


def plot_signal(input_signal, time, filepaths, Fs):
    print("Plotting signal..   ", end='')
    # Create the main figure
    fig = plt.figure(figsize=(6, 4))
    plt.plot(time, input_signal)
    plt.title(f"Signal: {filepaths[0].split('/')[-1].split('_ID')[0]}")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    print("Done.")

def plot_IMA_diff(input_signal, Fs, filepaths):
    print("Plotting IMA diff..   ", end='')

    imas, imas_time = calc_progressive_fft(input_signal, Fs)
    
    plt.figure(figsize=(6, 4))
    plt.plot(imas_time, imas)
    plt.xlabel("Time (s)")
    plt.title(f"IMA Low-High Component Difference ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    # plt.show()
    
    print("Done.")

def plot_emd_optimized(signal, sampling_rate, window_size, step_size):
    print("Plotting opt EMDs..   ")
    
    # Use NumPy for segmentation
    segments = segment_signal_numpy(signal, window_size, step_size)

    # Vectorized Median Frequency calculation for raw signal
    mfs_raw = np.array([
        calculate_median_frequency_vectorized(segment, sampling_rate) 
        for segment in segments
    ])

    # 1. Discrete Wavelet Transform (DWT) - Optimized
    coeffs = wavedec(signal, 'db4', level=5)
    d1_component = coeffs[0]
    segments_d1 = segment_signal_numpy(d1_component, window_size // 2, step_size // 2)
    mfs_dwt = np.array([
        calculate_median_frequency_vectorized(segment, sampling_rate / 2) 
        for segment in segments_d1
    ])
    print("wavedec complete, ")

    # 2. Empirical Mode Decomposition (EMD) - Parallel Processing
    emd = EMD()
    imfs_emd = emd(signal)
    segments_emd = segment_signal_numpy(imfs_emd[0], window_size, step_size)
    mfs_emd = np.array([
        calculate_median_frequency_vectorized(segment, sampling_rate) 
        for segment in segments_emd
    ])
    print("EMD complete, ")

    # 3. Ensemble Empirical Mode Decomposition (EEMD) - Parallel Processing
    # eemd = EEMD()
    # imfs_eemd = eemd(signal)
    # segments_eemd = segment_signal_numpy(imfs_eemd[0], window_size, step_size)
    # mfs_eemd = np.array([
    #     calculate_median_frequency_vectorized(segment, sampling_rate) 
    #     for segment in segments_eemd
    # ])
    # print("EEMD complete. ")

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(mfs_raw, label="Raw Signal", marker='o')
    plt.plot(mfs_dwt, label="DWT (D1 Component)", marker='x')
    plt.plot(mfs_emd, label="EMD (IMF1)", marker='s')
    # plt.plot(mfs_eemd, label="EEMD (IMF1)", marker='d')
    plt.xlabel("Segment Index")
    plt.ylabel("Median Frequency (Hz)")
    plt.title("Median Frequency Over Time Using OPT Different Preprocessing Methods")
    plt.legend()
    plt.grid()
    
    print("Done.")

def plot_emd(choice, signal, sampling_rate, window_size, step_size):
    print("Plotting EMDs..   ")
    
    segments = segment_signal(signal, window_size, step_size)

    # Median Frequency calculation for raw signal
    mfs_raw = [calculate_median_frequency(segment, sampling_rate) for segment in segments]
    plt.figure(figsize=(12, 8))

    # Preprocessing Methods
    # 1. Discrete Wavelet Transform (DWT)
    if choice[0]:
        coeffs = wavedec(signal, 'db4', level=5)
        d1_component = coeffs[0]  # Highest frequency component
        segments_d1 = segment_signal(d1_component, window_size // 2, step_size // 2)
        mfs_dwt = [calculate_median_frequency(segment, sampling_rate / 2) for segment in segments_d1]
        print("wavedec complete, ")
        plt.plot(mfs_dwt, label="DWT (D1 Component)", marker='x')

    # 2. Empirical Mode Decomposition (EMD)
    if choice[1]:
        emd = EMD()
        imfs_emd = emd(signal)
        segments_emd = segment_signal(imfs_emd[0], window_size, step_size)
        mfs_emd = [calculate_median_frequency(segment, sampling_rate) for segment in segments_emd]
        print("emd complete, ")
        plt.plot(mfs_emd, label="EMD (IMF1)", marker='s')

    # 3. Ensemble Empirical Mode Decomposition (EEMD)
    if choice[2]:
        eemd = EEMD()
        imfs_eemd = eemd(signal)
        segments_eemd = segment_signal(imfs_eemd[0], window_size, step_size)
        mfs_eemd = [calculate_median_frequency(segment, sampling_rate) for segment in segments_eemd]
        print("eemd complete.")
        plt.plot(mfs_eemd, label="EEMD (IMF1)", marker='d')

    # Plotting
    plt.plot(mfs_raw, label="Raw Signal", marker='o')
    plt.xlabel("Segment Index")
    plt.ylabel("Median Frequency (Hz)")
    plt.title("Median Frequency Over Time Using Different Preprocessing Methods")
    plt.legend()
    plt.grid()
    
    print("Done.")

def mfdma_analysis(signal, scale_range=(10, 500), q_values=[-5, -3, 0, 3, 5]):
    """
    Perform MFDMA on an sEMG signal and compute key multifractal features.

    Parameters:
        signal (numpy array): The input sEMG signal.
        scale_range (tuple): The range of scales for the analysis (min, max).
        q_values (list): List of q values for multifractal analysis.

    Returns:
        dict: A dictionary containing the Hurst exponents, SOM, DOM, and PSE.
    """
    def moving_average(y, scale):
        return np.convolve(y, np.ones(scale) / scale, mode='valid')

    def hurst_exponent(residuals, scale):
        rms = np.sqrt(np.mean(residuals**2))
        return np.log(rms) / np.log(scale)

    print("Calculating MFDMA..")
    
    N = len(signal)
    scales = np.arange(scale_range[0], scale_range[1])
    hurst_values = []

    for scale in scales:
        if scale >= len(signal):
            break
        smoothed = moving_average(signal, scale)
        residuals = signal[:len(smoothed)] - smoothed
        hurst_values.append([hurst_exponent(residuals, scale) for q in q_values])

    hurst_values = np.array(hurst_values)
    
    # Compute key multifractal features
    H_max = np.max(hurst_values, axis=0)
    H_min = np.min(hurst_values, axis=0)

    SOM = H_max - H_min
    DOM = H_max - H_min
    PSE = hurst_values[:, q_values.index(-5)]

    # Return results as a dictionary
    return {
        "hurst_values": hurst_values,
        "SOM": SOM,
        "DOM": DOM,
        "PSE": PSE
    }

def plot_mfdma_features(signal):
    """
    Plot the key multifractal features of the sEMG signal.

    Parameters:
        signal (numpy array): The input sEMG signal.
        features (dict): The computed multifractal features.
    """
    features = mfdma_analysis(signal)
    
    hurst_values = features["hurst_values"]
    SOM = features["SOM"]
    DOM = features["DOM"]
    PSE = features["PSE"]

    scales = np.arange(len(hurst_values))

    # Plot the Hurst exponent for different q-values
    plt.figure(figsize=(12, 6))
    for i, q in enumerate([-5, -3, 0, 3, 5]):
        plt.plot(scales, hurst_values[:, i], label=f"q={q}")
    plt.title("Hurst Exponent for Different q-values")
    plt.xlabel("Scales")
    plt.ylabel("Hurst Exponent")
    plt.legend()
    plt.grid(True)

    # Plot SOM, DOM, and PSE
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(SOM, label="SOM", color="b")
    plt.title("Span of Multifractal Singularity Intensity (SOM)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(DOM, label="DOM", color="r")
    plt.title("Degree of Multifractality (DOM)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(PSE, label="PSE", color="g")
    plt.title("Peak Singularity Exponent (PSE)")
    plt.grid(True)

    plt.tight_layout()
    
    print("Done.")

def mfdma_with_segments(data, window_sizes, q_vals, poly_degree=2, plot_segments=['first', 'last']):
    """
    Perform Multifractal Detrended Moving Average (MFDMA) and plot results for first and last segments.
    
    Parameters:
        data (array-like): The input time series data.
        window_sizes (list): List of window sizes (s) for analysis.
        q_vals (array-like): Range of q values (e.g., np.arange(-5, 6)).
        poly_degree (int): Degree of polynomial for detrending.
        plot_segments (list): List containing 'first', 'last', or both.
        
    Returns:
        dict: Contains h(q), alpha, and f(alpha) for each requested segment.
    """
    def calculate_fluctuations(data_segment, s, poly_degree):
        """Calculate fluctuation for a given segment and window size."""
        n_segments = len(data_segment) // s
        F_s = []
        for i in range(n_segments):
            segment = data_segment[i * s : (i + 1) * s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, poly_degree)
            trend = np.polyval(coeffs, x)
            F_s.append(np.mean((segment - trend) ** 2))
        return np.sqrt(np.mean(F_s))
    
    def perform_analysis(segment, window_sizes, q_vals):
        """Perform fluctuation analysis for each window size and q value."""
        fluctuation_results = []
        for s in window_sizes:
            F_q = []
            for q in q_vals:
                fluctuations = calculate_fluctuations(segment, s, poly_degree)
                if q == 0:
                    F_q.append(np.log(fluctuations))  # q = 0 is a special case
                else:
                    F_q.append((fluctuations ** q) ** (1 / q))
            fluctuation_results.append(F_q)
        return np.array(fluctuation_results)
    
    def compute_hurst_and_spectrum(results, q_vals):
        """Compute Hurst exponents and multifractal spectrum."""
        log_s = np.log(window_sizes)
        h_q = []
        for q_index in range(len(q_vals)):
            log_F_q = np.log(results[:, q_index])
            h, _ = np.polyfit(log_s, log_F_q, 1)
            h_q.append(h)
        h_q = np.array(h_q)
        alpha = h_q + q_vals * h_q
        f_alpha = q_vals * alpha - h_q
        return h_q, alpha, f_alpha
    
    first_segment = data[:window_sizes[-1]]
    last_segment = data[-window_sizes[-1]:]
    
    results = {}
    if 'first' in plot_segments:
        first_results = perform_analysis(first_segment, window_sizes, q_vals)
        h_q_first, alpha_first, f_alpha_first = compute_hurst_and_spectrum(first_results, q_vals)
        results['first'] = {'h_q': h_q_first, 'alpha': alpha_first, 'f_alpha': f_alpha_first}
    
    if 'last' in plot_segments:
        last_results = perform_analysis(last_segment, window_sizes, q_vals)
        h_q_last, alpha_last, f_alpha_last = compute_hurst_and_spectrum(last_results, q_vals)
        results['last'] = {'h_q': h_q_last, 'alpha': alpha_last, 'f_alpha': f_alpha_last}
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: f(α) vs α
    plt.subplot(1, 2, 1)
    if 'first' in plot_segments:
        plt.plot(results['first']['alpha'], results['first']['f_alpha'], label='First Segment')
    if 'last' in plot_segments:
        plt.plot(results['last']['alpha'], results['last']['f_alpha'], label='Last Segment')
    plt.xlabel('α (Singularity Strength)')
    plt.ylabel('f(α) (Multifractal Spectrum)')
    plt.title('Multifractal Spectrum')
    plt.legend()
    
    # Subplot 2: h(q) vs q
    plt.subplot(1, 2, 2)
    if 'first' in plot_segments:
        plt.plot(q_vals, results['first']['h_q'], label='First Segment')
    if 'last' in plot_segments:
        plt.plot(q_vals, results['last']['h_q'], label='Last Segment')
    plt.xlabel('q')
    plt.ylabel('h(q) (Hurst Exponent)')
    plt.title('Hurst Exponents vs q')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

def semg_mnf_arv_ratio(semg_signal, sampling_rate, window_size, overlap=0.5):
    """
    Calculate and plot the ratio of Mean Frequency (MNF) to Average Rectified Value (ARV)
    
    Parameters:
    -----------
    semg_signal : array_like
        Input sEMG time series signal
    sampling_rate : float
        Sampling frequency of the signal (Hz)
    window_size : float
        Window size in seconds
    overlap : float, optional
        Overlap between windows (0-1, default: 0.5)
    
    Returns:
    --------
    dict: Contains MNF/ARV ratio analysis results
    """
    # Convert signal to numpy array
    signal_data = semg_signal
    
    # Convert window size to samples
    window_samples = int(window_size * sampling_rate)
    overlap_samples = int(window_samples * overlap)
    
    # Prepare storage for results
    results = {
        'mnf_arv_ratio': [],
        'mnf': [],
        'arv': [],
        'time_points': []
    }
    
    # Sliding window analysis
    for start in range(0, len(signal_data) - window_samples + 1, window_samples - overlap_samples):
        # Extract window
        window = signal_data[start:start+window_samples]
        
        # Compute frequency spectrum
        f, Pxx = periodogram(window, fs=sampling_rate)
        
        # Mean Frequency Calculation
        mnf = np.sum(f * Pxx) / np.sum(Pxx)
        
        # Average Rectified Value
        arv = np.mean(np.abs(window))
        
        # Calculate MNF/ARV ratio (handle potential zero division)
        mnf_arv_ratio = mnf / arv if arv != 0 else 0
        
        # Store results
        results['mnf'].append(mnf)
        results['arv'].append(arv)
        results['mnf_arv_ratio'].append(mnf_arv_ratio)
        results['time_points'].append(start / sampling_rate)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['time_points'], results['mnf_arv_ratio'], 'g-')
    plt.title(f'MNF/ARV Ratio (Window Size: {window_size}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('MNF/ARV Ratio')
    plt.grid(True)
    plt.tight_layout()
    
    
    # Statistical summary
    print("MNF/ARV Ratio Analysis:")
    print(f"Mean Ratio: {np.mean(results['mnf_arv_ratio']):.4f}")
    print(f"Ratio Standard Deviation: {np.std(results['mnf_arv_ratio']):.4f}")
    # print(f"Minimum Ratio: {np.min(results['mnf_arv_ratio']):.4f}")
    # print(f"Maximum Ratio: {np.max(results['mnf_arv_ratio']):.4f}")
    
    return results

def calculate_median_frequency(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]
    mf = freqs[np.searchsorted(cumulative_energy, total_energy / 2)]
    return mf

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

def plot_interactive_metrics(timestamps, metrics, scaling_factors, metrics_window_sec):
    """
    Plot all normalized metrics with interactive legend and scaling information
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    lines = []
    labels = []
    for metric, values in metrics.items():
        scale = scaling_factors[metric]
        # Create label with scaling information
        label = (f"{metric}\n"
                f"[Range: {scale['min']:.2e} to {scale['max']:.2e}]")
        line, = ax.plot(timestamps, values, label=label, alpha=0.7)
        lines.append(line)
        labels.append(label)
    
    # Create the legend
    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Enable picking on the legend
    for legline in leg.get_lines():
        legline.set_picker(True)
        legline.set_pickradius(5)  # Distance in points
    
    # Create the picker function
    def on_pick(event):
        legline = event.artist
        origline = lines[labels.index(legline.get_label())]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change alpha of legend item
        legline.set_alpha(2.0 if visible else 0.2)
        fig.canvas.draw()
    
    # Connect the pick event
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    plt.title(f'Normalized Vibration Analysis Metrics (Window: {metrics_window_sec}s)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude (0-1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

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

def calc_progressive_fft(signal, Fs):
    # Setup frequencies for the FFT (only positive frequencies)
    frequencies = fftfreq(fft_window_size, 1 / Fs)[:fft_window_size // 2]
    index_25 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 25))
    index_80 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 80))
    index_350 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 350))

    imas = []
    imas_time = []
    
    # Progressively plot the FFT of each window
    idx = 0
    running = True  # Control flag for stopping early

    while idx + fft_window_size <= len(signal) and running:
        # Extract the current window of the signal
        windowed_signal = signal[idx:idx + fft_window_size]

        # Perform FFT and get magnitude (assuming m.get_fft_values exists)
        fft_magnitudes = get_fft_values(windowed_signal)
        
        # Update imas
        imas.append(np.mean(fft_magnitudes[index_25:index_80]) - np.mean(fft_magnitudes[index_80:index_350]))
        imas_time.append(idx + fft_window_size/2)

        # Increment the index by step_size
        idx += fft_step_size

    return imas, imas_time

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



if  __name__ == "__main__":
    main()