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
from scipy.signal import welch, hilbert
from scipy.stats import entropy
from math import log2
from PyEMD import EMD, EEMD
import pywt


#############################
# Configuration

INITIAL_RATE = 800.0

window_size = 1024*2
step_size = window_size//32

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



    # Initial Signal
    plot_signal(input_signal, time, filepaths, Fs)

    # MNF/ARV
    plot_mnf_arv_ratio(input_signal, time, Fs, window_size, step_size, filepaths)

    # IMA diff
    plot_IMA_diff(input_signal, time, Fs, filepaths)

    # Wavedec, EMD, EEMD
    # plot_emd([1,1,0], input_signal, Fs, window_size, step_size)
    # plot_emd_optimized(input_signal, Fs, window_size, step_size)
    
    # MFDMA
    # plot_mfdma_features(input_signal)
    mfdma_analysis_haiku(input_signal, window_sizes=None)


    plt.show()


#########################################################


def plot_signal(input_signal, time, filepaths, Fs):
    print("Plotting signal..   ", end='')
    # Create the main figure
    fig = plt.figure()
    plt.plot(time, input_signal)
    plt.title(f"Signal: {filepaths[0].split('/')[-1].split('_ID')[0]}")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    print("Done.")

def plot_IMA_diff(input_signal, time, Fs, filepaths):
    print("Plotting IMA diff..   ", end='')

    imas, imas_time = calc_progressive_fft(input_signal, time, Fs)
    
    plt.figure()
    plt.plot(imas_time, imas)
    plt.xlabel("Time (s)")
    plt.title(f"IMA Low-High Component Difference ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    plt.grid()
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
    coeffs = pywt.wavedec(signal, 'db4', level=5)
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
        coeffs = pywt.wavedec(signal, 'db4', level=5)
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

def mfdma_analysis_haiku(input_signal, window_sizes=None, min_segment_length=4):
    """
    Perform Multifractal Detrended Moving Average (MFDMA) analysis with improved robustness.
    
    Parameters:
    -----------
    input_signal : array_like
        Input signal time series
    window_sizes : array_like, optional
        Range of window sizes (default: logarithmically spaced)
    min_segment_length : int, optional
        Minimum segment length to prevent linear algebra issues (default: 4)
    
    Returns:
    --------
    dict: Contains multifractal features and analysis results
    """
    # Ensure input is a numpy array
    x = np.asarray(input_signal, dtype=float)
    
    # If window sizes not provided, create logarithmic scale
    if window_sizes is None:
        # Ensure window sizes are not too small
        max_window = len(x) // 10
        window_sizes = np.logspace(0, np.log10(max_window), 20, dtype=int)
        window_sizes = window_sizes[window_sizes >= min_segment_length]
    
    # Ensure window sizes are valid
    window_sizes = np.unique(np.clip(window_sizes, min_segment_length, len(x)//2))
    
    # Initialize storage for fluctuation functions
    q_range = np.linspace(-5, 5, 21)
    F_q = np.zeros((len(q_range), len(window_sizes)))
    
    # Integrate the signal
    x_integrated = np.cumsum(x - np.mean(x))
    
    # Compute fluctuation function for each q-order and window size
    for j, window in enumerate(window_sizes):
        # Divide integrated signal into non-overlapping segments
        segments = [x_integrated[i:i+window] for i in range(0, len(x_integrated)-window+1, window)]
        
        # Robust trend estimation
        segment_trends = []
        for seg in segments:
            try:
                # Use robust linear regression 
                if len(seg) < min_segment_length:
                    # If segment is too short, use mean as trend
                    trend = np.ones_like(seg) * np.mean(seg)
                else:
                    # Standard linear trend
                    x_seg = np.arange(len(seg))
                    coeffs = np.polyfit(x_seg, seg, 1)
                    trend = np.polyval(coeffs, x_seg)
                segment_trends.append(trend)
            except Exception:
                # Fallback to mean trend if any computation fails
                trend = np.ones_like(seg) * np.mean(seg)
                segment_trends.append(trend)
        
        # Compute fluctuations
        fluctuations = [(seg - trend)**2 for seg, trend in zip(segments, segment_trends)]
        
        # Compute fluctuation function for different q-orders
        for i, q in enumerate(q_range):
            if q == 0:
                # Use log of fluctuations for q=0
                F_q[i, j] = np.exp(0.5 * np.mean(np.log(np.maximum(fluctuations, 1e-10))))
            else:
                F_q[i, j] = (np.mean([fl**q for fl in fluctuations]))**(1/q)
    
    # Compute Hurst exponent and multifractal spectrum
    h_q = np.zeros_like(q_range)
    for i in range(len(q_range)):
        # Prevent log(0) and handle potential numerical issues
        valid_indices = (F_q[i, :] > 0) & (window_sizes > 0)
        if np.sum(valid_indices) < 2:
            # Not enough valid data points
            continue
        
        # Linear regression of log(F_q) vs log(window_sizes)
        log_windows = np.log(window_sizes[valid_indices])
        log_fq = np.log(F_q[i, valid_indices])
        
        try:
            coeffs = np.polyfit(log_windows, log_fq, 1)
            h_q[i] = coeffs[0]
        except Exception:
            h_q[i] = np.nan
    
    # Clean up Hurst exponent
    h_q = h_q[~np.isnan(h_q)]
    
    # Compute multifractal spectrum
    f_alpha = q_range[:len(h_q)] * h_q - 1
    alpha = (f_alpha - np.min(f_alpha)) / (np.max(f_alpha) - np.min(f_alpha))
    
    # Compute key multifractal features
    features = {
        'Span of Multifractal Singularity (SOM)': np.max(alpha) - np.min(alpha),
        'Degree of Multifractality (DOM)': np.std(h_q) if len(h_q) > 0 else 0,
        'Peak Singularity Exponent (PSE)': alpha[np.argmax(f_alpha)] if len(alpha) > 0 else 0
    }
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Fluctuation Function
    plt.subplot(131)
    plt.loglog(window_sizes, F_q.T, 'o-')
    plt.title('Fluctuation Function')
    plt.xlabel('Window Size')
    plt.ylabel('F(q, window_size)')
    plt.legend([f'q={q}' for q in q_range], loc='best', ncol=2)
    
    # # Plot 2: Hurst Exponent
    # plt.subplot(132)
    # plt.plot(q_range[:len(h_q)], h_q, 'o-')
    # plt.title('Generalized Hurst Exponent')
    # plt.xlabel('q-order')
    # plt.ylabel('h(q)')
    
    # # Plot 3: Multifractal Spectrum
    # plt.subplot(133)
    # plt.plot(alpha, f_alpha, 'o-')
    # plt.title('Multifractal Spectrum')
    # plt.xlabel('Singularity Strength (α)')
    # plt.ylabel('Singularity Dimension (f(α))')
    
    plt.tight_layout()
    
    # Print features
    print("Multifractal Features:")
    for name, value in features.items():
        print(f"{name}: {value:.4f}")
    
    return {
        'fluctuation_function': F_q,
        'window_sizes': window_sizes,
        'hurst_exponent': h_q,
        'multifractal_spectrum': {
            'alpha': alpha,
            'f_alpha': f_alpha
        },
        'features': features
    }
    
def plot_mnf_arv_ratio(semg_signal, time, sampling_rate, window_size, step_size, filepaths):

    results = segment_and_calculate_mnf_arv_ratio(semg_signal, time, sampling_rate, window_size, step_size)
    
    # Create plot
    plt.figure()
    plt.plot(results['time_points'], [1/x for x in results['mnf_arv_ratio']], 'g-') 
    # plt.plot(results['time_points'], results['mnf_arv_ratio'], 'g-')
    # plt.title(f'MNF/ARV Ratio ({filepaths[0].split('/')[-1].split('_ID')[0]})')
    # plt.xlabel('Time (s)')
    # plt.ylabel('MNF/ARV Ratio')
    plt.title(f"ARV/MNF Ratio ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    plt.xlabel('Time (s)')
    plt.ylabel('ARV/MNF Ratio')
    plt.grid()
    
    
    # Statistical summary
    # print("MNF/ARV Ratio Analysis:")
    # print(f"Mean Ratio: {np.mean(results['mnf_arv_ratio']):.4f}")
    # print(f"Ratio Standard Deviation: {np.std(results['mnf_arv_ratio']):.4f}")
    
    return results



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

    ratios = []
    time_values = []

    # Iterate over the signal with the given window and step size
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        segment = signal[start:start + window_samples]
        mnf = calculate_mnf(segment, sampling_rate)
        arv = np.mean(np.abs(segment))
        ratios.append(mnf/arv)
        time_values.append((time[start]+time[start + window_samples]) /2)

    return {'time_points': time_values, 'mnf_arv_ratio': ratios}

# Helper function for MNF calculation (from the earlier implementation)
def calculate_mnf(signal, sampling_rate):
    freqs = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    fft_vals = np.fft.rfft(signal)
    psd = np.abs(fft_vals) ** 2
    mnf = np.sum(freqs * psd) / np.sum(psd)
    return mnf

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

def calc_progressive_fft(signal, time, Fs):
    # Setup frequencies for the FFT (only positive frequencies)
    frequencies = fftfreq(window_size, 1 / Fs)[:window_size // 2]
    index_25 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 25))
    index_80 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 80))
    index_350 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 350))

    imas = []
    imas_time = []
    
    # Progressively plot the FFT of each window
    idx = 0
    running = True  # Control flag for stopping early

    while idx + window_size <= len(signal) and running:
        # Extract the current window of the signal
        windowed_signal = signal[idx:idx + window_size]

        # Perform FFT and get magnitude (assuming m.get_fft_values exists)
        fft_magnitudes = get_fft_values(windowed_signal)
        
        # Update imas
        imas.append(np.mean(fft_magnitudes[index_25:index_80]) - np.mean(fft_magnitudes[index_80:index_350]))
        imas_time.append((time[idx] + time[idx + window_size])/2)

        # Increment the index by step_size
        idx += step_size

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