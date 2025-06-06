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


#############################
# Configuration

INITIAL_RATE = 800.0

window_size = 1024*1
step_size = window_size//8

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

    # EMD, EEMD
    plot_EMD(input_signal, time, Fs, window_size, step_size, filepaths)
    
    # Fluctuation Metrics
    plot_signal_fluctuation_metrics(input_signal, time, window_size, step_size, filepaths)

    # MFDMA whole
    do_whole_mfdma_and_plot(input_signal)

    # MFDMA segments
    do_segmented_mfdma_and_plot_metrics(input_signal)

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

def plot_mnf_arv_ratio(semg_signal, time, sampling_rate, window_size, step_size, filepaths):
    print("Plotting MNF/ARV Ratio..   ", end='')

    results = segment_and_calculate_mnf_arv_ratio(semg_signal, time, sampling_rate, window_size, step_size)
    cor_window = window_size *len(results['mnf/arv'])  //len(semg_signal)
    cor_step = 2*step_size *len(results['mnf/arv'])  //len(semg_signal)
    if cor_step == 0:
        cor_step = 1

    # Create plot
    plt.figure()
    plt.plot(results['time_points'], results['mnf/arv'], 'g-')
    # plt.plot(results['time_points'], [1/x for x in results['mnf_arv_ratio']], 'g-') 
    plt.title(f"MNF/ARV Ratio ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    plt.xlabel('Time (s)')
    plt.ylabel('MNF/ARV Ratio')
    # plt.title(f"ARV/MNF Ratio ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    # plt.xlabel('Time (s)')
    # plt.ylabel('ARV/MNF Ratio')
    plt.grid()

    cor = []
    for i in range(0, len(results['mnf/arv'])-cor_window, cor_step):
        cor.append(np.corrcoef(results['mnfs'][i:i+cor_window], results['arvs'][i:i+cor_window])[0, 1])
    plt.figure()
    plt.plot(cor)
    plt.title(f"Correlation Coefficient between MNF - ARV ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    # plt.xlabel('Time (s)')
    plt.ylabel('Correlation Coefficient')
    plt.grid()
    
    # Statistical summary
    # print("MNF/ARV Ratio Analysis:")
    # print(f"Mean Ratio: {np.mean(results['mnf_arv_ratio']):.4f}")
    # print(f"Ratio Standard Deviation: {np.std(results['mnf_arv_ratio']):.4f}")

    print("Done.")
    return results

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

def plot_EMD(signal, time, sampling_rate, window_size, step_size, filepaths):

    print("Plotting EMDs..   ")
    
    # Empirical Mode Decomposition (EMD)

    emd = EMD()
    emd.FIXE_H = 5  # Maximum number of sifting iterations
    emd.spline_kind = 'cubic'  # Interpolation type for envelope computation
    imfs_emd = emd(signal)    

    plt.figure()
    mdf_1, mdf_2, times_mdf = [], [], []
    for start in range(0, len(imfs_emd[0]) - window_size + 1, step_size):
        mdf_1.append(calculate_median_frequency(imfs_emd[0][start:start + window_size], sampling_rate))
        mdf_2.append(calculate_median_frequency(imfs_emd[1][start:start + window_size], sampling_rate))
        times_mdf.append((time[start] + time[start + window_size])/2)
    plt.plot(mdf_1, label=f"IMF 1")
    plt.plot(mdf_2, label=f"IMF 2")
    plt.ylabel("Median Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(f"EMD ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    plt.legend(loc='upper right')
    plt.grid()
    

    # # Ensemble Empirical Mode Decomposition (EEMD)

    # eemd = EEMD()
    # eemd.noise_seed(42)  # For reproducibility
    # eemd.trials = 50  # Number of ensembles
    # eemd.noise_width = 0.2  # Noise amplitude as a fraction of signal SD
    # imfs_eemd = eemd(signal)

    # plt.figure()
    # for count, imfs in enumerate(imfs_eemd):
    #     mdf = []
    #     for start in range(0, len(signal) - window_size + 1, step_size):
    #         mdf.append(calculate_median_frequency(imfs[start:start + window_size], sampling_rate))
    #     plt.plot(mdf, label=f"IMF {count+1}")
    # plt.ylabel("Median Frequency (Hz)")
    # plt.xlabel("Time (s)")
    # plt.title(f"EEMD ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    # plt.legend(loc='upper right')
    # plt.grid()

    print("Done.")

def plot_signal_fluctuation_metrics(signal, time, window_len, step_len, filepaths):
    metrics, time_axis, processed_signal = compute_windowed_signal_metrics(
        signal, time, window_len, step_len
    )
    
    plt.figure()
    plt.plot(time, processed_signal)
    plt.xlabel("Time (s)")
    plt.title(f"Processed Signal ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    plt.grid()

    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, metrics["variance"], label="Variance", color="b")
    plt.title(f"Signal Variance ({filepaths[0].split('/')[-1].split('_ID')[0]})")
    plt.ylabel("Variance")
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, metrics["range"], label="Range", color="g")
    plt.title("Signal Range")
    plt.ylabel("Range")
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, metrics["mean_diff"], label="Mean Diff", color="r")
    plt.title("Mean Absolute Differences")
    plt.ylabel("Mean Diff")
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(time_axis, metrics["entropy"], label="Entropy", color="m")
    plt.title("Signal Entropy")
    plt.ylabel("Entropy")
    plt.xlabel("Time (s)")
    plt.grid(True)
    
    plt.tight_layout()

def do_whole_mfdma_and_plot(signal, q_orders=[-5, -3, -1, 0, 1, 3, 5], scales=None, window_size=None):
    """
    Implement true Multifractal Detrended Moving Average (MFDMA) analysis
    
    Parameters:
    signal: array-like, input signal
    q_orders: array-like, q-orders for multifractal analysis
    scales: array-like, scales for analysis (if None, automatically determined)
    window_size: int, size of moving average window (if None, automatically determined)
    
    Returns:
    dict containing:
        - fluctuation_functions: F(q,s) for each q-order and scale
        - hurst_exponents: h(q) for each q-order
        - multifractal_spectrum: f(α) vs α
    """
    if scales is None:
        scales = np.logspace(1, np.log10(len(signal)/4), 20, dtype=int)
    
    if window_size is None:
        window_size = len(signal) // 10

    # Calculate moving average
    ma_signal = moving_average(signal, window_size)
    
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
    
    rtrnble = {
        'fluctuation_functions': fluct_funcs,
        'scales': scales,
        'q_orders': q_orders,
        'hurst_exponents': hurst,
        'multifractal_spectrum': {
            'alpha': alpha,
            'f_alpha': f_alpha
        }
    }
    plot_mfdma_results(rtrnble)

def do_segmented_mfdma_and_plot_metrics(wholesignal, q_orders=[-5, -3, -1, 0, 1, 3, 5]):
    """
    Implement true Multifractal Detrended Moving Average (MFDMA) analysis
    
    Parameters:
    signal: array-like, input signal
    q_orders: array-like, q-orders for multifractal analysis
    scales: array-like, scales for analysis (if None, automatically determined)
    window_size: int, size of moving average window (if None, automatically determined)
    
    Returns:
    dict containing:
        - fluctuation_functions: F(q,s) for each q-order and scale
        - hurst_exponents: h(q) for each q-order
        - multifractal_spectrum: f(α) vs α
    """
    
    window_samples = window_size
    small_window_size = window_samples // 10
    
    dom_li, dfs_li, som_li, pse_li = [],[],[],[]

    for start in range(0, len(wholesignal) - window_samples + 1, step_size):
        signal = wholesignal[start:start + window_samples]
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
        # plot_mfdma_results(results)
        
        dom_li.append(abs(max(results['hurst_exponents']) - min(results['hurst_exponents'])))
        dfs_li.append(abs(results['multifractal_spectrum']['f_alpha'][0] - results['multifractal_spectrum']['f_alpha'][-1]))
        som_li.append(abs(results['multifractal_spectrum']['alpha'][-1] - results['multifractal_spectrum']['alpha'][0]))
        pse_li.append(results['multifractal_spectrum']['alpha'][0])
        # print(dom_li[-1])
    
    plt.figure()
    plt.plot(dom_li, label='DOM')
    plt.plot(dfs_li, label='DFS')
    plt.plot(som_li, label='SOM')
    plt.plot(pse_li, label='PSE')
    plt.xlabel('Segment Index')
    plt.ylabel('Value')
    plt.title('MFDMA Segment Metrics')
    plt.legend()




def plot_mfdma_results(results):
    """Plot MFDMA analysis results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot fluctuation functions
    for i, q in enumerate(results['q_orders']):
        ax1.loglog(results['scales'], results['fluctuation_functions'][i], 
                  'o-', label=f'q={q}')
    ax1.set_xlabel('Scale')
    ax1.set_ylabel('F(q,s)')
    ax1.set_title('Fluctuation Functions')
    ax1.legend()
    ax1.grid(True)
    
    # Plot generalized Hurst exponents
    ax2.plot(results['q_orders'], results['hurst_exponents'], 'o-')
    ax2.set_xlabel('q')
    ax2.set_ylabel('h(q)')
    ax2.set_title('Generalized Hurst Exponents')
    ax2.grid(True)
    
    # Plot multifractal spectrum
    ax3.plot(results['multifractal_spectrum']['alpha'], 
             results['multifractal_spectrum']['f_alpha'], 'o-')
    ax3.set_xlabel('α')
    ax3.set_ylabel('f(α)')
    ax3.set_title('Multifractal Spectrum')
    ax3.grid(True)
    
    plt.tight_layout()
    return fig

def compute_windowed_signal_metrics(signal, time, window_len, step_len):
    time_axis = []
    processed_signal = calculate_scaled_fluctuations(signal)
    
    variance_values = []
    range_values = []
    mean_diff_values = []
    entropy_values = []
    
    for start in range(0, len(processed_signal) - window_len + 1, step_len):
        segment = processed_signal[start:start + window_len]
        variance_values.append(calculate_signal_variance(segment))
        range_values.append(calculate_signal_range(segment))
        mean_diff_values.append(calculate_mean_differences(segment))
        entropy_values.append(calculate_signal_entropy(segment))
        time_axis.append((time[start] + time[start + window_len]) / 2)
    
    return {
        "variance": np.array(variance_values),
        "range": np.array(range_values),
        "mean_diff": np.array(mean_diff_values),
        "entropy": np.array(entropy_values),
    }, time_axis, processed_signal

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