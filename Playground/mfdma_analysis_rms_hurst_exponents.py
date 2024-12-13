import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import json
from scipy.stats import linregress

def select_files():
    """Opens a dialog window to select files and returns their paths as a list."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return list(file_paths)

def preprocess_time_series(time_series):
    """Preprocess the time series by calculating the cumulative sum and subtracting the mean."""
    mean_removed = time_series - np.mean(time_series)
    cumulative_sum = np.cumsum(mean_removed)
    return cumulative_sum

def centered_moving_average(time_series, window_size):
    """Calculate the centered moving average of a time series."""
    moving_avg = np.convolve(time_series, np.ones(window_size) / window_size, mode='valid')
    return moving_avg

def calculate_rms_fluctuations(detrended, segment_size, overlap):
    """Compute RMS fluctuations for overlapping segments of the detrended series."""
    overlap_step = segment_size - overlap
    rms_fluctuations = []

    for start in range(0, len(detrended) - segment_size + 1, overlap_step):
        segment = detrended[start:start + segment_size]
        rms = np.sqrt(np.mean(segment**2))
        rms_fluctuations.append(rms)

    return np.array(rms_fluctuations)

def mfdma_core(time_series, window_sizes, q_values):
    """Perform the core MF-DMA analysis."""
    results = {}
    for window_size in window_sizes:
        overlap = window_size // 2

        moving_avg = centered_moving_average(time_series, window_size)

        half_window = window_size // 2
        detrended = time_series[half_window:-half_window + 1] - moving_avg

        forward_segments = calculate_rms_fluctuations(detrended, window_size, overlap)
        backward_segments = calculate_rms_fluctuations(detrended[::-1], window_size, overlap)

        rms_fluctuations = np.concatenate([forward_segments, backward_segments])

        F_q = []
        for q in q_values:
            if q == 0:
                F_q.append(np.exp(0.5 * np.mean(np.log(rms_fluctuations**2))))
            else:
                F_q.append((np.mean(rms_fluctuations**q))**(1/q))
        results[window_size] = F_q

    return results

def calculate_hurst_exponents(results, window_sizes, q_values):
    """Calculate the generalized Hurst exponents h(q)."""

    log_window_sizes = np.log(window_sizes)
    hurst_exponents = {}

    for q_index, q in enumerate(q_values):
        F_q_values = [results[ws][q_index] for ws in window_sizes]
        log_F_q = np.log(F_q_values)
        slope, _, _, _, _ = linregress(log_window_sizes, log_F_q)
        hurst_exponents[q] = slope
    return hurst_exponents

def calculate_multifractal_mass_exponent_taf(q_values, hurst_exponents):
    """Calculate the multifractal mass exponent taf(q)."""
    Df = 1  # Topological dimension for 1D signal
    tau_q = [q * h - Df for q, h in zip(q_values, hurst_exponents.values())]
    return tau_q

def calculate_singularity_strength_a(q_values, tau_q):
    """Calculate the singularity strength a(q), the derivative of tau(q)."""
    a_q = np.gradient(tau_q, q_values)
    return a_q

def calculate_multifractal_spectrum_fa(q_values, a_q, tau_q):
    """Calculate the multifractal spectrum f(a) from a(q)."""
    fa_q = [q * a - tau for q, a, tau in zip(q_values, a_q, tau_q)]
    return fa_q

def calculate_som(a_q):
    """Calculate the Strength of Multifractality (SOM)."""
    som = np.max(a_q) - np.min(a_q)
    return som

def calculate_dom(hurst_exponents):
    """Calculate the Degree of Multifractality (DOM)."""
    dom = np.max(list(hurst_exponents.values())) - np.min(list(hurst_exponents.values()))
    return dom

def calculate_dfs(fa_q):
    """Calculate the Difference of Multifractal Spectrum (DFS)."""
    dfs = np.abs(np.max(fa_q) - np.min(fa_q))
    return dfs

def calculate_pse(a_q, q_values):
    """Calculate the Peak Singularity Exponent (PSE), which is a(q) at q = -5."""
    if -5 in q_values:
        pse = a_q[q_values.index(-5)]
    else:
        pse = None
    return pse

if __name__ == "__main__":
    # Select files
    file_paths = select_files()

    # Ask for parameters
    root = tk.Tk()
    root.withdraw()
    segment_size_min = simpledialog.askinteger("Input", "Enter minimum segment size (number of samples):")
    segment_size_max = simpledialog.askinteger("Input", "Enter maximum segment size (number of samples):")
    segment_size_step = 100  # Default segment_size step
    segment_size_values = list(range(segment_size_min, segment_size_max + 1, segment_size_step))

    # Ask for q parameters
    q_min = simpledialog.askinteger("Input", "Enter minimum q value:")
    q_max = simpledialog.askinteger("Input", "Enter maximum q value:")
    q_step = 1  # Default q step
    q_values = list(range(q_min, q_max + 1, q_step))

    for path in file_paths:
        # Load signal
        with open(path, "r") as json_file:
            data = json.load(json_file)
        
        signal = np.array(data['signal'])
        preprocessed_signal = preprocess_time_series(signal)

        # Core analysis
        results = mfdma_core(preprocessed_signal, segment_size_values, q_values)
        print(f"Results for {path}:", results)

        # Calculate h(q) using the results from the MF-DMA analysis
        hurst_exponents = calculate_hurst_exponents(results, segment_size_values, q_values)
        print(f"Hurst Exponents for {path}:", hurst_exponents)

        # Calculate taf(q)
        tau_q = calculate_multifractal_mass_exponent_taf(q_values, hurst_exponents)
        print(f"Multifractal Mass Exponent tau(q) for {path}:", tau_q)

        # Calculate a(q)
        a_q = calculate_singularity_strength_a(q_values, tau_q)
        print(f"Singularity Strength a(q) for {path}:", a_q)

        # Calculate multifractal spectrum f(a)
        fa_q = calculate_multifractal_spectrum_fa(q_values, a_q, tau_q)
        print(f"Multifractal Spectrum f(a) for {path}:", fa_q)

        # Calculate SOM
        som = calculate_som(a_q)
        print(f"Strength of Multifractality (SOM) for {path}: {som}")

        # Calculate DOM
        dom = calculate_dom(hurst_exponents)
        print(f"Degree of Multifractality (DOM) for {path}: {dom}")

        # Calculate DFS
        dfs = calculate_dfs(fa_q)
        print(f"Difference of Multifractal Spectrum (DFS) for {path}: {dfs}")

        # Calculate PSE
        pse = calculate_pse(a_q, q_values)
        print(f"Peak Singularity Exponent (PSE) for {path}: {pse}")
