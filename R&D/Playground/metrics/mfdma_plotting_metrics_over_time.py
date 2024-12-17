import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import json
import matplotlib.pyplot as plt
from scipy.stats import linregress

def select_files():
    """Opens a dialog window to select files and returns their paths as a list."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Files")
    return list(file_paths)

def preprocess_time_series(time_series):
    """Preprocess the time series by calculating the cumulative sum and subtracting the mean."""
    cumulative_sum = np.cumsum(time_series)
    return cumulative_sum

def moving_average_with_overlap(time_series, window_size, overlap=0.5):
    """Calculate the moving average with overlapping windows."""
    step = int(window_size * (1 - overlap))
    moving_avg = [
        np.mean(time_series[i:i + window_size])
        for i in range(0, len(time_series) - window_size + 1, step)
    ]
    return np.array(moving_avg)

def residual_signal(preprocessed_signal, moving_avg):
    """Calculate the residual signal by subtracting the moving average from the preprocessed signal."""
    # Ensure both arrays have the same length before subtraction
    start_index = len(preprocessed_signal) - len(moving_avg)
    residual = preprocessed_signal[start_index:] - moving_avg
    return residual

def divide_into_segments(residual, n):
    """Divide the residual signal into non-overlapping segments."""
    N = len(residual)
    Nn = (N - n + 1) // n
    segments = [residual[i * n:(i + 1) * n] for i in range(Nn)]
    return segments

def calculate_local_rms(segment):
    """Calculate the RMS for a single segment."""
    return np.sqrt(np.mean(segment**2))

def calculate_global_rms(segments, q, Nn):
    """Calculate the global RMS for order q."""
    F_s_values = [np.sqrt(np.mean(segment**2)) for segment in segments]
    
    if q == 0:
        return np.exp(0.5 * np.mean(np.log(F_s_values)))
    else:
        return (np.mean([F**q for F in F_s_values]) ** (1 / q))
    
def power_law_fit(scales, F_q_values):
    """Performs power law fit (log-log scale) and returns the slope H(q)."""
    log_scales = np.log(scales)
    log_F_q = np.log(F_q_values)
    slope, intercept, _, _, _ = linregress(log_scales, log_F_q)
    return slope  # This is H(q)

def calculate_F_q(segments, q):
    """Calculate the F_q value for each segment."""
    F_q_values = []
    for segment in segments:
        # RMS calculation for the segment
        rms = np.sqrt(np.mean(segment**2))
        if q == 0:
            F_q_values.append(np.exp(0.5 * np.log(rms)))
        else:
            F_q_values.append(np.mean([rms ** q for _ in range(len(segment))]) ** (1 / q))
    return F_q_values

def process_signal(file_paths, window_sizes, q_min, q_max):
    """Process the signal data, calculate F_q, H_q, and plot the results."""
    q_values = list(range(q_min, q_max + 1))

    for path in file_paths:
        # Load signal from file
        with open(path, "r") as json_file:
            data = json.load(json_file)
        
        signal = np.array(data['signal'])
        
        for window_size in window_sizes:
            # Segment the signal based on the window size
            segments = [signal[i:i + window_size] for i in range(0, len(signal), window_size)]
            F_q_results = []
            
            # Calculate F_q for each q
            for q in q_values:
                F_q_values = calculate_F_q(segments, q)
                F_q_results.append(F_q_values)
            
            # Perform power-law fitting and calculate H(q)
            log_window_sizes = np.log(window_sizes)  # Use log window sizes as the scales
            hurst_exponents = {}

            for i, q in enumerate(q_values):
                F_q_values = F_q_results[i]
                H_q = power_law_fit(log_window_sizes, F_q_values)
                hurst_exponents[q] = H_q

            # Plot H(q) vs q for the current window size
            plt.figure(figsize=(8, 6))
            plt.plot(q_values, [hurst_exponents[q] for q in q_values], marker='o')
            plt.xlabel('q')
            plt.ylabel('H(q)')
            plt.title(f'Hurst Exponent H(q) for window size {window_size}')
            plt.grid(True)
            plt.show()

            # Now, calculate and plot the multifractal spectrum and singularity strength
            tau_q = [q * hurst_exponents[q] - 1 for q in q_values]  # tau(q) = q * H(q) - 1
            a_q = [tau_q_i - hurst_exponents[q] for q, tau_q_i in zip(q_values, tau_q)]  # singularity strength a(q)
            f_a_q = [q * a_q_i - tau_q_i for q, a_q_i, tau_q_i in zip(q_values, a_q, tau_q)]  # multifractal spectrum f(a(q))

            # Plot the multifractal spectrum
            plt.figure(figsize=(8, 6))
            plt.plot(a_q, f_a_q, marker='o')
            plt.xlabel('Singularity strength a(q)')
            plt.ylabel('Multifractal Spectrum f(a(q))')
            plt.title(f'Multifractal Spectrum for window size {window_size}')
            plt.grid(True)
            plt.show()

def calculate_tau_q(H_q, q_values):
    """Calculate the multifractal mass exponent τ(q)."""
    Df = 1  # Topological dimension for 1D signal
    tau_q = [q * H_q - Df for q in q_values]
    return tau_q

def calculate_singularity_strength_a(q_values, tau_q):
    """Calculate the singularity strength a(q), the derivative of tau(q)."""
    a_q = np.gradient(tau_q, q_values)
    return a_q

def calculate_multifractal_spectrum_fa(q_values, a_q, tau_q):
    """Calculate the multifractal spectrum f(a) from a(q)."""
    fa_q = [q * a - tau for q, a, tau in zip(q_values, a_q, tau_q)]
    return fa_q

def plot_results(q_values, a_q_first, fa_q_first, a_q_last, fa_q_last, hurst_exponents_first, hurst_exponents_last):
    """Plot the multifractal spectrum and H(q) curves."""
    # Multifractal spectrum f(a) vs. a(q)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(a_q_first, fa_q_first, label="First Segment", color="blue")
    plt.plot(a_q_last, fa_q_last, label="Last Segment", color="red")
    plt.xlabel("a(q)")
    plt.ylabel("f(a)")
    plt.title("Multifractal Spectrum f(a)")
    plt.legend()

    # H(q) vs. q
    plt.subplot(1, 2, 2)
    plt.plot(q_values, hurst_exponents_first, label="First Segment", color="blue")
    plt.plot(q_values, hurst_exponents_last, label="Last Segment", color="red")
    plt.xlabel("q")
    plt.ylabel("H(q)")
    plt.title("Hurst Exponents H(q)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Select files
    file_paths = select_files()

    # Ask for parameters
    root = tk.Tk()
    root.withdraw()
    window_sizes = simpledialog.askstring("Input", "Enter window sizes (comma-separated):")
    window_sizes = [int(size) for size in window_sizes.split(',')]

    q_min = simpledialog.askinteger("Input", "Enter minimum q value:")
    q_max = simpledialog.askinteger("Input", "Enter maximum q value:")

    # Process the signal data
    process_signal(file_paths, window_sizes, q_min, q_max)