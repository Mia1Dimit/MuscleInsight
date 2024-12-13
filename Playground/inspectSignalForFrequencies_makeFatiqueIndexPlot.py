import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, TextBox
from scipy import stats
from time import sleep
from scipy.signal import resample
import pandas as pd
from scipy.signal import butter, lfilter, freqz
import sounddevice as sd
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog
import json
from scipy.signal import welch, hilbert
from scipy.stats import entropy
from pywt import wavedec
from math import log2


INITIAL_RATE = 800.0

USE_FILTER = False  # Set to True to apply a bandpass filter
USE_BANDPASS_FILTER = False
lowcut = 1.0  # Low frequency of the bandpass filter
highcut = 20.0  # High frequency of the bandpass filter
order = 2  # Order of the bandpass filter


fft_window_size = 1024*1  
fft_step_size = fft_window_size//2 
metrics_window_sec = 1.5  # Length of the metrics window in seconds

target_sound_sampling_rate = 44100  # 44.1 kHz



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
    fig, ax = plt.subplots(figsize=(15, 8))
    
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

def get_sampling_rate_acc_fromSD(timestamps_sd):
    """Specific func to get sampling rate from acc data stored in SD card. Needs specific timestamps from SD."""

    if len(timestamps_sd) < 2:
        return 0  # Not enough timestamps to calculate sampling rate

    # Calculate the time difference between the first two timestamps
    time_diffs = [ (timestamps_sd[i+1] - timestamps_sd[i]).total_seconds() for i in range(len(timestamps_sd) - 1)]
    avg_time_diff = sum(time_diffs)/len(time_diffs)
    # avg_time_diff = (timestamps_sd[1] - timestamps_sd[0]).total_seconds()

    # Each chunk corresponds to 57600 / 6 samples
    num_samples_per_chunk = 57600 / 6
    sampling_rate = num_samples_per_chunk / avg_time_diff
    print(f"Sampling Rate: {sampling_rate}")

    return sampling_rate

def onselect(xmin, xmax):
    # Function to update the selected region indices
    print(f"Selected time range: {xmin} to {xmax}\nDuration: {xmax - xmin}")
    global start_idx, end_idx
    start_idx = int(xmin * len(input_signal) / time[-1])
    end_idx = int(xmax * len(input_signal) / time[-1])

def plot_selected_interval(event):
    # Function to plot the selected interval

    if start_idx is not None and end_idx is not None:
        # fig2, ax2 = plt.subplots()
        # ax2.plot(time[start_idx:end_idx], input_signal[start_idx:end_idx], color='turquoise')
        # ax2.set_title('Selected Interval')
        # ax2.set_xlabel('Time [s]')
        # ax2.set_ylabel('Amplitude')

        # fft = m.get_fft_values(input_signal[start_idx:end_idx])
        # bins = m.get_fft_bins(input_signal[start_idx:end_idx], Fs)
        # fig3, ax3 = plt.subplots()
        # ax3.plot(bins, fft, color='red')
        # ax3.set_title('Selected Interval')
        # ax3.set_xlabel('Freq [Hz]')
        # ax3.set_ylabel('Amplitude')


        window_timestamps, metrics, scaling_factors = analyze_vibration_signal(input_signal[start_idx:end_idx], time[start_idx:end_idx], Fs, metrics_window_sec)
        plot_interactive_metrics(window_timestamps, metrics, scaling_factors, metrics_window_sec)


        plt.show()

def lowpass_filter(data, highcut, Fs, order=5):
    """
    Applies a low-pass Butterworth filter to the input signal and plots the transfer function.
    
    Parameters:
    - data: array-like, the signal data to filter
    - highcut: float, the cutoff frequency of the filter (in Hz)
    - Fs: float, the sampling frequency of the signal (in Hz)
    - order: int, the order of the Butterworth filter (default is 5)

    Returns:
    - y: array-like, the filtered signal
    """
    # Define the normalized cutoff frequency
    nyquist = 0.5 * Fs
    normal_cutoff = highcut / nyquist
    
    # Design the Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter to the data
    y = lfilter(b, a, data)
    
    # Plot the transfer function
    w, h = freqz(b, a, worN=8000)
    plt.plot((Fs * 0.5 / np.pi) * w, abs(h), 'b')
    plt.plot(highcut, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(highcut, color='k')
    plt.xlim(0, 0.5 * Fs)
    plt.title("Low-pass Filter Frequency Response")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid()
    # plt.show()
    
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)

    # Get the transfer function
    w, h = bandpass_transfer_function(lowcut, highcut, fs, order)
    # Plot the transfer function (magnitude response)
    plt.plot(0.5 * Fs * w / np.pi, np.abs(h), 'b')
    plt.title("Bandpass Filter Transfer Function")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    plt.grid()

    return y

def bandpass_transfer_function(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    w, h = freqz(b, a, worN=8000)
    return w, h

def compute_magnitude(arr):
    # Calculate the magnitude along axis 0
    magnitude = np.sqrt(arr[0]**2 + arr[1]**2 + arr[2]**2)
    return magnitude

def mean_frequency(signal, sampling_rate):
    # Compute Power Spectral Density (PSD)
    freqs, psd = welch(signal, fs=sampling_rate)
    
    # Calculate the Mean Frequency
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    return mean_freq

def upsample_signal(input_signal, target_sampling_rate, Fs):
    num_samples = int(len(input_signal) * target_sampling_rate / Fs)
    upsampled_signal = resample(input_signal, num_samples)

    # Normalize the signal to the range -1.0 to 1.0 (for maximum volume)
    upsampled_signal = np.array(upsampled_signal)
    upsampled_signal = upsampled_signal / np.max(np.abs(upsampled_signal))

    # Plot the upsampled signal
    time = np.arange(num_samples) / target_sampling_rate  # Time array for plotting

    return upsampled_signal, time

def play_sound(event):
    sd.stop()
    if start_idx == end_idx:
        return
    signal, timestamps = upsample_signal(input_signal[start_idx:end_idx], target_sound_sampling_rate, Fs)

    # Play the signal
    sd.play(signal, target_sound_sampling_rate)
    # sd.wait()  # Wait until the sound has finished playing

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

def progressive_fft_plot(signal):
    # Setup frequencies for the FFT (only positive frequencies)
    frequencies = fftfreq(fft_window_size, 1 / Fs)[:fft_window_size // 2]
    index_25 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 25))
    index_80 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 80))
    index_350 = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - 350))

    imas = []
    
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

        # Increment the index by step_size
        idx += fft_step_size
    return imas

def live_FFT(event):
    if start_idx == end_idx:
        return
    
    if start_idx-fft_window_size < 0:
        start_fft = 0
    else:
        start_fft = start_idx-fft_window_size

    imas = progressive_fft_plot(input_signal[start_fft:end_idx])
    
    plt.figure()
    plt.plot(imas)
    plt.title(f"{filepaths[0].split('/')[-1].split('_ID')[0]}")
    plt.show()

    

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


#########################################################


filepaths = open_dialog_and_select_multiple_files()
data, fs_list = [], []

for filepath in filepaths:
    with open(filepath, "r") as json_file:
        newdata = json.load(json_file)
        data.extend(newdata["signal"])

Fs = INITIAL_RATE
data_formation = np.array(data)

if USE_FILTER:
    if USE_BANDPASS_FILTER:
        input_signal = bandpass_filter(data_formation, lowcut, highcut, Fs, order=order)
    else:
        input_signal = lowpass_filter(data_formation, highcut, Fs, order=order)
else:
    input_signal = data_formation

num_samples = len(input_signal)



# Plot the upsampled signal
print('Plotting time..')
time = np.arange(num_samples) / Fs  # Time array for plotting


# Variables to store the selected region
start_idx = None
end_idx = None

# Create the main figure
fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(time, input_signal)
ax.set_title(f"{filepaths[0].split('/')[-1]}")
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')

# Create a SpanSelector to select the interval
# span = SpanSelector(ax, onselect, 'horizontal', useblit=True)
span_selector = SpanSelector(
        ax,
        onselect,
        'horizontal',
        useblit=True,
        props=dict(alpha=0.2, facecolor='blue'),
        interactive=True,
        drag_from_anywhere=True
    )

# Create a button to plot the selected interval
ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])  # x, y, width, height
button = Button(ax_button, 'Plot Interval')
button.on_clicked(plot_selected_interval)

# Create a button to plot the selected interval
ax_button_play = plt.axes([0.7, 0.02, 0.1, 0.04])  # x, y, width, height
button_play = Button(ax_button_play, 'Play Sound')
button_play.on_clicked(play_sound)

ax_button_live_FFT = plt.axes([0.6, 0.02, 0.1, 0.04])  # x, y, width, height
button_live_FFT = Button(ax_button_live_FFT, 'Live FFT')
button_live_FFT.on_clicked(live_FFT)

# Add a TextBox to input a number
ax_text_box = plt.axes([0.5, 0.02, 0.05, 0.04])  # Position for the TextBox
text_box = TextBox(ax_text_box, 'Metric Window (s)', initial=str(metrics_window_sec))


def submit_number(text):
    try:
        global metrics_window_sec
        metrics_window_sec = float(text)
        print(f"Number entered: {metrics_window_sec}")
    except ValueError:
        print("Please enter a valid number.")

# Link the submit function to the TextBox
text_box.on_submit(submit_number)
plt.show()