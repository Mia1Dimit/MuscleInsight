import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import os

class FatigueCalculator:
    def __init__(self):
        self.mpf_values = []
        self.fatigue_start_index = 0
        self.fatigue_B_values = []
        self.lfc_ima_values = []  # Store IMA values for LFC
        self.hfc_ima_values = []  # Store IMA values for HFC
        self.iemg_initial = None

    def algorithm_B_fatigue(self, iemg_values, amplitudes, cutoff):
        self.iemg_values = iemg_values
        self.amplitudes = amplitudes
        self.fatigue_start_index = 0

        while len(self.iemg_values) > self.fatigue_start_index:
            iemg_current = self.iemg_values[self.fatigue_start_index]
            print("Number of IEMG value:", self.fatigue_start_index)

            if self.iemg_initial is None:
                self.iemg_initial = iemg_current
                print("Initial IEMG value:", self.iemg_initial)

            if iemg_current > self.iemg_initial:
                print("Algorithm B Triggered")
                start_ind = self.fatigue_start_index*400
                end_ind = self.fatigue_start_index*400 + 800

                # Obtain the low-frequency sub-signal (LFSS) and the high-frequency sub-signal (HFSS).
                lfc, hfc = self.butterworth_bandpass_filter(self.amplitudes[start_ind:end_ind], cutoff)

                # Perform FFT
                fft_lfc = fft(lfc)
                fft_hfc = fft(hfc)

                # Calculate IMA of the Low Frequency Component (LFC) and High Frequency Component (HFC)
                instantaneous_mean_amplitude_lfc = np.mean(np.abs(fft_lfc))
                instantaneous_mean_amplitude_hfc = np.mean(np.abs(fft_hfc))

                ima_avg = instantaneous_mean_amplitude_lfc + instantaneous_mean_amplitude_hfc
                ima_lfc = instantaneous_mean_amplitude_lfc / ima_avg
                ima_hfc = instantaneous_mean_amplitude_hfc / ima_avg

                self.lfc_ima_values.append(ima_lfc)
                self.hfc_ima_values.append(ima_hfc)

                # Calculate Fatigue Index
                fatigue_index = instantaneous_mean_amplitude_lfc - instantaneous_mean_amplitude_hfc
                self.fatigue_B_values.append(fatigue_index)
                print(f"Fatigue Index: {fatigue_index}")
            self.fatigue_start_index += 1

    def butterworth_bandpass_filter(self, signal, cutoff):
        def butter_bandpass(lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        # Filter 1: 25-79Hz
        lowcut1, highcut1 = 25.0, cutoff
        b1, a1 = butter_bandpass(lowcut1, highcut1, fs=800.0)
        lfc = filtfilt(b1, a1, signal)

        # Filter 2: 80-350Hz
        lowcut2, highcut2 = cutoff, 350.0
        b2, a2 = butter_bandpass(lowcut2, highcut2, fs=800.0)
        hfc = filtfilt(b2, a2, signal)

        return lfc, hfc

csv.field_size_limit(1000000)
# Load data from the CSV file
file_path = "C:\\Users\\DeLL\\Documents\\Arduino\\thesis_BioAmp\\Thesis-2\\final_data\\alexia_leg_extension3.csv"
data = {}

with open(file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        array_name = row[0]
        array_values = list(map(float, row[1].split(',')))
        data[array_name] = array_values

# Extract individual measures
amplitudes = data["amplitudes"]
rms_values = data["rms_values"]
iemg_values = data["iemg_values"]
mnf_values = data["mnf_values"]
mpf_values = data["mpf_values"]

# Create separate subplots for RMS, IEMG, MNF, and MPF
fig, axs = plt.subplots(4, 1, figsize=(10, 16))

axs[0].plot(amplitudes, label='Amplitudes', color='blue')
axs[0].set_ylabel('Amplitudes')
axs[0].set_title('Amplitudes')

# Plot RMS values
axs[1].plot(rms_values, label='RMS', color='blue')
axs[1].set_ylabel('RMS Values')
axs[1].set_title('RMS Values')

# Plot IEMG values
axs[2].plot(iemg_values, label='IEMG', color='orange')
axs[2].set_ylabel('IEMG Values')
axs[2].set_title('IEMG Values')

# Plot MNF and MPF values together
axs[3].plot(mnf_values, label='MNF', color='green')
axs[3].plot(mpf_values, label='MPF', color='red')
axs[3].set_ylabel('MNF/MPF Values')
axs[3].set_xlabel('Sample Index')
axs[3].set_title('MNF and MPF Values')
axs[3].legend()

# Initialize FatigueCalculator for Algorithm B
fatigue_calculator_B = FatigueCalculator()

# Implement fatigue calculation algorithm B for each cutoff frequency and save results to CSV
cutoff_frequency_B = 80.0  # Only for 80Hz
fatigue_calculator_B.algorithm_B_fatigue(iemg_values, amplitudes, cutoff_frequency_B)

# Save fatigue B results to CSV
mode = 'a' if os.path.exists(file_path) else 'w'
with open(file_path, mode, newline='') as csvfile:
    writer = csv.writer(csvfile)
    if mode == 'w':
        writer.writerow(['id', 'fatigue_B_values'])
    #writer.writerow(['fatigue_B_values',','.join(map(str, fatigue_calculator_B.fatigue_B_values))])

# Adjust layout for better spacing
plt.tight_layout()

# Show the subplots
plt.show()

print(f"Fatigue B results appended to {file_path}")
