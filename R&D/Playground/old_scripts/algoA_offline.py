import csv
import matplotlib.pyplot as plt
import numpy as np
import os

class FatigueCalculator:
    def __init__(self):
        self.mpf_values = []
        self.baseline_initialized = False
        self.baseline = 0
        self.fatigue_start_index = 0
        self.fatigue_A_values = []

    def algorithm_A_fatigue(self):
        if len(self.mpf_values) >= 6 + self.fatigue_start_index:  
            if not self.baseline_initialized:
                self.baseline = np.mean(self.mpf_values[:3])
                print("Baseline value:", self.baseline)
                self.baseline_initialized = True

            self.fatigue_start_index += 3
            recent_mpf_values = self.mpf_values[-3:]
            average_mpf = np.mean(recent_mpf_values)

            if average_mpf >= self.baseline:
                self.baseline = average_mpf
            
            fatigue_level = ((self.baseline - average_mpf) / self.baseline) * 100
            self.fatigue_A_values.append(fatigue_level)
            print(f"Fatigue Level: {average_mpf}")

csv.field_size_limit(10000000)
# Load data from the CSV file
file_path = "C:\\Users\\DeLL\\Documents\\Arduino\\thesis_BioAmp\\Thesis-2\\final_data\\vasilis_leg_extension3.csv"
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
fig, axs = plt.subplots(5, 1, figsize=(10, 16))

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

# Initialize FatigueCalculator
fatigue_calculator = FatigueCalculator()

# Implement fatigue calculation algorithm and save results to CSV
for mpf_value in mpf_values:
    fatigue_calculator.mpf_values.append(mpf_value)
    fatigue_calculator.algorithm_A_fatigue()

axs[4].plot(fatigue_calculator.fatigue_A_values, label='Fatigue A', color='purple', marker='o')
axs[4].set_ylabel('Fatigue A')
axs[4].set_title('Fatigue A')

# Save fatigue results to CSV
mode = 'a' if os.path.exists(file_path) else 'w'
with open(file_path, mode, newline='') as csvfile:
    writer = csv.writer(csvfile)
    if mode == 'w':
        writer.writerow(['id', 'fatigue_A_values'])
    writer.writerow(['fatigue_A_values',",".join(map(str, fatigue_calculator.fatigue_A_values))])

# Adjust layout for better spacing
plt.tight_layout()

# Show the subplots
plt.show()

print(f"Fatigue results appended to {file_path}")
