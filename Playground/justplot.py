import csv
import matplotlib.pyplot as plt
import numpy as np
import os

<<<<<<< HEAD
file_path = r'C:\Dimitris\Thesis\final_data\thumios_leg_extension2.csv'
data = {}
csv.field_size_limit(1000000)
=======
# Define folder path containing the CSV files
folder_path = r'C:\Users\DeLL\Documents\Arduino\thesis_BioAmp\Thesis-2\final_data'
>>>>>>> a7ca2c4d5d4975c1c7679476a27ef8ed5eb151e3

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # Process only CSV files
        file_path = os.path.join(folder_path, file_name)
        data = {}
        csv.field_size_limit(1000000)
        
        # Read CSV data
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
        fatigue_A_values = data["fatigue_A_values"]
        fatigue_B_values = data["fatigue_B_values"]

        # Create subplots for measures
        fig, axs = plt.subplots(4, 1, figsize=(8, 8))

        # Plot amplitudes
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

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the first plot
        plot1_path = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}_plot1.png")
        plt.savefig(plot1_path)
        plt.close(fig)

        # Plot Fatigue A and Fatigue B values together
        plt.figure(figsize=(6, 6))

        # Plot Fatigue A
        plt.subplot(2, 1, 1)
        plt.plot(fatigue_A_values, label='Fatigue A', color='blue')
        plt.xlabel('Sample Index')
        plt.ylabel('Fatigue A Values')
        plt.title('Fatigue A Values')

        # Plot Fatigue B with polynomial fit and moving mean
        plt.subplot(2, 1, 2)
        plt.plot(fatigue_B_values, label='Fatigue B', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel('Fatigue B Values')
        plt.title('Fatigue B Values')

        # Calculate polynomial fit for Fatigue B
        poly_fit_B = np.polyfit(range(len(fatigue_B_values)), fatigue_B_values, deg=3)
        poly_fit_values_B = np.polyval(poly_fit_B, range(len(fatigue_B_values)))

        # Calculate moving mean for Fatigue B
        window_size = 4  
        moving_mean_B = np.convolve(fatigue_B_values, np.ones(window_size) / window_size, mode='valid')

        # Plot polynomial fit and moving mean for Fatigue B
        plt.plot(poly_fit_values_B, label='Fatigue B Polynomial Fit', linestyle='--', color='purple')
        plt.plot(range(window_size - 1, len(fatigue_B_values)), moving_mean_B, label='Fatigue B Moving Mean', linestyle='--', color='pink')

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the second plot
        plot2_path = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}_plot2.png")
        plt.savefig(plot2_path)
        plt.close()
