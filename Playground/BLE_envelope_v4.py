import asyncio
from bleak import BleakClient
import struct
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import keyboard
from pymongo import MongoClient
from datetime import datetime
import signal
import sys

#BioAmp EXG Pill
device_address = 'f4:12:fa:63:47:29'

#Myoware 2.0
#device_address = 'f4:12:fa:63:c2:2d'
filtered_characteristic_uuid = "a3b1a544-8794-11ee-b9d1-0242ac120002"

#MongoDB
uri = "mongodb+srv://jimiaoul:jimakos25@cluster0.rxwufw8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client["EMGdata"]
PersonalDetails = db["PersonalDetails"]
TestDetails = db["TestDetails"]


class CtrlBPressed(Exception):
    pass


class TerminateSignalReceived(Exception):
    pass


def signal_handler(sig, frame):
    raise CtrlBPressed

signal.signal(signal.SIGINT, signal_handler)

class RealTimePlotter:
    def __init__(self, window_size=1000):
        self.received_data = bytearray()
        self.amplitudes = []
        self.envelope_values = []  
        self.rms_values = []
        self.iemg_values = []
        self.mnf_values = []
        self.mpf_values = []
        self.fatigue_A_values = []
        self.fatigue_B_values = []
        self.sample_count = 0
        self.window_size = window_size
        self.start_index = 0
        self.end_index = 800
        self.baseline_initialized = False
        self.iemg_initial = None
        self.fatigueA_start_index = 0
        self.fatigueB_start_index = 0
        self.lfc_ima_values = []
        self.hfc_ima_values = []
        self.mnf_linear_fit = []
        self.mpf_linear_fit = []
        self.mnf_slope = []
        self.mpf_slope = []
        self.packet_index = 0


        # First figure for EMG signal and its envelope
        plt.ion()
        self.fig_emg, ( self.ax_emg, self.ax_rms, self.ax_iemg, self.ax_mnf) = plt.subplots(4)
        self.line_emg, = self.ax_emg.plot([], [], 'b-', label='EMG Data')
        self.envelope_line_emg, = self.ax_emg.plot([], [], 'r-', label='Envelope')
        self.line_rms, = self.ax_rms.plot([], [], 'g-', label='RMS')
        self.line_iemg, = self.ax_iemg.plot([], [], 'm-', label='IEMG')
        self.line_mnf, = self.ax_mnf.plot([], [], 'r-', label='MNF')
        self.line_mpf, = self.ax_mnf.plot([], [], 'g-', label='MPF')
        self.line_mnf_linear_fit, = self.ax_mnf.plot([], [], 'y--', label='LR_mnf')  
        self.line_mpf_linear_fit, = self.ax_mnf.plot([], [], 'm--', label='LR_mpf')  

        # Set labels and legends
        self.ax_emg.set_xlabel('Time (sec)')
        self.ax_emg.set_ylabel('EMG Amplitude')
        #self.ax_emg.set_title('Real-time EMG Data')
        self.ax_emg.set_ylim(-250.0, 250.0)
        #self.ax_emg.set_xlim(0, self.window_size)
        self.ax_emg.legend()

        self.ax_rms.set_xlabel('Time (sec)')
        self.ax_rms.set_ylabel('RMS Value')
        #self.ax_rms.set_title('RMS')
        self.ax_rms.set_ylim(0.0, 100.0)
        self.ax_rms.legend()

        self.ax_iemg.set_xlabel('Time (sec)')
        self.ax_iemg.set_ylabel('IEMG Value')
        #self.ax_iemg.set_title('IEMG')
        self.ax_iemg.set_ylim(0.0, 50000.0)
        self.ax_iemg.legend()

        # Set labels and legends for the frequency plot
        self.ax_mnf.set_xlabel('Time (sec)')
        self.ax_mnf.set_ylabel('Frequency (Hz)')
        #self.ax_mnf.set_title('MNF/MPF')
        self.ax_mnf.set_ylim(20.0, 100.0)
        self.ax_mnf.legend()

        self.buffer_size = 208
        self.circular_buffer = [0] * self.buffer_size
        self.sum_buffer = 0
        self.data_index = 0

        # Add a new figure for plotting fatigue values
        self.fig_fatigue, (self.ax_fatigue_A, self.ax_fatigue_B) = plt.subplots(2)
        self.line_fatigue_A, = self.ax_fatigue_A.plot([], [], 'b-', label='Fatigue A')
        self.line_fatigue_B, = self.ax_fatigue_B.plot([], [], 'r-', label='Fatigue B')

        # Set labels and legends for fatigue plots
        self.ax_fatigue_A.set_xlabel('Time (sec)')
        self.ax_fatigue_A.set_ylabel('Fatigue A Level')
        self.ax_fatigue_A.set_ylim(-20.0, 120.0)
        self.ax_fatigue_A.legend()

        self.ax_fatigue_B.set_xlabel('Time (sec)')
        self.ax_fatigue_B.set_ylabel('Fatigue B Level')
        self.ax_fatigue_B.set_ylim(-20.0, 350.0)
        self.ax_fatigue_B.legend()

    def ctrl_b_handler(self):
        if keyboard.is_pressed('ctrl+b'):
            raise CtrlBPressed
        
    def set_data_and_plot(self):
        try:
            self.ctrl_b_handler()
            # Update EMG signal and envelope plot
            self.line_emg.set_data(range(0, len(self.amplitudes)), self.amplitudes[0:len(self.amplitudes)])        
            self.envelope_line_emg.set_data(range(0, len(self.envelope_values)), self.envelope_values[0:len(self.envelope_values)])
            self.ax_emg.set_xlim(0, self.sample_count)

            # Update RMS, IEMG plots
            self.line_rms.set_data(range(0, len(self.rms_values)), self.rms_values[0:len(self.rms_values)])
            self.ax_rms.set_xlim(0, len(self.rms_values))

            self.line_iemg.set_data(range(0, len(self.iemg_values)), self.iemg_values[0:len(self.iemg_values)])
            self.ax_iemg.set_xlim(0, len(self.iemg_values))

            self.line_mnf.set_data(range(0, len(self.mnf_values)), self.mnf_values[0:len(self.mnf_values)])
            self.line_mpf.set_data(range(0, len(self.mpf_values)), self.mpf_values[0:len(self.mpf_values)])
            self.line_mnf_linear_fit.set_data(range(0, len(self.mnf_values)), np.polyval(self.mnf_linear_fit, range(len(self.mnf_values))))
            self.line_mpf_linear_fit.set_data(range(0, len(self.mpf_values)), np.polyval(self.mpf_linear_fit, range(len(self.mpf_values))))
            self.ax_mnf.set_xlim(0, len(self.mnf_values))

            # Update fatigue plots
            self.line_fatigue_A.set_data(range(0, len(self.fatigue_A_values)), self.fatigue_A_values[0:len(self.fatigue_A_values)])
            self.line_fatigue_B.set_data(range(0, len(self.fatigue_B_values)), self.fatigue_B_values[0:len(self.fatigue_B_values)])
            
            # Set xlim for fatigue plots
            self.ax_fatigue_A.set_xlim(0, len(self.fatigue_A_values))
            self.ax_fatigue_B.set_xlim(0, len(self.fatigue_B_values))

            plt.pause(0.01)
        except CtrlBPressed:
            print("Ctrl+B pressed. Disconnecting device and plotting final data.")
            raise
    
    def plot_final_data(self):
        self.line_emg.set_data(range(0, len(self.amplitudes)), self.amplitudes[0:len(self.amplitudes)])        
        self.envelope_line_emg.set_data(range(0, len(self.envelope_values)), self.envelope_values[0:len(self.envelope_values)])
        self.ax_emg.set_xlim(0, self.sample_count)
        
        self.line_rms.set_data(range(0, len(self.rms_values)), self.rms_values[0:len(self.rms_values)])
        self.ax_rms.set_xlim(0, len(self.rms_values))

        self.line_iemg.set_data(range(0, len(self.iemg_values)), self.iemg_values[0:len(self.iemg_values)])
        self.ax_iemg.set_xlim(0, len(self.iemg_values))

        self.line_mnf.set_data(range(0, len(self.mnf_values)), self.mnf_values[0:len(self.mnf_values)])
        self.line_mpf.set_data(range(0, len(self.mpf_values)), self.mpf_values[0:len(self.mpf_values)])
        self.line_mnf_linear_fit.set_data(range(0, len(self.mnf_values)), np.polyval(self.mnf_linear_fit, range(len(self.mnf_values))))
        self.line_mpf_linear_fit.set_data(range(0, len(self.mpf_values)), np.polyval(self.mpf_linear_fit, range(len(self.mpf_values))))
        self.ax_mnf.set_xlim(0, len(self.mnf_values))

        self.line_fatigue_A.set_data(range(0, len(self.fatigue_A_values)), self.fatigue_A_values[0:len(self.fatigue_A_values)])
        self.line_fatigue_B.set_data(range(0, len(self.fatigue_B_values)), self.fatigue_B_values[0:len(self.fatigue_B_values)])
        self.ax_fatigue_A.set_xlim(0, len(self.fatigue_A_values))
        self.ax_fatigue_B.set_xlim(0, len(self.fatigue_B_values))

        with open('randommm_leg_extension4.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ArrayID", "Array"])
            # Write each array to a separate row with an identifier
            writer.writerow(["amplitudes", ",".join(map(str, self.amplitudes))])
            writer.writerow(["rms_values", ",".join(map(str, self.rms_values))])
            writer.writerow(["iemg_values", ",".join(map(str, self.iemg_values))])
            writer.writerow(["mnf_values", ",".join(map(str, self.mnf_values))])
            writer.writerow(["mpf_values", ",".join(map(str, self.mpf_values))])
            writer.writerow(["fatigue_A_values", ",".join(map(str, self.fatigue_A_values))])
            writer.writerow(["fatigue_B_values", ",".join(map(str, self.fatigue_B_values))])
            writer.writerow(["mnf_slope", ",".join(map(str, self.mnf_slope))])
            writer.writerow(["mpf_slope", ",".join(map(str, self.mpf_slope))])
        
        #post = {
        #    "date_of_test": datetime.now().strftime("%d-%m-%Y"),
        #    "fatigue_A_values": self.fatigue_A_values,
        #    "fatigue_B_values": self.fatigue_B_values,
        #    "mnf_values": self.mnf_values,
        #    "mpf_values": self.mpf_values
        #}

        #TestDetails.insert_one(post)

        plt.show()
    
    def update_received_data(self, data):
        self.received_data.extend(data)
        self.packet_index += len(data)
        #print("Length of received data:", len(data))
        #print("Length of received data:", self.packet_index)

        # Calculate the maximum number of floats that can be processed
        max_floats = len(self.received_data) // 4
        floats_to_process = max_floats * 4  # Calculate the actual bytes to process

        if floats_to_process > 0:
            floats_data = self.received_data[:floats_to_process]
            self.received_data = self.received_data[len(self.received_data):]            
            floats = [struct.unpack('f', floats_data[i:i+4])[0] for i in range(0, floats_to_process, 4)]
            #print("Float values:", floats)  # Print the extracted float values
            self.amplitudes.extend(floats)
            self.sample_count = len(self.amplitudes)

            for val in floats:
                self.envelope_values.append(self.calculate_envelope(abs(val)))
                #print("Float value:", val)

            #print("Length of amplitudes:", len(self.amplitudes))
            #print("Length of envelope_values:", len(self.envelope_values))

    def features_calculation(self):
        step = 400
        if (len(self.amplitudes) >= 800 + self.start_index):                
            epoch_data = self.amplitudes[self.start_index:self.end_index]
            self.rms_values.append(self.calculate_rms(epoch_data))
            self.iemg_values.append(self.calculate_iemg(epoch_data))

            self.algorithm_B_fatigue()
            self.algorithm_A_fatigue()

            fft_result = fft(epoch_data)
            freq_values = fftfreq(len(epoch_data), 1 / 800)
            
            # Select only positive frequencies and corresponding FFT values
            positive_freq_mask = (freq_values >= 0) & (freq_values <= 350)
            positive_freq_values = freq_values[positive_freq_mask]
            positive_fft_values = np.abs(fft_result)[positive_freq_mask]
            
            self.mnf_values.append(self.calculate_mnf(positive_freq_values, positive_fft_values))
            self.mpf_values.append(self.calculate_mpf(positive_freq_values, positive_fft_values))

            if len(self.mnf_values) >= 5:
                self.mnf_linear_fit = np.polyfit(range(len(self.mnf_values)), self.mnf_values, 1)
                self.mpf_linear_fit = np.polyfit(range(len(self.mpf_values)), self.mpf_values, 1)
                # Extract slopes
                self.mnf_slope.append(self.mnf_linear_fit[0])
                self.mpf_slope.append(self.mpf_linear_fit[0])

            self.start_index +=step
            self.end_index += self.start_index

    def calculate_envelope(self, abs_emg_value):
        self.sum_buffer -= self.circular_buffer[self.data_index]
        self.sum_buffer += abs_emg_value
        self.circular_buffer[self.data_index] = abs_emg_value
        self.data_index = (self.data_index + 1) % self.buffer_size
        return (self.sum_buffer / self.buffer_size) * 2        

    def calculate_rms(self,rms_val):
        return np.sqrt(np.mean(np.square(rms_val)))         

    def calculate_iemg(self,iemg_values):
        return np.sum(np.abs(iemg_values)) 
    
    def calculate_mnf(self, freq_values, fft_result):
        mnf = np.sum(freq_values * np.abs(fft_result)**2) / np.sum(np.abs(fft_result)**2)
        return mnf

    def calculate_mpf(self, freq_values, fft_result):
        power_spectrum = np.abs(fft_result)**2
        total_power = np.sum(power_spectrum)
        normalized_cumulative_power = np.cumsum(power_spectrum) / total_power
        mpf_index = np.argmax(normalized_cumulative_power >= 0.5)
        mpf = freq_values[mpf_index]
        return mpf

    def algorithm_A_fatigue(self):
        if len(self.mpf_values) >= 6 + self.fatigueA_start_index:  
            if not self.baseline_initialized:
                self.baseline = np.mean(self.mpf_values[:3])
                print("Baseline value:", self.baseline)
                self.baseline_initialized = True

            self.fatigueA_start_index += 3
            recent_mpf_values = self.mpf_values[-3:]
            average_mpf = np.mean(recent_mpf_values)

            if average_mpf >= self.baseline:
                self.baseline = average_mpf
            
            fatigue_level = ((self.baseline - average_mpf) / self.baseline) * 100
            self.fatigue_A_values.append(fatigue_level)
            #print(f"Fatigue Level: {fatigue_level}")

    def algorithm_B_fatigue(self):
        if (len(self.iemg_values) >= self.fatigueB_start_index):  
            iemg_current = self.iemg_values[self.fatigueB_start_index]
            
            if self.iemg_initial is None:
                self.iemg_initial = iemg_current
                #print("Initial IEMG value:", self.iemg_initial)

            if iemg_current > self.iemg_initial:
                #print("Algorithm B Triggered")
                start_ind = self.fatigueB_start_index*400
                end_ind = self.fatigueB_start_index*400 + 800
                lfc, hfc = self.butterworth_bandpass_filter(self.amplitudes[start_ind:end_ind])

                #Perform FFT
                fft_lfc = fft(lfc)
                fft_hfc = fft(hfc)

                #Calculate IMA of the Low Frequency Component (LFC) and High Frequency Component (HFC)
                instantaneous_mean_amplitude_lfc = np.mean(np.abs(fft_lfc))
                instantaneous_mean_amplitude_hfc = np.mean(np.abs(fft_hfc))

                ima_avg = instantaneous_mean_amplitude_lfc + instantaneous_mean_amplitude_hfc
                ima_lfc = instantaneous_mean_amplitude_lfc / ima_avg
                ima_hfc = instantaneous_mean_amplitude_hfc / ima_avg

                self.lfc_ima_values.append(ima_lfc)
                self.hfc_ima_values.append(ima_hfc)

                # Step 4: Calculate Fatigue Index
                fatigue_index = instantaneous_mean_amplitude_lfc - instantaneous_mean_amplitude_hfc
                self.fatigue_B_values.append(fatigue_index)
                #print(f"Fatigue Index: {fatigue_index}")
            self.fatigueB_start_index += 1

    def butterworth_bandpass_filter(self, signal):
        def butter_bandpass(lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        # Filter 1: 25-79Hz
        lowcut1, highcut1 = 25.0, 79.0
        b1, a1 = butter_bandpass(lowcut1, highcut1, fs=800.0)
        lfc = filtfilt(b1, a1, signal)

        # Filter 2: 80-350Hz
        lowcut2, highcut2 = 80.0, 350.0
        b2, a2 = butter_bandpass(lowcut2, highcut2, fs=800.0)
        hfc = filtfilt(b2, a2, signal)

        return lfc, hfc


async def notification_handler_wrapper(sender, data, plotter):
    await notification_handler(sender, data, plotter)

async def notification_handler(sender, data, plotter):
    plotter.update_received_data(data)

async def plot_handler(plotter):
    while True:
        plotter.set_data_and_plot()
        plotter.features_calculation()        
        await asyncio.sleep(0.1)

async def main():
    plotter = RealTimePlotter()

    client = BleakClient(device_address)

    try:
        await client.connect()
        print(f"Connected to device with MAC address: {client.address}")

        # Enable notifications for the filtered characteristic
        await client.start_notify(filtered_characteristic_uuid, lambda sender, data: asyncio.ensure_future(notification_handler_wrapper(sender, data, plotter)))

        # Concurrently run notification handling, plotting, and calculations
        await asyncio.gather(plot_handler(plotter), asyncio.sleep(0))

    except CtrlBPressed:
        print("Ctrl+B pressed. Disconnecting device and plotting final data.")
        plotter.plot_final_data()
        await client.disconnect() 
        plt.show(block=True)       
        sys.exit(0)
    except TerminateSignalReceived:
        print("Signal received. Plotting final data and terminating script.")
        plotter.plot_final_data()
        await client.disconnect()
        sys.exit(0)
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Plotting final data and terminating script.")
        plotter.plot_final_data()
        await client.disconnect()
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        await client.disconnect()
        sys.exit(1)
    finally:
        await client.disconnect()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n\n *** Interrupted.\n')
