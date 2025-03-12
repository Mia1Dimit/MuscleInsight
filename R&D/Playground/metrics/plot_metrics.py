import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import json
import matplotlib.pyplot as plt




def main():

    filepaths = open_dialog_and_select_multiple_files()
    for filepath in filepaths:
        with open(filepath, "r") as json_file:
            m = json.load(json_file)

        m['mnf_arv_ratio'] = -np.array(m['mnf_arv_ratio'])
        # m['emd_mdf1'] = -np.array(m['emd_mdf1'])
        # m['emd_mdf2'] = -np.array(m['emd_mdf2'])
        m.pop('person')



        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        # plt.title(f"{filepaths[0].split('/')[-1].split('_ID')[0]}")

        axs[0].plot((np.array(m["rms"])-min(m["rms"])) / (max(m["rms"])-min(m["rms"])), label="RMS", linewidth=0.85)        
        axs[0].plot((np.array(m["mnf_arv_ratio"])-min(m["mnf_arv_ratio"])) / (max(m["mnf_arv_ratio"])-min(m["mnf_arv_ratio"])), label="-MNF/ARV", linewidth=0.85)
        axs[0].plot((np.array(m["ima_diff"])-min(m["ima_diff"])) / (max(m["ima_diff"])-min(m["ima_diff"])), label="IMA Diff", linewidth=0.85)
        axs[0].legend()
        axs[0].grid()
        
        axs[1].plot((np.array(m["emd_mdf1"])-min(m["emd_mdf1"])) / (max(m["emd_mdf1"])-min(m["emd_mdf1"])), label="EMD MDF1", linewidth=0.85)
        axs[1].plot((np.array(m["emd_mdf2"])-min(m["emd_mdf2"])) / (max(m["emd_mdf2"])-min(m["emd_mdf2"])), label="EMD MDF2", linewidth=0.85)
        axs[1].legend()
        axs[1].grid()

        axs[2].plot((np.array(m["fluct_variance"])-min(m["fluct_variance"])) / (max(m["fluct_variance"])-min(m["fluct_variance"])), label="Fluct Variance", linewidth=0.85)
        axs[2].plot((np.array(m["fluct_range_values"])-min(m["fluct_range_values"])) / (max(m["fluct_range_values"])-min(m["fluct_range_values"])),   label="Fluct Range", linewidth=0.85)
        axs[2].plot((np.array(m["fluct_mean_diff_values"])-min(m["fluct_mean_diff_values"])) / (max(m["fluct_mean_diff_values"])-min(m["fluct_mean_diff_values"])), label="Fluct Mean Diff", linewidth=0.85)
        axs[2].legend()
        axs[2].grid()

        plt.xlabel("Samples")
        plt.tight_layout()
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



if  __name__ == "__main__":
    main()