import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import json
import matplotlib.pyplot as plt




def main():

    all_weightedsum_unfilt = []
    all_weightedsum_filt = []
    all_indexcalc_unfilt = []
    all_indexcalc_filt = []
    all_pca_unfilt = []
    all_pca_filt = []
    all_tsne_unfilt = []
    all_tsne_filt = []
    file_end_indices = []  # Store indices where each file ends

    index_offset = 0  # Track cumulative index

    filepaths = open_dialog_and_select_multiple_files()
    for filepath in filepaths:
        with open(filepath, "r") as json_file:
            m = json.load(json_file)

        # Append the data from each file
        all_weightedsum_unfilt.extend(m["weightedsum_unfilt"])
        all_weightedsum_filt.extend(m["weightedsum_filt"])
        all_indexcalc_unfilt.extend(m["indexcalc_unfilt"])
        all_indexcalc_filt.extend(m["indexcalc_filt"])
        all_pca_unfilt.extend(m["pca_unfilt"])
        all_pca_filt.extend(m["pca_filt"])
        all_tsne_unfilt.extend(m["tsne_unfilt"])
        all_tsne_filt.extend(m["tsne_filt"])

        # Store the index where this file's data ends
        index_offset += len(m["weightedsum_unfilt"])
        file_end_indices.append(index_offset)

    fig, axs = plt.subplots(4, 1, figsize=(9, 9))

    axs[0].plot(all_weightedsum_unfilt, label="Equal-Weight Sum", linewidth=0.85)
    axs[0].plot(all_weightedsum_filt, label="LP Filtered")
    axs[0].legend()
    axs[0].grid()
    
    axs[1].plot(all_indexcalc_unfilt, label="Average", linewidth=0.85)
    axs[1].plot(all_indexcalc_filt, label="LP Filtered")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(all_pca_unfilt, label="PCA", linewidth=0.85)
    axs[2].plot(all_pca_filt, label="LP Filtered")
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(all_tsne_unfilt, label="t-SNE", linewidth=0.85)
    axs[3].plot(all_tsne_filt, label="LP Filtered")
    axs[3].legend()
    axs[3].grid()

    # Add red vertical lines at file boundaries
    for idx in file_end_indices[:-1]:  # Exclude the last index since it's the total length
        axs[0].axvline(x=idx, color='red', linestyle='--', linewidth=1)
        axs[1].axvline(x=idx, color='red', linestyle='--', linewidth=1)
        axs[2].axvline(x=idx, color='red', linestyle='--', linewidth=1)
        axs[3].axvline(x=idx, color='red', linestyle='--', linewidth=1)

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