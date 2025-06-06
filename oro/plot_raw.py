import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Open file dialog to select an Excel file
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    return file_path

def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    # Read Excel file
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check if required columns are present
    required_columns = ['time', 'raw', 'rectify', 'RMS']
    if not all(col in df.columns for col in required_columns):
        print("Missing required columns in the file.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['raw'], label='Raw', alpha=0.7)
    plt.plot(df['time'], df['rectify'], label='Rectify', alpha=0.7)
    plt.plot(df['time'], df['RMS'], label='RMS', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Signal over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
