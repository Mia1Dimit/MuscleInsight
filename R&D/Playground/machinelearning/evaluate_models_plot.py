import tkinter as tk
from tkinter import filedialog
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

def select_file(title, file_types):
    """Helper function to select a file using a dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=file_types)
    if not file_path:
        print(f"No {title.lower()} selected.")
        return None
    return file_path

def dict_to_matrix(data):
    """Convert a dictionary of lists to a matrix"""
    if not data:
        return []

    # Ensure all values are lists and have the same length
    lengths = {len(v) for v in data.values()}
    if len(lengths) > 1:
        raise ValueError("All lists in the dictionary must have the same length")

    # Convert dict to 2D matrix (rows are the list elements, columns are the keys)
    return [list(row) for row in zip(*data.values())]

def main():
    # Select model
    model_path = select_file("Select Model File", [("Joblib files", "*.joblib")])
    if not model_path:
        return
    
    # Select input JSON
    input_path = select_file("Select Input JSON File", [("JSON files", "*.json")])
    if not input_path:
        return
    
    # Select output JSON with expected results
    expected_output_path = select_file("Select Expected Output JSON File", [("JSON files", "*.json")])
    if not expected_output_path:
        return
    
    # Load model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load input data
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            
        # Remove 'person' key if it exists (based on your earlier code)
        if 'person' in input_data:
            input_data.pop('person')
            
        # Convert to matrix
        input_matrix = dict_to_matrix(input_data)
        X = np.array(input_matrix)
        print(f"Input data loaded: {X.shape}")
    except Exception as e:
        print(f"Error loading input data: {e}")
        return
    
    # Load expected output
    try:
        with open(expected_output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
            
        # Assuming 'pca_filt' is the target variable based on your earlier code
        if 'pca_filt' in output_data:
            y_true = np.array(output_data['pca_filt'])
            print(f"Expected output loaded: {y_true.shape}")
        else:
            print("Warning: 'pca_filt' not found in output data. Available keys:", list(output_data.keys()))
            output_key = list(output_data.keys())[0]  # Use the first key as fallback
            y_true = np.array(output_data[output_key])
            print(f"Using '{output_key}' instead: {y_true.shape}")
    except Exception as e:
        print(f"Error loading expected output: {e}")
        return
    
    # Make predictions
    try:
        y_pred = model.predict(X)
        print(f"Predictions made: {y_pred.shape}")
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    # Ensure the predictions and expected outputs have the same shape
    if len(y_pred) != len(y_true):
        print(f"Warning: Prediction length ({len(y_pred)}) doesn't match expected output length ({len(y_true)})")
        # Use the shorter length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Predicted vs Expected
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(y_true)), y_true, label='Expected', alpha=0.7, s=20)
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7, s=20)
    plt.title(f'Model Prediction vs Expected Output - R²: {r2:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Correlation
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # Add a perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Correlation Plot: Expected vs Predicted')
    plt.xlabel('Expected Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add model name and path to figure
    model_name = os.path.basename(model_path)
    plt.figtext(0.5, 0.01, f"Model: {model_name} | RMSE: {rmse:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    # Ask if user wants to save the figure
    root = tk.Tk()
    root.withdraw()
    if tk.messagebox.askyesno("Save Figure", "Would you like to save this figure?"):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Figure As"
        )
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
    
if __name__ == "__main__":
    main()