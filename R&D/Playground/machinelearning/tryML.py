import numpy as np
import pandas as pd
import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def choose_json_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask for input JSON file
    input_file = filedialog.askopenfilename(title="Select Input JSON File", filetypes=[("JSON files", "*.json")])
    if not input_file:
        print("No input file selected.")
        return None, None

    # Read JSON data into a dictionary
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Ask for output JSON file
    output_file = filedialog.askopenfilename(title="Select Output JSON File", filetypes=[("JSON files", "*.json")])
    if not output_file:
        print("No output file selected.")
        return input_data, None

    # Read JSON data into a dictionary
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    return input_data, output_data

def dict_to_matrix(data):
    if not data:
        return []

    # Ensure all values are lists and have the same length
    lengths = {len(v) for v in data.values()}
    if len(lengths) > 1:
        raise ValueError("All lists in the dictionary must have the same length")

    # Convert dict to 2D matrix (rows are the list elements, columns are the keys)
    return [list(row) for row in zip(*data.values())]



# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

###############################################################################
# Data Loading (placeholder - replace with your actual data)
###############################################################################

input_m, output_m = choose_json_files()
input_m.pop('person')
in_matrix = dict_to_matrix(input_m)
print("size of input matrix: ", len(in_matrix))




# Example data (replace with your actual data)
X = np.array(in_matrix)
y = np.array(output_m['pca_filt'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function for evaluation and plotting
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse:.4f}")
    print(f"{model_name} - R² Score: {r2:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Fatigue')
    plt.ylabel('Predicted Fatigue')
    plt.title(f'{model_name}: Actual vs Predicted Fatigue')
    plt.grid(True)
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_results.png")
    plt.close()
    
    return mse, r2

###############################################################################
# 1. Support Vector Regression (SVR)
###############################################################################
# Research Reference: Dong, H., et al. (2020). "Muscle fatigue estimation in 
# repetitive lifting tasks with different loads using surface electromyography signals"
# Journal of Electromyography and Kinesiology.
# Used RBF kernel SVR to map frequency and time-domain sEMG features to muscle fatigue levels.

def train_svr_model():
    print("\n===== Training Support Vector Regression Model =====")
    
    # Simple parameter grid - extend for better results
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    
    grid_search = GridSearchCV(
        SVR(), 
        param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Train final model with best parameters
    svr_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = svr_model.predict(X_test_scaled)
    
    # Evaluate
    evaluate_model(y_test, y_pred, "Support Vector Regression")
    
    return svr_model

###############################################################################
# 2. Random Forest Regression
###############################################################################
# Research Reference: Halilaj, E., et al. (2021). "Machine learning in human 
# movement biomechanics: Best practices, common pitfalls, and new opportunities"
# Journal of Biomechanics.
# Used ensemble methods including Random Forest for analyzing biomechanical data
# including EMG during fatigue protocols.

def train_random_forest_model():
    print("\n===== Training Random Forest Regression Model =====")
    
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train_scaled, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    
    # Train final model with best parameters
    rf_model = random_search.best_estimator_
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate
    evaluate_model(y_test, y_pred, "Random Forest Regression")
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(X_train.shape[1]), feature_importance[indices])
    plt.title('Random Forest Feature Importance')
    plt.xticks(range(X_train.shape[1]), [f'Feature {i}' for i in indices])
    plt.savefig("rf_feature_importance.png")
    plt.close()
    
    return rf_model

###############################################################################
# 3. LSTM Neural Network
###############################################################################
# Research Reference: Jia, X., et al. (2022). "LSTM-based muscle fatigue 
# estimation using surface EMG signals" Biomedical Signal Processing and Control.
# Used LSTM networks to capture temporal dependencies in sEMG signals for 
# real-time muscle fatigue prediction during dynamic contractions.

def train_lstm_model():
    print("\n===== Training LSTM Model =====")
    
    # Reshape input for LSTM [samples, time steps, features]
    # For this example, we'll treat each feature as a time step
    # In a real application, you might have actual time series data
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Create LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_lstm, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("lstm_training_history.png")
    plt.close()
    
    # Make predictions
    y_pred = model.predict(X_test_lstm).flatten()
    
    # Evaluate
    evaluate_model(y_test, y_pred, "LSTM Neural Network")
    
    return model

###############################################################################
# 4. Gradient Boosting Machines (GBM)
###############################################################################
# Research Reference: Zhu, Q., et al. (2023). "An intelligent fatigue monitoring 
# system based on XGBoost regression and sEMG features" IEEE Access.
# Used XGBoost to predict fatigue levels from sEMG signals during sustained 
# contractions, outperforming traditional methods in terms of accuracy and generalization.

def train_gbm_model():
    print("\n===== Training Gradient Boosting Machine Model =====")
    
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    random_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train_scaled, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    
    # Train final model with best parameters
    gbm_model = random_search.best_estimator_
    
    # Make predictions
    y_pred = gbm_model.predict(X_test_scaled)
    
    # Evaluate
    evaluate_model(y_test, y_pred, "Gradient Boosting Machine")
    
    # Feature importance
    feature_importance = gbm_model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(X_train.shape[1]), feature_importance[indices])
    plt.title('GBM Feature Importance')
    plt.xticks(range(X_train.shape[1]), [f'Feature {i}' for i in indices])
    plt.savefig("gbm_feature_importance.png")
    plt.close()
    
    return gbm_model

###############################################################################
# 5. Convolutional Neural Network (CNN)
###############################################################################
# Research Reference: Yang, C., et al. (2022). "CNN-based feature extraction and 
# regression for sEMG-driven muscle fatigue estimation" Frontiers in Neuroscience.
# Used 1D CNNs to automatically extract features from raw sEMG signals and 
# estimate fatigue with higher accuracy than traditional feature-based approaches.

def train_cnn_model():
    print("\n===== Training CNN Model =====")
    
    # Reshape input for CNN [samples, time steps, features]
    # For demonstration, treating features as time steps
    # In real application, you might have multichannel time series data
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Create CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("cnn_training_history.png")
    plt.close()
    
    # Make predictions
    y_pred = model.predict(X_test_cnn).flatten()
    
    # Evaluate
    evaluate_model(y_test, y_pred, "CNN")
    
    return model

###############################################################################
# Model Comparison
###############################################################################

def compare_models(results):
    model_names = list(results.keys())
    mse_values = [results[model]['mse'] for model in model_names]
    r2_values = [results[model]['r2'] for model in model_names]
    
    # Plot MSE comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(model_names, mse_values)
    plt.title('MSE Comparison (Lower is Better)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Plot R² comparison
    plt.subplot(1, 2, 2)
    plt.bar(model_names, r2_values)
    plt.title('R² Comparison (Higher is Better)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig("model_comparison.png")
    plt.close()

###############################################################################
# Main Execution
###############################################################################

if __name__ == "__main__":
    print("sEMG Fatigue Estimation - ML Model Comparison")
    
    # Train all models
    svr_model = train_svr_model()
    rf_model = train_random_forest_model()
    lstm_model = train_lstm_model()
    gbm_model = train_gbm_model()
    cnn_model = train_cnn_model()
    
    # Collect results for comparison
    results = {}
    
    # SVR
    y_pred = svr_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, "SVR")
    results["SVR"] = {"mse": mse, "r2": r2}
    
    # Random Forest
    y_pred = rf_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, "RF")
    results["RF"] = {"mse": mse, "r2": r2}
    
    # LSTM
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    y_pred = lstm_model.predict(X_test_lstm).flatten()
    mse, r2 = evaluate_model(y_test, y_pred, "LSTM")
    results["LSTM"] = {"mse": mse, "r2": r2}
    
    # GBM
    y_pred = gbm_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, "GBM")
    results["GBM"] = {"mse": mse, "r2": r2}
    
    # CNN
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    y_pred = cnn_model.predict(X_test_cnn).flatten()
    mse, r2 = evaluate_model(y_test, y_pred, "CNN")
    results["CNN"] = {"mse": mse, "r2": r2}
    
    # Compare all models
    compare_models(results)
    
    print("\nModel training and evaluation complete!")
    print("Check the generated plots for visual comparison of the models.")