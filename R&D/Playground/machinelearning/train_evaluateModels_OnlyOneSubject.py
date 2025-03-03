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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


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
    
    subjectName = "R&D/Playground/machinelearning/results/" + input_file.split('.')[0].split('/')[-1]

    # Ask for output JSON file
    output_file = filedialog.askopenfilename(title="Select Output JSON File", filetypes=[("JSON files", "*.json")])
    if not output_file:
        print("No output file selected.")
        return input_data, None

    # Read JSON data into a dictionary
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    return input_data, output_data, subjectName

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
# Data Loading
###############################################################################

input_m, output_m, subjectName = choose_json_files()
input_m.pop('person')
in_matrix = dict_to_matrix(input_m)
print("size of input matrix: ", len(in_matrix))

# Data Management
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
    plt.savefig(subjectName+f"{model_name.lower().replace(' ', '_')}_results.png")
    plt.close()
    
    return mse, r2

###############################################################################
# 1. Linear Regression (NEW)
###############################################################################
# Research Reference: Gonzalez-Izal, M., et al. (2010). "EMG spectral indices 
# and muscle power fatigue during dynamic contractions" Journal of Electromyography and Kinesiology.
# Used linear regression models to correlate EMG spectral indices with muscle fatigue levels.

def train_linear_model():
    print("\n===== Training Linear Regression Models =====")
    
    # Train three types of linear models
    models = {
        'Simple Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }
    
    best_model = None
    best_score = -float('inf')
    best_name = ""
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name} Regression...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse, r2 = evaluate_model(y_test, y_pred, f"{name} Regression")
        
        # Track best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    # For the best linear model, analyze coefficients
    if isinstance(best_model, (LinearRegression, Ridge, Lasso)):
        coefs = best_model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(X_train.shape[1]), coefs[indices])
        plt.title(f'{best_name} Regression: Feature Coefficients')
        plt.xticks(range(X_train.shape[1]), [f'Feature {i}' for i in indices])
        plt.savefig(subjectName+f"{best_name.lower().replace(' ', '_')}_coefficients.png")
        plt.close()
    
    return best_model, best_name

###############################################################################
# 2. Support Vector Regression (SVR) - Already implemented
###############################################################################

def train_svr_model():
    print("\n===== Training Support Vector Regression Model =====")
    
    # Simple parameter grid - extend for better results
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear', 'poly'],
        'epsilon': [0.1, 0.01, 0.001]
    }
    
    grid_search = GridSearchCV(
        SVR(), 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
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
# 3. Random Forest Regression - Already implemented
###############################################################################

def train_random_forest_model():
    print("\n===== Training Random Forest Regression Model =====")
    
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
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
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), feature_importance[indices])
    plt.title('Random Forest Feature Importance')
    plt.xticks(range(X_train.shape[1]), [f'Feature {i}' for i in indices])
    plt.savefig(subjectName+"rf_feature_importance.png")
    plt.close()
    
    return rf_model

###############################################################################
# 4. LSTM Neural Network - Already implemented
###############################################################################

def train_lstm_model():
    print("\n===== Training LSTM Model =====")
    
    # Reshape input for LSTM [samples, time steps, features]
    # For this example, we'll treat each feature as a time step
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
        patience=20,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_lstm, y_train,
        epochs=150,
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
    plt.savefig(subjectName+"lstm_training_history.png")
    plt.close()
    
    # Make predictions
    y_pred = model.predict(X_test_lstm).flatten()
    
    # Evaluate
    evaluate_model(y_test, y_pred, "LSTM Neural Network")
    
    return model

###############################################################################
# 5. Gradient Boosting Machines (GBM) - Already implemented
###############################################################################

def train_gbm_model():
    print("\n===== Training Gradient Boosting Machine Model =====")
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    random_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
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
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), feature_importance[indices])
    plt.title('GBM Feature Importance')
    plt.xticks(range(X_train.shape[1]), [f'Feature {i}' for i in indices])
    plt.savefig(subjectName+"gbm_feature_importance.png")
    plt.close()
    
    return gbm_model

###############################################################################
# 6. Convolutional Neural Network (CNN) - Already implemented
###############################################################################

def train_cnn_model():
    print("\n===== Training CNN Model =====")
    
    # Reshape input for CNN [samples, time steps, features]
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
        patience=20,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_cnn, y_train,
        epochs=150,
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
    plt.savefig(subjectName+"cnn_training_history.png")
    plt.close()
    
    # Make predictions
    y_pred = model.predict(X_test_cnn).flatten()
    
    # Evaluate
    evaluate_model(y_test, y_pred, "CNN")
    
    return model

###############################################################################
# 7. K-Nearest Neighbors (KNN) - NEW
###############################################################################
# Research Reference: Cifrek, M., et al. (2009). "Surface EMG based muscle 
# fatigue evaluation in biomechanics" Clinical Biomechanics.
# Used non-parametric methods to estimate fatigue from EMG features, where KNN
# approaches showed good performance for complex non-linear relationships.

def train_knn_model():
    print("\n===== Training K-Nearest Neighbors Model =====")
    
    # Parameter grid
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2, 3]  # p=1 for manhattan, p=2 for euclidean
    }
    
    grid_search = GridSearchCV(
        KNeighborsRegressor(),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Train final model with best parameters
    knn_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = knn_model.predict(X_test_scaled)
    
    # Evaluate
    evaluate_model(y_test, y_pred, "K-Nearest Neighbors")
    
    # Analyze performance across different k values
    k_range = list(range(1, 21))
    mse_values = []
    
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, mse_values, marker='o')
    plt.title('KNN: MSE vs. k Value')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig(subjectName+"knn_k_analysis.png")
    plt.close()
    
    return knn_model

###############################################################################
# Model Comparison
###############################################################################

def compare_models(results):
    model_names = list(results.keys())
    mse_values = [results[model]['mse'] for model in model_names]
    r2_values = [results[model]['r2'] for model in model_names]
    
    # Sort by performance (R²)
    sorted_indices = np.argsort(r2_values)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_mse = [mse_values[i] for i in sorted_indices]
    sorted_r2 = [r2_values[i] for i in sorted_indices]
    
    # Plot MSE comparison
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    bars = plt.bar(sorted_names, sorted_mse, color='salmon')
    plt.title('MSE Comparison (Lower is Better)', fontsize=14)
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot R² comparison
    plt.subplot(2, 1, 2)
    bars = plt.bar(sorted_names, sorted_r2, color='skyblue')
    plt.title('R² Comparison (Higher is Better)', fontsize=14)
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(subjectName+"model_comparison.png", dpi=300)
    plt.close()
    
    # Print ranking
    print("\n===== Model Performance Ranking =====")
    print("Ranked by R² Score (higher is better):")
    for i, model in enumerate(sorted_names):
        print(f"{i+1}. {model}: R²={sorted_r2[i]:.4f}, MSE={sorted_mse[i]:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': sorted_names,
        'R²': sorted_r2,
        'MSE': sorted_mse
    })
    results_df.to_csv('model_performance_results.csv', index=False)
    
    return sorted_names[0]  # Return the best model name

###############################################################################
# Main Execution
###############################################################################

if __name__ == "__main__":
    print("Muscle Fatigue Index Prediction - ML Model Comparison")
    print("====================================================")
    print(f"Input features: {X.shape[1]} metrics")
    print(f"Total samples: {X.shape[0]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print("====================================================")
    
    # Train all models
    linear_model, linear_name = train_linear_model()
    svr_model = train_svr_model()
    rf_model = train_random_forest_model()
    lstm_model = train_lstm_model()
    gbm_model = train_gbm_model()
    cnn_model = train_cnn_model()
    knn_model = train_knn_model()
    
    # Collect results for comparison
    results = {}
    
    # Linear
    y_pred = linear_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, linear_name)
    results[linear_name] = {"mse": mse, "r2": r2}
    
    # SVR
    y_pred = svr_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, "SVR")
    results["SVR"] = {"mse": mse, "r2": r2}
    
    # Random Forest
    y_pred = rf_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, "Random Forest")
    results["Random Forest"] = {"mse": mse, "r2": r2}
    
    # LSTM
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    y_pred = lstm_model.predict(X_test_lstm).flatten()
    mse, r2 = evaluate_model(y_test, y_pred, "LSTM")
    results["LSTM"] = {"mse": mse, "r2": r2}
    
    # GBM
    y_pred = gbm_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, "Gradient Boosting")
    results["Gradient Boosting"] = {"mse": mse, "r2": r2}
    
    # CNN
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    y_pred = cnn_model.predict(X_test_cnn).flatten()
    mse, r2 = evaluate_model(y_test, y_pred, "CNN")
    results["CNN"] = {"mse": mse, "r2": r2}
    
    # KNN
    y_pred = knn_model.predict(X_test_scaled)
    mse, r2 = evaluate_model(y_test, y_pred, "KNN")
    results["KNN"] = {"mse": mse, "r2": r2}
    
    # Compare all models
    best_model = compare_models(results)
    
    print("\nModel training and evaluation complete!")
    print(f"The best performing model is: {best_model}")
    print("Check the generated plots and CSV file for detailed comparison.")