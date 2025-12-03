import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecaster import Forecaster
import matplotlib.pyplot as plt

#  Automatic path and station name setup
# Path to your current data file
data_path = 'data/PRSA_Data_Wanshouxigong_20130301-20170228.csv'

# Automatically extract station name from filename (e.g., Aotizhongxin)
# Logic: Filename is "PRSA_Data_Aotizhongxin_...", split by underscore, take the 3rd element
file_name = os.path.basename(data_path)
station_name = file_name.split('_')[2] 
print(f"Station detected: {station_name}")

# Set result output folder
output_folder = 'results/XGBoost'
os.makedirs(output_folder, exist_ok=True)

# Configuration 
config = {
    'save_base': output_folder, 
    'debug_mode': True,                 # Debug mode
    'multiple': True,                   # Enable weather features
    'max_window_size': 24,              # Look back past 24 hours
    'horizon': 1,                       # Predict next 1 hour
    'test_days': 365,                   
    'target_column': 'PM2.5',           
    'data_path': data_path 
}

# Initialization & Preprocessing 
forecaster = Forecaster(config)

print(">>> Step 1: Loading & Preprocessing...")
X, y = forecaster.preprocess()

# Train/Test Split
# 80% for training, 20% for testing
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]
print(f"Train set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")

#  Training 
print(">>> Step 2: Training XGBoost...")
forecaster.fit(X_train, y_train)

# Prediction 
print(">>> Step 3: Predicting...")
y_pred = forecaster.predict(X_test)

# Evaluation 
# RMSE calculation: Manual square root (** 0.5) to avoid version compatibility issues
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)

print(f"\nSUCCESS! Evaluation Results for {station_name}:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# Plotting 
plt.figure(figsize=(12, 6))
# Plot only the last 200 hours for clarity
plt.plot(y_test[-200:], label='True PM2.5', color='blue', alpha=0.6)
plt.plot(y_pred[-200:], label='Predicted PM2.5 (XGBoost)', color='red', linestyle='--', alpha=0.8)
plt.legend()
plt.title(f"Air Pollution Forecast: {station_name} (RMSE: {rmse:.2f})")
plt.xlabel("Time (Last 200 Hours)")
plt.ylabel("PM2.5 Concentration")

# Automatically generate filename with station name
save_path = os.path.join(output_folder, f"{station_name}_forecast.png")
plt.savefig(save_path)
print(f"\nChart saved to: {save_path}")