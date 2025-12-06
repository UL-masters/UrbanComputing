import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecastor import Forecaster

# ==========================================
# 1. Configuration
# ==========================================
data_path = '../data/PRSA_Data_Wanshouxigong_20130301-20170228.csv'
try:
    station_name = os.path.basename(data_path).split('_')[2]
except:
    station_name = "Unknown"

output_folder = 'results/Repeated_SingleOutput_Refit'
os.makedirs(output_folder, exist_ok=True)

horizon_hours = 24 

config = {
    'save_base': output_folder,
    'debug_mode': False,
    'multiple': True,       
    'max_window_size': 72,  
    'horizon': horizon_hours, 
    'target_column': 'PM2.5',
    'data_path': data_path
}

print(f">>> Initializing Repeated Single-Output Forecaster (Paper Strategy)...")
forecaster = Forecaster(config)

# Load and Preprocess Data
df_full = forecaster.preprocess()

# ==========================================
# 2. Data Splitting & Training
# ==========================================
print(">>> Preparing Training Data...")
# Split into Train and Validation sets for model selection
X_train, y_train, X_val, y_val = forecaster.create_train_data(df_full, split_ratio=0.8)

print(f"    Train size: {X_train.shape}")
print(f"    Holdout Validation size: {X_val.shape}")

# Train the model (Phase 1: Early Stopping -> Phase 2: Refit on Train+Val)
forecaster.fit(X_train, y_train, X_val, y_val)

# ==========================================
# 3. Recursive Forecasting (Inference)
# ==========================================
print(">>> Performing Recursive Forecasting on Test Set...")

# Define Test period (The remaining 20% of data)
test_start_index = int(len(df_full) * 0.8)
test_end_index = len(df_full) - horizon_hours

# Randomly sample indices for plotting (to mimic the previous visualization style)
# Or use specific indices if you want to reproduce specific events
np.random.seed(42)
eval_indices = np.arange(test_start_index, test_end_index, 24) # Evaluate every 24h block

# Generate recursive predictions
y_pred_recursive = forecaster.predict_recursive(df_full, eval_indices)

# Extract ground truth for evaluation
y_test_truth = []
for idx in eval_indices:
    y_test_truth.append(df_full[config['target_column']].values[idx : idx+horizon_hours])
y_test_truth = np.array(y_test_truth)

# ==========================================
# 4. Evaluation & Visualization
# ==========================================
rmse = mean_squared_error(y_test_truth, y_pred_recursive) ** 0.5
mae = mean_absolute_error(y_test_truth, y_pred_recursive)

print(f"\nRecursive Evaluation Results (Refit Strategy):")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

# Plotting specific samples
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

plot_indices = np.random.choice(len(y_pred_recursive), 4, replace=False)

for i, idx in enumerate(plot_indices):
    truth = y_test_truth[idx]
    pred = y_pred_recursive[idx]
    
    hours = range(1, horizon_hours + 1)
    axes[i].plot(hours, truth, 'o-', label='Truth', color='#ff7f0e')
    axes[i].plot(hours, pred, 'x-', label='Recursive Pred', color='#1f77b4')
    axes[i].set_title(f"Forecast starting at index {eval_indices[idx]}")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle(f"Repeated Single-Output Forecast ({horizon_hours}h) w/ Refit Strategy\nRMSE: {rmse:.2f}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_folder, 'recursive_forecast_refit.png'))
plt.show()