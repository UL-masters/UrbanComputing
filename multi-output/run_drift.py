import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecastor import Forecaster

# ---- WITH DRIFT DETECTION -----

data_path = 'data/PRSA_Data_Wanshouxigong_20130301-20170228.csv'
try:
    file_name = os.path.basename(data_path)
    station_name = file_name.split('_')[2]
except:
    station_name = "UnknownStation"

output_folder = 'results/MultiOutput_XGBoost'
os.makedirs(output_folder, exist_ok=True)

horizon_hours = 24  # forecast horizon
config = {
    'save_base': output_folder,
    'debug_mode': False,       
    'multiple': True,         
    'max_window_size': 72,    
    'horizon': horizon_hours,
    'target_column': 'PM2.5',
    'data_path': data_path
}

print(">>> Step 1: Preprocessing Data...")
forecaster = Forecaster(config)
X, y = forecaster.preprocess()

# ==========================================
# 2. data split (Train / Validation / Test)
# ==========================================
#  Holdout Method
# 1. get Test Set 
test_split_idx = int(len(X) * 0.8)
X_train_full, y_train_full = X[:test_split_idx], y[:test_split_idx]
X_test, y_test = X[test_split_idx:], y[test_split_idx:]

# 2. get Validation Set from remaining data (last 10% for validation)
val_split_idx = int(len(X_train_full) * 0.9)
X_train, y_train = X_train_full[:val_split_idx], y_train_full[:val_split_idx]
X_val, y_val = X_train_full[val_split_idx:], y_train_full[val_split_idx:]

print(f"Data Split Summary:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Valid: {X_val.shape[0]} samples (Used for Early Stopping)")
print(f"  Test:  {X_test.shape[0]} samples")

# ==========================================
# 3. Initial Training (Before Test Phase)
# ==========================================
print(">>> Step 2: Initial Training...")
forecaster.fit(X_train, y_train, X_val, y_val)

# ==========================================
# 4. Drift-Adaptive Rolling Forecast
# ==========================================
print(">>> Step 3: Drift-Adaptive Forecasting...")

predictions = []
errors = []

update_stride = 250   
drift_threshold = 12.0  # PM2.5 error threshold

for t in range(0, len(X_test), update_stride):

    X_batch = X_test[t:t + update_stride]
    y_batch = y_test[t:t + update_stride]

    # 1. Predict current batch
    y_pred_batch = forecaster.predict(X_batch)

    predictions.append(y_pred_batch)

    # 2. Drift Detection
    drift, batch_error = forecaster.detect_drift(
        y_batch,
        y_pred_batch,
        threshold=drift_threshold
    )

    errors.append(batch_error)

    print(f"Batch {t//update_stride}: MAE={batch_error:.2f} | Drift={drift}")

    # 3. Adaptation Strategy (CHOOSE ONE)
    
    # ---- OPTIONS: FULL REFIT, PARTIAL REFIT, RETRAIN ----
    if drift:
        forecaster.retrain_recent(X_train_full, y_train_full, window_size=5000)
    else:
        forecaster.full_refit(X_batch, y_batch) # changable function

# Stack all batch predictions into final array
y_pred = np.vstack(predictions)


# ==========================================
# 4. Evaluation & Visualization
# ==========================================
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
print(f"\nEvaluation Results (Horizon={horizon_hours}):")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

# --- Plotting (Inspired by Figure 9 in the paper) ---
def plot_trajectory(ax, truth, prediction, start_idx, horizon):
    hours = range(1, horizon + 1)
    ax.plot(hours, truth, color='#ff7f0e', label='Truth', linewidth=2, marker='o', markersize=3)
    ax.plot(hours, prediction, color='#1f77b4', label='Prediction', linewidth=2, marker='x', markersize=3)
    
    # Simple error band simulation
    error_std = np.std(truth - prediction)
    ax.fill_between(hours, prediction - error_std*0.5, prediction + error_std*0.5, color='#1f77b4', alpha=0.15)
    
    ax.set_xlabel(f"Hours ahead")
    ax.set_ylabel("PM2.5")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Randomly select 4 test samples
np.random.seed(42)
indices = np.random.choice(len(X_test), 4, replace=False)
indices = np.sort(indices)

for i, idx in enumerate(indices):
    plot_trajectory(axes[i], y_test[idx], y_pred[idx], start_idx=idx, horizon=horizon_hours)
    axes[i].set_title(f"Sample #{idx}")

plt.suptitle(f"Multi-step ({horizon_hours}h) Forecast for {station_name}\n"
             f"RMSE: {rmse:.2f} | Log-Transformed XGBoost", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

save_path = os.path.join(output_folder, f"{station_name}_final_forecast.png")
plt.savefig(save_path, dpi=300)
print(f"Chart saved to: {save_path}")
plt.show()