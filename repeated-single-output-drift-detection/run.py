import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecaster import Forecaster
import time

# 1. Configuration & Strategy
# Options: 'full_refit', 'partial_refit', 'retrain'
DRIFT_STRATEGY = 'full_refit' 

# Experiment Parameters
horizon_hours = 24 
drift_threshold = 30  # [NEW] MAE Threshold. If error > 30, trigger update.
val_window_hours = 24 * 30 

config = {
    'save_base': f'results/DriftDetection_{DRIFT_STRATEGY}',
    'debug_mode': False,
    'multiple': True,       
    'max_window_size': 72,  
    'horizon': horizon_hours, 
    'target_column': 'PM2.5',
    'data_path': '../data/PRSA_Data_Wanshouxigong_20130301-20170228.csv'
}

os.makedirs(config['save_base'], exist_ok=True)
print(f">>> Active Drift Detection Mode")
print(f"    Strategy on Drift: {DRIFT_STRATEGY.upper()}")
print(f"    Detection Threshold (MAE): {drift_threshold}")

# 2. Train/Val Split
def get_train_val_data(forecaster, subset_df, val_hours):
    X_all, y_all = forecaster._make_instances(subset_df, horizon=1)
    
    if len(X_all) <= val_hours:
        split_point = int(len(X_all) * 0.8)
    else:
        split_point = len(X_all) - val_hours
        
    return X_all[:split_point], y_all[:split_point], X_all[split_point:], y_all[split_point:]

# 3. Initialization
forecaster = Forecaster(config)
df_full = forecaster.preprocess()

# Define Test Period (Last 20%)
split_idx = int(len(df_full) * 0.8)
train_initial_end = split_idx

print(f"Total Data: {len(df_full)}")
print(f"Initial Training End: {train_initial_end}")

print(">>> Phase 1: Initial Training (Base Model)...")
initial_df = df_full.iloc[:train_initial_end]
X_train, y_train, X_val, y_val = get_train_val_data(forecaster, initial_df, val_window_hours)
forecaster.fit(X_train, y_train, X_val, y_val, warm_start=False)

# 4. Drift-Aware Prediction Loop
print(">>> Phase 2: Dynamic Forecasting with ACTIVE Detection...")

all_predictions = []
all_truths = []
timestamps = []
drift_events = [] # To record when drift happened

# Memory to store the PREVIOUS prediction for verification
prev_prediction_window = None
prev_truth_window = None

current_t = train_initial_end
n_steps = 0

target_vals = df_full[config['target_column']].values
feat_vals = df_full[forecaster.feature_cols].values

start_time = time.time()

while current_t < len(df_full) - horizon_hours:
    
    # We can only detect drift if we have a previous prediction to verify
    drift_detected = False
    current_mae = 0.0
    
    if prev_prediction_window is not None:
        # Get the GROUND TRUTH for the window we just finished forecasting
        # The previous window covered [current_t - horizon : current_t]
        truth_check = target_vals[current_t - horizon_hours : current_t]
        
        # Check for drift
        drift_detected, current_mae = forecaster.detect_drift(
            truth_check, 
            prev_prediction_window, 
            threshold=drift_threshold
        )
        
        if drift_detected:
            print(f"    [!] DRIFT DETECTED at time {current_t} | MAE: {current_mae:.2f} > {drift_threshold}")
            drift_events.append(current_t)
            
            # ADAPTATION (Triggered by Drift)
            print(f"        -> Executing Strategy: {DRIFT_STRATEGY}...")
            
            if DRIFT_STRATEGY == 'full_refit':
                subset_df = df_full.iloc[:current_t]
                X_t, y_t, X_v, y_v = get_train_val_data(forecaster, subset_df, val_window_hours)
                forecaster.fit(X_t, y_t, X_v, y_v, warm_start=False) 
                
            elif DRIFT_STRATEGY == 'retrain':
                # Sliding window 1 year (to capture seasonality)
                window_size = 24 * 365 
                start_idx = max(0, current_t - window_size)
                subset_df = df_full.iloc[start_idx : current_t]
                X_t, y_t, X_v, y_v = get_train_val_data(forecaster, subset_df, val_window_hours)
                forecaster.fit(X_t, y_t, X_v, y_v, warm_start=False)
                
            elif DRIFT_STRATEGY == 'partial_refit':
                # Use recent data (e.g. last 14 days) to update weights
                recent_window = 24 * 14
                subset_df = df_full.iloc[current_t - recent_window : current_t]
                X_new, y_new = forecaster._make_instances(subset_df, horizon=1)
                forecaster.fit(X_new, y_new, warm_start=True)

    # --- C. FORECASTING STEP (Recursive) ---
    current_history = list(target_vals[current_t - config['max_window_size'] : current_t])
    sample_preds = []
    
    for h in range(horizon_hours):
        exog_feats = feat_vals[current_t + h]
        pred_val = forecaster.predict_single_step_vector(current_history, exog_feats)
        sample_preds.append(pred_val)
        current_history.append(pred_val)
    
    # Store predictions for NEXT loop's drift detection
    prev_prediction_window = np.array(sample_preds)
    
    # Store for final evaluation
    all_predictions.append(sample_preds)
    all_truths.append(target_vals[current_t : current_t + horizon_hours])
    timestamps.append(current_t)
    
    current_t += horizon_hours
    n_steps += 1
    
    if n_steps % 50 == 0:
        print(f"    Step {n_steps}: Current MAE ~ {current_mae:.2f}")

print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")

# 5. Final Evaluation
preds = np.array(all_predictions)
truths = np.array(all_truths)

rmse = mean_squared_error(truths, preds) ** 0.5
mae = mean_absolute_error(truths, preds)

filename = os.path.basename(config['data_path'])
try:
    station_name = filename.split('_')[2]
except IndexError:
    station_name = "UnknownStation"

print(f"\n==========================================")
print(f"FINAL RESULTS: Active Detection ({DRIFT_STRATEGY})")
print(f"Station: {station_name}")
print(f"==========================================")
print(f"Total Drift Events Detected: {len(drift_events)}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Pick 4 random samples
np.random.seed(42) 
if len(preds) >= 4:
    plot_indices = np.random.choice(len(preds), 4, replace=False)
    plot_indices = np.sort(plot_indices)
else:
    plot_indices = range(len(preds))

for i, idx in enumerate(plot_indices):
    if i >= 4: break
    truth = truths[idx]
    pred = preds[idx]
    
    hours = range(1, horizon_hours + 1)
    axes[i].plot(hours, truth, 'o-', color='#ff7f0e', label='Truth')
    axes[i].plot(hours, pred, 'x--', color='#1f77b4', label='Pred')
    axes[i].set_title(f"Window #{idx}")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle(f"Station: {station_name} | Strategy: {DRIFT_STRATEGY}\nThreshold={drift_threshold} | Drifts: {len(drift_events)} | RMSE: {rmse:.2f}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

save_filename = f'drift_detection_{station_name}_{DRIFT_STRATEGY}.png'
save_path = os.path.join(config['save_base'], save_filename)
plt.savefig(save_path)
print(f"Result plot saved to: {save_path}")

plt.show()