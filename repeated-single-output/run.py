import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecaster import Forecaster
import time

# ==========================================
# 1. Configuration
# ==========================================
# Strategy: STATIC (No Drift Detection, No Retraining)
# We train ONCE on the training set, and predict the whole test set.

horizon_hours = 24 
val_window_hours = 24 * 30  # Validation: Last 30 days of training data

config = {
    'save_base': 'results/Baseline_Static',  # Output folder
    'debug_mode': False,
    'multiple': True,       
    'max_window_size': 72,  
    'horizon': horizon_hours, 
    'target_column': 'PM2.5',
    'data_path': '../data/PRSA_Data_Wanshouxigong_20130301-20170228.csv'
}

os.makedirs(config['save_base'], exist_ok=True)
print(f">>> Running Strategy: STATIC (No Drift Adaptation)")

# ==========================================
# 2. Helper: Train/Validation Split
# ==========================================
def get_train_val_data(forecaster, subset_df, val_hours):
    """
    Splits the data into Training and Validation sets.
    Validation Set = The LAST 'val_hours' (30 days).
    Training Set = Everything before that.
    """
    X_all, y_all = forecaster._make_instances(subset_df, horizon=1)
    
    if len(X_all) <= val_hours:
        split_point = int(len(X_all) * 0.8)
    else:
        split_point = len(X_all) - val_hours
        
    return X_all[:split_point], y_all[:split_point], X_all[split_point:], y_all[split_point:]

# ==========================================
# 3. Initialization
# ==========================================
forecaster = Forecaster(config)
df_full = forecaster.preprocess()

# Define Test Period: Last 20%
split_idx = int(len(df_full) * 0.8)
train_initial_end = split_idx

print(f"Total Data Points: {len(df_full)}")
print(f"Training End / Test Start Index: {train_initial_end}")

# ==========================================
# 4. Phase 1: Train ONCE (Static Model)
# ==========================================
print(">>> Phase 1: Training Static Model...")
initial_df = df_full.iloc[:train_initial_end]

# Split Train/Val
X_train, y_train, X_val, y_val = get_train_val_data(forecaster, initial_df, val_window_hours)

# Fit the model (Standard training)
forecaster.fit(X_train, y_train, X_val, y_val)

# ==========================================
# 5. Phase 2: Prediction Loop (No Updates)
# ==========================================
print(">>> Phase 2: Static Forecasting (No Adaptation)...")

all_predictions = []
all_truths = []
timestamps = []

current_t = train_initial_end
n_steps = 0

target_vals = df_full[config['target_column']].values
feat_vals = df_full[forecaster.feature_cols].values

start_time = time.time()

# Loop through the test set
while current_t < len(df_full) - horizon_hours:
    
    # --- Forecasting Step (Recursive 24h) ---
    # Construct history window for the current prediction point
    current_history = list(target_vals[current_t - config['max_window_size'] : current_t])
    sample_preds = []
    
    for h in range(horizon_hours):
        # Oracle Assumption: Future weather features are known (Exogenous)
        exog_feats = feat_vals[current_t + h]
        
        # Predict one step
        pred_val = forecaster.predict_single_step_vector(current_history, exog_feats)
        
        # Recursive: Append prediction to history for the next step input
        sample_preds.append(pred_val)
        current_history.append(pred_val) 
    
    # Store results
    all_predictions.append(sample_preds)
    all_truths.append(target_vals[current_t : current_t + horizon_hours])
    timestamps.append(current_t)
    
    # Move forward
    current_t += horizon_hours 
    n_steps += 1
    
    if n_steps % 50 == 0:
        print(f"    Step {n_steps}: Forecasted up to index {current_t}")

print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")

# ==========================================
# 6. Evaluation
# ==========================================
preds = np.array(all_predictions)
truths = np.array(all_truths)

rmse = mean_squared_error(truths, preds) ** 0.5
mae = mean_absolute_error(truths, preds)

# --- [FEATURE] Extract Station Name from Filename ---
filename = os.path.basename(config['data_path'])
try:
    # Assumes format: PRSA_Data_StationName_...
    station_name = filename.split('_')[2]
except IndexError:
    station_name = "UnknownStation"

print(f"\n==========================================")
print(f"FINAL RESULTS: STATIC BASELINE")
print(f"Station: {station_name}")
print(f"==========================================")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# ==========================================
# 7. Visualization
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Randomly select 4 distinct time points to visualize
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
    start_idx = timestamps[idx]
    
    hours = range(1, horizon_hours + 1)
    
    axes[i].plot(hours, truth, 'o-', color='#ff7f0e', label='Truth', markersize=4)
    axes[i].plot(hours, pred, 'x-', color='#1f77b4', label='Static Pred', markersize=4)
    
    axes[i].set_title(f"Window starting at index {start_idx}")
    axes[i].set_xlabel("Horizon (Hours)")
    axes[i].set_ylabel("PM2.5")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# --- [UPDATE] Title with Station Name ---
plt.suptitle(f"Station: {station_name} | Static Baseline (No Drift Detection)\nRMSE: {rmse:.2f}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- [UPDATE] Dynamic Filename with Station Name ---
save_filename = f'static_results_{station_name}.png'
save_path = os.path.join(config['save_base'], save_filename)
plt.savefig(save_path, dpi=300)

print(f"Chart saved to: {save_path}")
plt.show()