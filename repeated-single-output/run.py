import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecastor import Forecaster
import time

# ==========================================
# 1. Configuration & Strategy Selection
# ==========================================
# Options: 'static', 'full_refit', 'partial_refit', 'retrain'
DRIFT_STRATEGY = 'retrain' 

# Experiment Parameters
horizon_hours = 24 
refit_interval = 24 * 7  # Update model every 7 days
                         # Set to 24 for daily updates (strictest adaptation)

# Validation Strategy (Option B: Engineering Approach)
# We use the last 30 days of the *training window* as holdout validation
val_window_hours = 24 * 30 

config = {
    'save_base': f'results/Drift_{DRIFT_STRATEGY}',
    'debug_mode': False,
    'multiple': True,       
    'max_window_size': 72,  
    'horizon': horizon_hours, 
    'target_column': 'PM2.5',
    'data_path': '../data/PRSA_Data_Wanshouxigong_20130301-20170228.csv'
}

os.makedirs(config['save_base'], exist_ok=True)
print(f">>> Running Strategy: {DRIFT_STRATEGY.upper()}")
print(f"    Refit Interval: Every {refit_interval} hours")
print(f"    Validation: Last {val_window_hours} hours (30 days) of available history")

# ==========================================
# 2. Helper: Strict Train/Validation Splitting
# ==========================================
def get_train_val_data(forecaster, subset_df, val_hours):
    """
    Splits the available history (subset_df) into Train and Validation sets.
    
    Logic:
    - Validation Set: The LAST 'val_hours' (30 days) of the provided history.
    - Training Set: All history BEFORE the validation set.
    
    This ensures the model is tuned on the most recent data trends 
    immediately preceding the forecasting period.
    """
    # 1. Create all instances (X, y) from the dataframe
    X_all, y_all = forecaster._make_instances(subset_df, horizon=1)
    
    # 2. Safety check: ensure enough data exists
    if len(X_all) <= val_hours:
        # Fallback for very small datasets: use last 20%
        split_point = int(len(X_all) * 0.8)
        print("    ! Warning: Data insufficient for full 30-day validation. Using 20% split.")
    else:
        # Standard: Split out the last 30 days
        split_point = len(X_all) - val_hours
        
    X_t = X_all[:split_point]
    y_t = y_all[:split_point]
    X_v = X_all[split_point:]
    y_v = y_all[split_point:]
    
    return X_t, y_t, X_v, y_v

# ==========================================
# 3. Initialization & Data Preparation
# ==========================================
forecaster = Forecaster(config)
df_full = forecaster.preprocess()

# Define Test Period: Last 20% of the dataset
# This covers approx. 9.6 months (covering all seasons: Spring, Summer, Autumn, Winter)
split_idx = int(len(df_full) * 0.8)
train_initial_end = split_idx

print(f"Total Data Points: {len(df_full)}")
print(f"Initial Training End Index: {train_initial_end} (Start of Test Phase)")

# --- Phase 1: Initial Training (Base Model) ---
# Before predicting the first test point, we train on the initial 80% history.
# We strictly use the last 30 days of this 80% as validation.
print(">>> Phase 1: Initial Training (Base Model)...")
initial_df = df_full.iloc[:train_initial_end]

# Split Train/Val
X_train, y_train, X_val, y_val = get_train_val_data(forecaster, initial_df, val_window_hours)

print(f"    Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")
# Fit with warm_start=False (Fresh training)
forecaster.fit(X_train, y_train, X_val, y_val, warm_start=False)

# ==========================================
# 4. Dynamic Prediction Loop (Drift Adaptation)
# ==========================================
print(">>> Phase 2: Dynamic Forecasting with Drift Adaptation...")

all_predictions = []
all_truths = []
timestamps = []

# Start simulation from the beginning of the Test Period
current_t = train_initial_end
n_steps = 0

target_vals = df_full[config['target_column']].values
feat_vals = df_full[forecaster.feature_cols].values

start_time = time.time()

while current_t < len(df_full) - horizon_hours:
    
    # --- A. Drift Adaptation Step (Model Update) ---
    # Check if it is time to update the model (e.g., every 7 days)
    if n_steps > 0 and (current_t - train_initial_end) % refit_interval == 0:
        
        print(f"    [Adaptation] Time {current_t}: Updating model ({DRIFT_STRATEGY})...")
        
        if DRIFT_STRATEGY == 'static':
            # Do nothing. The model remains frozen as trained in Phase 1.
            pass 
            
        elif DRIFT_STRATEGY == 'full_refit':
            # Strategy: Full Refit (Growing Window)
            # Use ALL data available from start (0) up to current time (current_t).
            # The training set grows over time.
            subset_df = df_full.iloc[:current_t]
            
            # Split Train/Val (Last 30 days of CURRENT history becomes new validation)
            X_t, y_t, X_v, y_v = get_train_val_data(forecaster, subset_df, val_window_hours)
            
            # Train from scratch (warm_start=False)
            forecaster.fit(X_t, y_t, X_v, y_v, warm_start=False) 
            
        elif DRIFT_STRATEGY == 'retrain':
            # Strategy: Retrain (Sliding Window)
            # Use only the recent N days (e.g., 90 days train + 30 days val = 120 days total)
            # Discard very old data to adapt to new regimes.
            window_size = 24 * 120 
            start_idx = max(0, current_t - window_size)
            subset_df = df_full.iloc[start_idx : current_t]
            
            # Split Train/Val
            X_t, y_t, X_v, y_v = get_train_val_data(forecaster, subset_df, val_window_hours)
            
            # Train from scratch
            forecaster.fit(X_t, y_t, X_v, y_v, warm_start=False)
            
        elif DRIFT_STRATEGY == 'partial_refit':
            # Strategy: Partial Refit (Incremental Learning)
            # Only use the newest batch of data (since last refit) to update weights.
            # NOTE: Data batch is usually too small for a 30-day validation set.
            # We skip validation split here and update directly on new data.
            new_data_start = current_t - refit_interval
            subset_df = df_full.iloc[new_data_start : current_t]
            
            X_new, y_new = forecaster._make_instances(subset_df, horizon=1)
            
            # warm_start=True keeps existing trees and adds new ones
            forecaster.fit(X_new, y_new, warm_start=True)

    # --- B. Forecasting Step (Recursive 24h) ---
    # Construct history window for the current prediction point
    current_history = list(target_vals[current_t - config['max_window_size'] : current_t])
    sample_preds = []
    
    # Recursively predict t+1, t+2, ..., t+24
    for h in range(horizon_hours):
        # Oracle Assumption: Future weather features are known
        exog_feats = feat_vals[current_t + h]
        
        # Predict one step
        pred_val = forecaster.predict_single_step_vector(current_history, exog_feats)
        
        # Append prediction to history for the next step
        sample_preds.append(pred_val)
        current_history.append(pred_val)
    
    # Store results
    all_predictions.append(sample_preds)
    all_truths.append(target_vals[current_t : current_t + horizon_hours])
    timestamps.append(current_t)
    
    # Move forward
    current_t += horizon_hours
    n_steps += 1
    
    # Progress logging
    if n_steps % 20 == 0:
        print(f"    Step {n_steps}: Forecasted up to index {current_t}")

print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")

# ==========================================
# 5. Evaluation & Visualization (Updated Style)
# ==========================================
preds = np.array(all_predictions)
truths = np.array(all_truths)

# Compute global metrics
rmse = mean_squared_error(truths, preds) ** 0.5
mae = mean_absolute_error(truths, preds)

print(f"\n==========================================")
print(f"FINAL RESULTS: {DRIFT_STRATEGY.upper()}")
print(f"==========================================")
print(f"Total Forecast Windows: {len(preds)}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# --- Drawing the 2x2 Grid Plot (Same style as your screenshot) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Randomly select 4 distinct time points to visualize
# We use a fixed seed to make sure the "random" choice is stable for presentation
np.random.seed(42) 
if len(preds) >= 4:
    plot_indices = np.random.choice(len(preds), 4, replace=False)
    plot_indices = np.sort(plot_indices) # Sort them chronologically
else:
    # Fallback if we have fewer than 4 predictions (e.g. testing with tiny data)
    plot_indices = range(len(preds))

for i, idx in enumerate(plot_indices):
    # Only handle up to 4 plots
    if i >= 4: break
    
    truth = truths[idx]
    pred = preds[idx]
    start_time_index = timestamps[idx] # The actual time index in the original DF
    
    hours = range(1, horizon_hours + 1)
    
    # Truth line (Orange with circle markers)
    axes[i].plot(hours, truth, 'o-', color='#ff7f0e', label='Truth', markersize=4)
    
    # Prediction line (Blue with 'x' markers)
    axes[i].plot(hours, pred, 'x-', color='#1f77b4', label=f'Pred ({DRIFT_STRATEGY})', markersize=4)
    
    # Styling
    axes[i].set_title(f"Forecast starting at index {start_time_index}")
    axes[i].set_xlabel("Horizon (Hours)")
    axes[i].set_ylabel("PM2.5")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Add a main title with the strategy name
plt.suptitle(f"Drift Adaptation Strategy: {DRIFT_STRATEGY.upper()}\nRMSE: {rmse:.2f}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

# Save the plot
save_path = os.path.join(config['save_base'], 'grid_forecast_samples.png')
plt.savefig(save_path, dpi=300)
print(f"Chart saved to: {save_path}")
plt.show()