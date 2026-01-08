import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecaster_drift import Forecaster
import time


DRIFT_STRATEGY = 'partial refit'   # 'full_refit' | 'retrain' | 'partial_refit'
STATION_NAME = 'Aotizhongxin'
horizon_hours = 24
drift_threshold = 30
val_window_hours = 24 * 30
COOLDOWN_STEPS = 2*horizon_hours  # or 2*horizon
last_drift_step = -np.inf


config = {
    'save_base': f'multi-output-drift-detection/results/DriftDetection_{DRIFT_STRATEGY}',
    'debug_mode': False,
    'multiple': True,
    'max_window_size': 72,
    'horizon': horizon_hours,
    'target_column': 'PM2.5',
    'data_path': 'data/PRSA_Data_Aotizhongxin_20130301-20170228.csv'
}

os.makedirs(config['save_base'], exist_ok=True)

print(">>> MULTI-OUTPUT DRIFT DETECTION MODE")
print(f"    Strategy: {DRIFT_STRATEGY.upper()}")
print(f"    Drift Threshold (MAE): {drift_threshold}")

# ==========================================
# 2. Train / Validation Split
# ==========================================
def train_val_split(X, y, val_hours):
    if len(X) <= val_hours:
        split = int(len(X) * 0.8)
    else:
        split = len(X) - val_hours
    return X[:split], y[:split], X[split:], y[split:]

# ==========================================
# 3. Initialization
# ==========================================
forecaster = Forecaster(config)
X_all, y_all = forecaster.preprocess()

split_idx = int(len(X_all) * 0.8)

X_train, y_train = X_all[:split_idx], y_all[:split_idx]
X_test, y_test = X_all[split_idx:], y_all[split_idx:]

print(f"Total samples: {len(X_all)}")
print(f"Initial train size: {len(X_train)}")

# ==========================================
# 4. Initial Training
# ==========================================
print(">>> Phase 1: Initial Training")
X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train, val_window_hours)
forecaster.fit(X_tr, y_tr, X_val, y_val)

# ==========================================
# 5. Drift-Aware Prediction Loop (Optimized)
# ==========================================
print(">>> Phase 2: Drift-Aware Forecasting (Optimized Partial Refit)")

all_predictions = []
all_truths = []
drift_events = []

prev_pred = None
start_time = time.time()

# Increase cooldown to avoid too frequent partial refits
COOLDOWN_STEPS = 2 * horizon_hours
last_drift_step = -np.inf

for i in range(len(X_test)):

    # --------------------------
    # A. Drift Detection (rolling window)
    # --------------------------
    if prev_pred is not None:
        # Use last 3 predictions for drift detection to reduce false alarms
        window_size = min(3, i)
        truth_window = y_test[i - window_size:i]
        pred_window = np.array(all_predictions[-window_size:])
        drift, mae = forecaster.detect_drift(truth_window, pred_window, threshold=drift_threshold)

        if drift and (i - last_drift_step) >= COOLDOWN_STEPS:
            print(f"[!] DRIFT @ sample {i} | MAE={mae:.2f}")
            drift_events.append(i)
            last_drift_step = i

            # --------------------------
            # Partial Refit Strategy
            # --------------------------
            if DRIFT_STRATEGY == 'partial refit':
                recent_window = 24 * 3  # last 3 days
                start = max(0, split_idx + i - recent_window)

                X_recent = X_all[start:split_idx + i]
                y_recent = y_all[start:split_idx + i]

                forecaster.partial_refit(
                    X_recent,
                    y_recent,
                    n_new_trees=5  # small increment
                )

            # --------------------------
            # Other strategies if needed
            # --------------------------
            elif DRIFT_STRATEGY == 'full refit':
                X_tr, y_tr, X_val, y_val = train_val_split(
                    X_all[:split_idx + i],
                    y_all[:split_idx + i],
                    val_window_hours
                )
                forecaster.fit(X_tr, y_tr, X_val, y_val)

            elif DRIFT_STRATEGY == 'retrain':
                window = 24 * 365  # 1 year
                start = max(0, split_idx + i - window)
                X_tr, y_tr, X_val, y_val = train_val_split(
                    X_all[start:split_idx + i],
                    y_all[start:split_idx + i],
                    val_window_hours
                )
                forecaster.fit(X_tr, y_tr, X_val, y_val)

    # --------------------------
    # B. Forecast Step
    # --------------------------
    X_curr = X_test[i:i+1]
    pred = forecaster.predict(X_curr)[0]

    all_predictions.append(pred)
    all_truths.append(y_test[i])
    prev_pred = pred

    if i % 50 == 0:
        print(f"Step {i}/{len(X_test)} | Last MAE: {mae if 'mae' in locals() else 'N/A'}")

print(f"Simulation finished in {time.time() - start_time:.2f}s")

# final evaluation
preds = np.array(all_predictions)
truths = np.array(all_truths)

rmse = mean_squared_error(truths, preds) ** 0.5
mae = mean_absolute_error(truths, preds)

print("\n==========================================")
print(f"FINAL RESULTS ({DRIFT_STRATEGY.upper()})")
print("==========================================")
print(f"Drift events: {len(drift_events)}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

np.random.seed(42)
plot_indices = np.random.choice(len(preds), min(4, len(preds)), replace=False)

for ax, idx in zip(axes, plot_indices):
    ax.plot(truths[idx], 'o-', label='Truth')
    ax.plot(preds[idx], 'x--', label='Prediction')
    ax.set_title(f"Sample {idx}")
    ax.grid(alpha=0.3)
    ax.legend()

plt.suptitle(
    f"Multi-output XGBoost | {DRIFT_STRATEGY} | "
    f"Drifts={len(drift_events)} | RMSE={rmse:.2f}",
    fontsize=16
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

save_path = os.path.join(
    config['save_base'],
    f"multioutput_{STATION_NAME}_{DRIFT_STRATEGY}.png"
)
plt.savefig(save_path)
plt.show()

print(f"Plot saved to: {save_path}")
