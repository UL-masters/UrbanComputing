import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# Configuration
# -----------------------------
data_path = 'data/PRSA_Data_Wanshouxigong_20130301-20170228.csv'
try:
    file_name = os.path.basename(data_path)
    station_name = file_name.split('_')[2]
except:
    station_name = "UnknownStation"

output_folder = 'results/ARIMA_multi'
os.makedirs(output_folder, exist_ok=True)

horizon_hours = 24 # forecast horizon
arima_order = (5, 1, 0)  # adjust as needed
target_column = 'PM2.5'

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(data_path)
series = df[target_column].interpolate().ffill().bfill().values

# Split train/test
test_size = int(len(series) * 0.2)
train_series = series[:-test_size]
test_series = series[-test_size:]

# -----------------------------
# Fit ARIMA
# -----------------------------
print("Fitting ARIMA...")
model = ARIMA(train_series, order=arima_order)
model_fit = model.fit()
print(model_fit.summary())

# -----------------------------
# Multi-step forecasts for plotting
# -----------------------------
preds_list = []
truth_list = []
window_size = horizon_hours

# Slide window across test set
for start_idx in range(0, len(test_series) - window_size, window_size):
    history = series[:len(train_series) + start_idx]
    forecast = model_fit.apply(history).forecast(steps=window_size)
    preds_list.append(forecast)
    truth_list.append(test_series[start_idx:start_idx + window_size])

preds_array = np.array(preds_list)
truth_array = np.array(truth_list)

# -----------------------------
# Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(truth_array.flatten(), preds_array.flatten()))
mae = mean_absolute_error(truth_array.flatten(), preds_array.flatten())
print(f"\nEvaluation Results (Horizon={horizon_hours}):")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")

# -----------------------------
# Plotting (like your LSTM plots)
# -----------------------------
def plot_trajectory(ax, truth, prediction, horizon, sample_idx):
    hours = range(1, horizon + 1)
    ax.plot(hours, truth, color='#ff7f0e', label='Truth', linewidth=2, marker='o', markersize=3)
    ax.plot(hours, prediction, color='#1f77b4', label='Prediction', linewidth=2, marker='x', markersize=3)
    error_std = np.std(truth - prediction)
    ax.fill_between(hours, prediction - 0.5*error_std, prediction + 0.5*error_std, color='#1f77b4', alpha=0.15)
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel(target_column)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Sample #{sample_idx}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

np.random.seed(42)
indices = np.random.choice(len(preds_array), 4, replace=False)

for i, idx in enumerate(indices):
    plot_trajectory(axes[i], truth_array[idx], preds_array[idx], horizon=horizon_hours, sample_idx=idx)

plt.suptitle(f"Multi-step ARIMA ({horizon_hours}h) Forecast for {station_name}\n"
             f"RMSE: {rmse:.2f}", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

save_path = os.path.join(output_folder, f"{station_name}_ARIMA_multi.png")
plt.savefig(save_path, dpi=300)
plt.show()
print(f"Chart saved to: {save_path}")
