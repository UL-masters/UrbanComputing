# ============================
# run_arima_forecast.py
# ARIMA/SARIMAX Forecasting Runner (same style as your XGBoost script)
# ============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import our ARIMA forecaster (save the previous class as arima_forecaster.py)
from ARIMA import ARIMAForecaster  # ← Make sure this file exists!

# ============================
# Automatic path and station name setup
# ============================
data_path = 'data/PRSA_Data_Aotizhongxin_20130301-20170228.csv'  # Change if needed

file_name = os.path.basename(data_path)
station_name = file_name.split('_')[2]  # e.g., Wanshouxigong
print(f"Station detected: {station_name}")

# Output folder
output_folder = 'results/ARIMA'
os.makedirs(output_folder, exist_ok=True)

# ============================
# Configuration (same style as XGBoost)
# ============================
config = {
    'save_base': output_folder,
    'debug_mode': False,                  # Set True for quick testing
    'multiple': True,                     # Use weather features → SARIMAX
    'max_window_size': 168,               # 7 days of history (recommended for ARIMA)
    'horizon': 24,                        # Predict next 24 hours ahead
    'test_days': 180,                     # Last 180 days as test (approx 6 months)
    'target_column': 'PM2.5',
    'data_path': data_path,
    'arima_order': (5, 1, 1),             # Good default
    'seasonal_order': (1, 1, 1, 24),      # Daily seasonality (hourly data)
    'trend': 'c'                          # Include constant trend
}

# ============================
# Initialize & Run Forecasting
# ============================
print(">>> Starting ARIMA/SARIMAX Forecasting Pipeline...")
forecaster = ARIMAForecaster(config)

# Preprocess: creates train/test with proper exog handling
(X_train, X_test, y_train_series, y_test,
 exog_train, exog_test) = forecaster.preprocess()

print(f"Training period  : up to {y_train_series.index[-1]}")
print(f"Test period starts: {pd.Series(y_test).index[0] if hasattr(pd.Series(y_test), 'index') else 'N/A'}")
print(f"Forecast horizon : {config['horizon']} hours")

# Train the model
print(">>> Step 1: Fitting ARIMA/SARIMAX model...")
forecaster.fit(y_train_series, exog_train=exog_train)

# Predict
print(f">>> Step 2: Forecasting next {config['horizon']} hours...")
forecast_result = forecaster.predict(horizon=config['horizon'], exog_future=exog_test)

y_pred = forecast_result['pred']
lower = forecast_result['lower']
upper = forecast_result['upper']

# ============================
# Evaluation
# ============================
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nSUCCESS! Evaluation Results for {station_name}:")
print(f"RMSE        : {rmse:.4f}")
print(f"MAE         : {mae:.4f}")
print(f"Model       : {'SARIMAX' if config['multiple'] else 'ARIMA'}")
print(f"Horizon     : {config['horizon']} hours")

# ============================
# Plotting Results
# ============================
plt.figure(figsize=(14, 7))

# Plot last 200 points for clarity
plot_len = min(200, len(y_test))
idx = np.arange(len(y_test))[-plot_len:]

plt.plot(idx, y_test[-plot_len:], label='True PM2.5', color='blue', linewidth=2)
plt.plot(idx, y_pred[-plot_len:], label='Predicted PM2.5 (ARIMA)', color='red', linestyle='--', linewidth=2)
plt.fill_between(idx, lower[-plot_len:], upper[-plot_len:], color='red', alpha=0.15, label='95% Confidence Interval')

plt.title(f"PM2.5 Forecast - {station_name}\n"
          f"ARIMA/SARIMAX | Horizon={config['horizon']}h | RMSE={rmse:.2f}", fontsize=14)
plt.xlabel("Hours into Test Period")
plt.ylabel("PM2.5 Concentration (μg/m³)")
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot
plot_path = os.path.join(output_folder, f"{station_name}_ARIMA_forecast_h{config['horizon']}.png")
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
print(f"Chart saved: {plot_path}")

# Optional: Save predictions to CSV
results_df = pd.DataFrame({
    'true': y_test,
    'pred': y_pred,
    'lower': lower,
    'upper': upper
})
csv_path = os.path.join(output_folder, f"{station_name}_predictions_h{config['horizon']}.csv")
results_df.to_csv(csv_path, index=False)
print(f"Predictions saved: {csv_path}")

print("\nARIMA forecasting completed successfully!")