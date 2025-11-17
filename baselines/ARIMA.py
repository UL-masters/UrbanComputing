import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "..", "data")
results_folder = os.path.join(script_dir, "..", "results")
os.makedirs(results_folder, exist_ok=True)

print("ARIMA 365-day Forecasting Evaluation\n")

csv_files = [f for f in os.listdir(data_folder) if f.startswith("PRSA_Data_") and f.endswith(".csv")]
print(f"Found {len(csv_files)} stations\n")

metrics = []

for file in csv_files:
    station = file.split("_")[2].split("_2013")[0]
    print(f"=== {station} ===")

    # Load
    df = pd.read_csv(os.path.join(data_folder, file))
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('date')
    series = df['PM2.5'].resample('D').mean().ffill().bfill()

    if len(series) < 730:
        print("   Not enough data")
        continue

    train = series[:-365]          # All except last 365 days
    test  = series[-365:]          # Last 365 days = ground truth

    print(f"   Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"   Test : {test.index[0].date()} to {test.index[-1].date()} (365 days)")

    # Fit auto_arima
    print("   Fitting auto_arima...", end=" ")
    model = auto_arima(
        train,
        seasonal=True, m=7,
        stepwise=True,
        suppress_warnings=True,
        trace=False,
        maxiter=100,
        error_action='ignore'
    )
    print("Done")

    # Forecast
    forecast, conf_int = model.predict(n_periods=365, return_conf_int=True)
    forecast = pd.Series(forecast, index=test.index)

    # Metrics
    mse  = mean_squared_error(test, forecast)
    mae  = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mse)

    print(f"   RMSE = {rmse:6.1f} | MAE = {mae:6.1f} | MSE = {mse:7.1f}\n")

    metrics.append({
        'Station': station,
        'RMSE': round(rmse, 1),
        'MAE':  round(mae, 1),
        'MSE':  round(mse, 1)
    })

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(train[-730:], label="Training (last 2 years)", color="gray", alpha=0.8)
    plt.plot(test, label="Actual (last 365 days)", color="#1f77b4", linewidth=2.5)
    plt.plot(forecast, label="auto_arima Forecast", color="red", linewidth=2.5)
    plt.fill_between(test.index, conf_int[:,0], conf_int[:,1], color="red", alpha=0.15)

    plt.title(f"PM2.5 365-Day Forecast vs Actual - {station}\n"
              f"RMSE = {rmse:.1f} µg/m³ | MAE = {mae:.1f} µg/m³", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/Final_{station}.png", dpi=300, bbox_inches='tight')
    plt.close()

# === FINAL TABLE - exactly like Tetteroo et al. (2022) ===
df_results = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
print("\n" + "="*80)
print("FINAL RESULTS - auto_arima vs Last 365 Days (Hold-Out Test)")
print("="*80)
print(df_results.to_string(index=False))

# Save
df_results.to_csv(f"{results_folder}/ARIMA_Final_Results.csv", index=False)
df_results.to_latex(f"{results_folder}/ARIMA_Final_Results.tex", index=False,
                    caption="ARIMA Baseline Performance (365-day Forecast)")
