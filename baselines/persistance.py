import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "..", "data")
results_folder = os.path.join(script_dir, "..", "results/Persistance")
os.makedirs(results_folder, exist_ok=True)

print("Persistence (Naive) Baseline -> 365-day Forecast\n")

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
        continue

    # last 365 days = test
    train = series[:-365]
    test  = series[-365:]

    persistence_value = train.iloc[-1]                    # x_t
    forecast = np.full(shape=365, fill_value=persistence_value)

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae  = mean_absolute_error(test, forecast)

    print(f"   Persistence value = {persistence_value:.1f}")
    print(f"   RMSE = {rmse:5.1f} | MAE = {mae:5.1f}\n")

    metrics.append({'Station': station, 'RMSE': round(rmse,1), 'MAE': round(mae,1)})

    # plot
    plt.figure(figsize=(14,6))
    plt.plot(train[-730:], label="Training (last 2 years)", color="gray")
    plt.plot(test, label="Actual (last 365 days)", color="#1f77b4", linewidth=2.5)
    plt.axhline(y=persistence_value, color="#e74c3c", linewidth=2.5, label="Persistence Forecast")

    plt.title(f"Persistence (Naive) Baseline — {station}\n"
              f"RMSE = {rmse:.1f} | MAE = {mae:.1f}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/Persistence_{station}.png", dpi=300, bbox_inches='tight')
    plt.close()

# === FINAL TABLE ===
df_results = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
print("\n" + "="*70)
print("PERSISTENCE BASELINE -> 365-day forecast")
print("="*70)
print(df_results.to_string(index=False))

df_results.to_csv(f"{results_folder}/Persistence_Results.csv", index=False)