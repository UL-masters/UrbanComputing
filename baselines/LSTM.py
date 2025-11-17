import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Force CPU + silence TF (optional but clean)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.random.set_seed(42)
np.random.seed(42)

# === PATHS ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "..", "data")
results_folder = os.path.join(script_dir, "..", "results/LSTM")
os.makedirs(results_folder, exist_ok=True)

print("LSTM 365-day Forecast\n")

csv_files = [f for f in os.listdir(data_folder) if f.startswith("PRSA_Data_") and f.endswith(".csv")]
print(f"Found {len(csv_files)} stations\n")

metrics = []
seq_length = 60

for file in csv_files:
    station = file.split("_")[2].split("_2013")[0]
    print(f"=== {station} ===")

    # Load + prepare
    df = pd.read_csv(os.path.join(data_folder, file))
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('date')
    values = df['PM2.5'].resample('D').mean().ffill().bfill().values.astype('float32')

    if len(values) < 730:
        continue

    log_values = np.log1p(values)
    train_log = log_values[:-365]
    test_log  = log_values[-365:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_log.reshape(-1, 1))
    test_scaled  = scaler.transform(test_log.reshape(-1, 1))

    # sequences
    def create_seq(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled, seq_length)
    X_test,  y_test  = create_seq(test_scaled,  seq_length)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

    # LSTM
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("   Training...", end=" ")
    model.fit(X_train, y_train,
              epochs=200, batch_size=32,
              validation_split=0.1,
              callbacks=[early_stop], verbose=0)
    print(f"Done ({len(model.history.history['loss'])} epochs)")

    # Predict
    pred_scaled = model.predict(X_test, verbose=0)
    pred = np.expm1(scaler.inverse_transform(pred_scaled).flatten())
    actual = np.expm1(scaler.inverse_transform(y_test.reshape(-1,1)).flatten())

    # Metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)

    print(f"   RMSE = {rmse:5.1f} | MAE = {mae:5.1f}\n")

    metrics.append({'Station': station, 'RMSE': round(rmse,1), 'MAE': round(mae,1)})

    dates = df['PM2.5'].resample('D').mean().index
    train_dates = dates[:-365]
    test_dates  = dates[-365:]

    forecast_dates = test_dates[seq_length : seq_length + len(pred)]

    plt.figure(figsize=(14,6))
    plt.plot(train_dates[-730:], np.expm1(log_values[:-365])[-730:], color='gray', label='Training')
    plt.plot(test_dates,  np.expm1(log_values[-365:]), color='#1f77b4', linewidth=2.5, label='Actual')
    plt.plot(forecast_dates, pred, color='#e74c3c', linewidth=2.5, label='LSTM Forecast')

    plt.title(f"LSTM 365-Day Forecast — {station}\nRMSE = {rmse:.1f} µg/m³ | MAE = {mae:.1f} µg/m³", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/LSTM_{station}_FINAL.png", dpi=300, bbox_inches='tight')
    plt.close()

# === FINAL TABLE ===
df_results = pd.DataFrame(metrics).sort_values("RMSE")
print("\n" + "="*70)
print("LSTM RESULTS")
print("="*70)
print(df_results.to_string(index=False))

df_results.to_csv(f"{results_folder}/LSTM_FINAL_RESULTS.csv", index=False)