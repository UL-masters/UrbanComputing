import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# LSTM MULTI-STEP FORECASTER
# ============================================================

class LSTMForecaster:

    def __init__(self, config):
        self.save_base = config['save_base']
        self.debug_mode = config['debug_mode']
        self.multiple = config['multiple']
        self.window_size = config['max_window_size']
        self.horizon = config['horizon']
        self.target_column = config['target_column']
        self.data_path = config['data_path']
        self.model = None

   
    # PREPROCESSING
    def preprocess(self):
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)

        #  Timestamp processing 
        if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        else:
            df['date'] = pd.to_datetime(df.iloc[:, 0])

        df = df.sort_values('date').reset_index(drop=True)

        #  Missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Time encoding 
        df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

        #  Rolling features 
        target = self.target_column
        for w in [6, 12, 24]:
            df[f'roll_mean_{w}'] = df[target].rolling(window=w).mean()
            df[f'roll_std_{w}'] = df[target].rolling(window=w).std()

        df = df.ffill().bfill()

        #  Feature list 
        feature_cols = [
            'hour_sin', 'hour_cos',
            'month_sin', 'month_cos',
            'roll_mean_6', 'roll_mean_12', 'roll_mean_24',
            'roll_std_6', 'roll_std_12', 'roll_std_24'
        ]

        if self.multiple:
            weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            available_weather = [c for c in weather_cols if c in df.columns]
            feature_cols.extend(available_weather)
            print(f"Using weather & extra features: {feature_cols}")

        print(f"Creating LSTM sequences (Input={self.window_size}, Horizon={self.horizon})...")

        X, y = create_lstm_instances(
            df,
            self.window_size,
            self.horizon,
            self.target_column,
            feature_cols
        )

        return X, y


    
    # MODEL DEFINITION
    def build_model(self, input_shape):
        print("Building LSTM model...")

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))

        model.add(LSTM(64))
        model.add(Dropout(0.2))

        model.add(Dense(self.horizon))  # 24-step output

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )

        self.model = model
        print(model.summary())


    
    # TRAINING
    def fit(self, X_train, y_train, X_val, y_val):
        print("Training LSTM model...")

        # Log-transform targets (stabilizes peaks)
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        epochs = 10 if self.debug_mode else 60

        self.model.fit(
            X_train, y_train_log,
            validation_data=(X_val, y_val_log),
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )

        print("LSTM training finished.")


   
    # PREDICTION
    def predict(self, X_test):
        pred_log = self.model.predict(X_test)
        return np.expm1(pred_log)   # inverse log




# LSTM SEQUENCE CREATOR

def create_lstm_instances(df, window_size, horizon, target_column, feature_cols):
    X = []
    y = []

    feature_values = df[feature_cols].values
    target_values = df[target_column].values

    n_samples = len(df)

    for i in range(window_size, n_samples - horizon):
        X.append(feature_values[i - window_size:i])
        y.append(target_values[i:i + horizon])

    X = np.array(X)  # (samples, timesteps, features)
    y = np.array(y)  # (samples, horizon)

    print(f"LSTM Data Shape: X={X.shape}, y={y.shape}")
    return X, y
