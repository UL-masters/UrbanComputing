import pandas as pd
import numpy as np
import xgboost as xgb

class Forecaster:

    def __init__(self, config):
        self.save_base = config['save_base']
        self.debug_mode = config['debug_mode']
        self.multiple = config['multiple']
        self.max_window_size = config['max_window_size']
        self.horizon = config['horizon'] # This is mainly for inference loop count
        self.target_column = config['target_column']
        self.data_path = config['data_path']
        self.model = None
        self.feature_cols = None 

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the model using the Holdout strategy described in the paper.
        
        Phase 1: Train on X_train, validate on X_val to find optimal n_estimators (Early Stopping).
        Phase 2 (Refit): Retrain on (X_train + X_val) using the best n_estimators found in Phase 1.
        """
        print(f"Training XGBoost Model (Repeated Single-Output)...")
        print(f"Strategy: Train-Validation Split -> Early Stopping -> Refit on Full Data")
        
        # 1. Target Transformation: Log-transform to handle skewness
        y_train_log = np.log1p(y_train)
        
        # Default n_estimators
        n_estimators = 50 if self.debug_mode else 1000
        early_stopping_rounds = None
        eval_set = None
        
        # Configure validation if provided
        if X_val is not None and y_val is not None:
            y_val_log = np.log1p(y_val)
            eval_set = [(X_train, y_train_log), (X_val, y_val_log)]
            early_stopping_rounds = 50
            print("Validation set provided. Early stopping enabled.")
        
        # --- Phase 1: Model Selection (Finding best iteration) ---
        temp_model = xgb.XGBRegressor(
            n_estimators=n_estimators, 
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1, 
            random_state=42,
            early_stopping_rounds=early_stopping_rounds
        )
        
        if eval_set:
            temp_model.fit(X_train, y_train_log, eval_set=eval_set, verbose=False)
            best_iteration = temp_model.best_iteration
            print(f"Best iteration found via Early Stopping: {best_iteration}")
        else:
            temp_model.fit(X_train, y_train_log, verbose=False)
            best_iteration = n_estimators
            print(f"No validation set. Using full estimators: {best_iteration}")

        # --- Phase 2: Refit on Full Data (Train + Validation) ---
        # As per paper: "refit the ensembles on the full train and validation set" 
        # This ensures the model learns from the most recent data (Validation set)
        
        print(">>> Refitting model on Combined Data (Train + Validation)...")
        
        if X_val is not None and y_val is not None:
            X_combined = np.concatenate([X_train, X_val], axis=0)
            y_combined_log = np.concatenate([y_train_log, y_val_log], axis=0)
        else:
            X_combined = X_train
            y_combined_log = y_train_log
            
        # Initialize the final model with the optimal number of trees found
        self.model = xgb.XGBRegressor(
            n_estimators=best_iteration, # Use the best N found
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1, 
            random_state=42
            # No early stopping needed here, we run exactly 'best_iteration' rounds
        )
        
        self.model.fit(X_combined, y_combined_log, verbose=False)
        print("Refit complete.")

    def preprocess(self):      
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # --- Timestamp Processing ---
        if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        else:
            df['date'] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values('date').reset_index(drop=True)
        
        # --- Missing Value Imputation ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        # --- Feature Engineering ---
        # 1. Cyclical Time Features (Sin/Cos encoding)
        df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
        
        # 2. Rolling Statistics (Trend & Volatility)
        target = self.target_column
        for w in [6, 12, 24]:
            df[f'roll_mean_{w}'] = df[target].rolling(window=w).mean()
            df[f'roll_std_{w}'] = df[target].rolling(window=w).std()
        df = df.fillna(method='bfill')

        # --- Define Feature List ---
        # Lag features are handled dynamically during recursion.
        # These are exogenous/static features.
        self.feature_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                             'roll_mean_6', 'roll_mean_12', 'roll_mean_24',
                             'roll_std_6', 'roll_std_12', 'roll_std_24']
        
        if self.multiple:
            weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            available = [c for c in weather_cols if c in df.columns]
            self.feature_cols.extend(available)
        
        return df

    def create_train_data(self, df, split_ratio=0.8):
        """
        Prepares data for Single-Step training (Horizon = 1).
        Splits data chronologically into Train and Validation (Holdout).
        """
        # Split total development set (Train + Val) from Test set
        n_dev = int(len(df) * split_ratio)
        dev_df = df.iloc[:n_dev]
        
        # Create instances with horizon=1 (Model learns t -> t+1)
        X, y = self._make_instances(dev_df, horizon=1)
        
        # Further split Development set into Train and Validation (Holdout)
        # Paper strategy: Use the last part of training data as holdout
        val_idx = int(len(X) * 0.9) # Last 10% of development data
        
        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:], y[val_idx:]
        
        return X_train, y_train, X_val, y_val

    def predict_recursive(self, df, test_start_indices):
        """
        Performs Repeated Single-Output (Recursive) forecasting.
        Input: df (full dataframe), test_start_indices (starting points for forecast).
        Output: Matrix of shape (N_samples, Horizon).
        """
        all_predictions = []
        target_vals = df[self.target_column].values
        feat_vals = df[self.feature_cols].values
        
        print(f"Starting Recursive Forecast for {len(test_start_indices)} samples...")
        
        for start_idx in test_start_indices:
            # 1. Initialize history window with actual observed data
            current_history = list(target_vals[start_idx - self.max_window_size : start_idx])
            sample_preds = []
            
            # 2. Recursive Loop
            for h in range(self.horizon):
                # A. Prepare Input: Lags (Target History)
                input_vector = []
                input_vector.extend(current_history[-self.max_window_size:]) 
                
                # B. Prepare Input: Exogenous Features
                # We assume we know the weather/time at t+h (Oracle assumption for simplification)
                current_time_idx = start_idx + h
                if current_time_idx >= len(feat_vals):
                    break # Edge case: end of dataset
                input_vector.extend(feat_vals[current_time_idx])
                
                # C. Predict t+h
                input_matrix = np.array([input_vector])
                pred_log = self.model.predict(input_matrix)[0]
                pred_val = np.expm1(pred_log) # Inverse Log
                
                # D. Store and Update History
                sample_preds.append(pred_val)
                current_history.append(pred_val) # Append PREDICTION to history for next step
            
            all_predictions.append(sample_preds)
            
        return np.array(all_predictions)

    def _make_instances(self, df, horizon):
        """Helper to create supervised learning pairs (X, y)"""
        X, y = [], []
        target = df[self.target_column].values
        feats = df[self.feature_cols].values
        
        for i in range(self.max_window_size, len(df) - horizon + 1):
            sample = []
            sample.extend(target[i-self.max_window_size : i]) # Lags
            sample.extend(feats[i]) # Exogenous features at current step
            X.append(sample)
            y.append(target[i]) # Single step target
                
        return np.array(X), np.array(y)