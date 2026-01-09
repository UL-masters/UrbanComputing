import pandas as pd
import numpy as np
import xgboost as xgb

class Forecaster:

    def __init__(self, config):
        self.save_base = config['save_base']
        self.debug_mode = config['debug_mode']
        self.multiple = config['multiple']
        self.max_window_size = config['max_window_size']
        self.horizon = config['horizon']
        self.target_column = config['target_column']
        self.data_path = config['data_path']
        self.model = None
        self.feature_cols = None 

    def fit(self, X_train, y_train, X_val=None, y_val=None, warm_start=False):
        """
        Supports Drift Adaptation:
        - warm_start=True: Partial Refit. We enable sklearn's warm_start to increment trees.
        - warm_start=False: Full Refit / Initial. We reset and train from scratch.
        """
        # 1. Log Transform
        y_train_log = np.log1p(y_train)
        
        # Configuration
        base_estimators = 50 if self.debug_mode else 500
        
        early_stopping_rounds = None
        eval_set = None
        
        # Validation set logic
        if X_val is not None and y_val is not None:
            y_val_log = np.log1p(y_val)
            eval_set = [(X_train, y_train_log), (X_val, y_val_log)]
            if not warm_start:
                early_stopping_rounds = 50
        
        # Drift Adaptation Logic
        
        if warm_start and self.model is not None:
            print("    >>> [Partial Refit] Updating existing model weights...")
            
            # 1. Get current tree count
            xgb_model = self.model.get_booster()
            current_trees = xgb_model.num_boosted_rounds()
            
            # 2. Calculate new target
            new_n_estimators = current_trees + 10
            
            print(f"        Extending model: {current_trees} -> {new_n_estimators} trees")
            
            self.model.set_params(
                n_estimators=new_n_estimators,
                warm_start=True, 
                early_stopping_rounds=None
            )
            
            # The instance already holds the booster internally because we reused 'self.model'.
            self.model.fit(
                X_train, y_train_log, 
                eval_set=eval_set, 
                verbose=False
            )
            
        else:
            # Fresh Training
            if warm_start:
                print("    >>> Warning: No existing model found. Training from scratch.")
            print(f"    >>> Training new model (N={base_estimators})...")

            # Initialize New Model (Default warm_start=False)
            self.model = xgb.XGBRegressor(
                n_estimators=base_estimators, 
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1, 
                random_state=42,
                early_stopping_rounds=early_stopping_rounds,
                warm_start=False # Ensure fresh start
            )
            
            # Train
            self.model.fit(
                X_train, y_train_log, 
                eval_set=eval_set, 
                verbose=False
            )

    def detect_drift(self, y_true, y_pred, threshold=15.0):
        # Calculate Mean Absolute Error for the recent batch
        mae = np.mean(np.abs(y_true - y_pred))
        
        drift_detected = mae > threshold
        return drift_detected, mae

    def preprocess(self):      
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Time processing
        if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        else:
            df['date'] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        # Feature Engineering
        # 1. Cyclical Time
        df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
        
        # 2. Rolling Stats
        target = self.target_column
        for w in [6, 12, 24]:
            df[f'roll_mean_{w}'] = df[target].rolling(window=w).mean()
            df[f'roll_std_{w}'] = df[target].rolling(window=w).std()
        df = df.fillna(method='bfill')

        self.feature_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                             'roll_mean_6', 'roll_mean_12', 'roll_mean_24',
                             'roll_std_6', 'roll_std_12', 'roll_std_24']
        
        if self.multiple:
            weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            available = [c for c in weather_cols if c in df.columns]
            self.feature_cols.extend(available)
        
        return df

    def predict_single_step_vector(self, current_history, exogenous_features):
        input_vector = []
        input_vector.extend(current_history[-self.max_window_size:]) 
        input_vector.extend(exogenous_features)
        
        input_matrix = np.array([input_vector])
        pred_log = self.model.predict(input_matrix)[0]
        return np.expm1(pred_log)

    def _make_instances(self, df, horizon):
        X, y = [], []
        target = df[self.target_column].values
        feats = df[self.feature_cols].values
        
        for i in range(self.max_window_size, len(df) - horizon + 1):
            sample = []
            sample.extend(target[i-self.max_window_size : i]) 
            sample.extend(feats[i]) 
            X.append(sample)
            y.append(target[i]) 
                
        return np.array(X), np.array(y)