import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

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
        Multi-output training for horizon prediction
        """
        # Log-transform targets
        y_train_log = np.log1p(y_train)
        eval_set = None
        early_stopping_rounds = None

        if X_val is not None and y_val is not None:
            y_val_log = np.log1p(y_val)
            eval_set = [(X_train, y_train_log), (X_val, y_val_log)]
            early_stopping_rounds = 50

        base_estimators = 50 if self.debug_mode else 500

        xgb_base = xgb.XGBRegressor(
            n_estimators=base_estimators,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )

        # Wrap in MultiOutputRegressor for horizon-length prediction
        self.model = MultiOutputRegressor(xgb_base, n_jobs=-1)

        if eval_set:
            self.model.fit(X_train, y_train_log)
        else:
            self.model.fit(X_train, y_train_log)

        print(f"Model trained. Input shape: {X_train.shape}, Output shape: {y_train.shape}")
        
    def partial_refit(self, X_new, y_new, n_new_trees=5):
        """
        Incrementally update only first few horizons with a small subsample.
        """
        MAX_REFIT_HORIZON = min(6, y_new.shape[1])  # Update first 6 horizons only
        SUBSAMPLE_SIZE = 32  # small random batch

        # Subsample to speed up
        idx = np.random.choice(len(X_new), size=min(SUBSAMPLE_SIZE, len(X_new)), replace=False)
        X_sub = X_new[idx]
        y_sub = y_new[idx]

        for h, estimator in enumerate(self.model.estimators_[:MAX_REFIT_HORIZON]):
            booster = estimator.get_booster()
            current_trees = booster.num_boosted_rounds()

            estimator.set_params(
                n_estimators=current_trees + n_new_trees,
                warm_start=True
            )

            # Fit only on small subsample
            estimator.fit(
                X_sub,
                np.log1p(y_sub[:, h]),
                xgb_model=booster,
                verbose=False
            )

        print(f"Partial refit done for first {MAX_REFIT_HORIZON} horizons with {n_new_trees} new trees each")

   
    def predict(self, X_test):
        y_pred_log = self.model.predict(X_test)
        return np.expm1(y_pred_log)  

    
    def detect_drift(self, y_true, y_pred, threshold=12.0):
        error = np.mean(np.abs(y_true - y_pred))
        return error > threshold, error

    
    def preprocess(self):      
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Timestamp
        if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        else:
            df['date'] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values('date').reset_index(drop=True)
        
        # handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        # cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

        # rolling statistics
        target = self.target_column
        for w in [6, 12, 24]:
            df[f'roll_mean_{w}'] = df[target].rolling(window=w).mean()
            df[f'roll_std_{w}'] = df[target].rolling(window=w).std()
        df = df.fillna(method='bfill')

        # features
        self.feature_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                             'roll_mean_6', 'roll_mean_12', 'roll_mean_24',
                             'roll_std_6', 'roll_std_12', 'roll_std_24']
        if self.multiple:
            weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            available_weather = [c for c in weather_cols if c in df.columns]
            self.feature_cols.extend(available_weather)

        #  multi-output instances
        X, y = self.create_instances(df)
        return X, y
    


    def create_instances(self, df):
        X, y = [], []
        target_vals = df[self.target_column].values
        feature_vals = df[self.feature_cols].values

        n_samples = len(df)
        for i in range(self.max_window_size, n_samples - self.horizon + 1):
            # History + features
            x_i = list(target_vals[i-self.max_window_size:i]) + list(feature_vals[i-1])
            X.append(x_i)
            # Multi-step targets
            y.append(target_vals[i:i+self.horizon])

        X = np.array(X)
        y = np.array(y)
        print(f"Created instances: X={X.shape}, y={y.shape}")
        return X, y
