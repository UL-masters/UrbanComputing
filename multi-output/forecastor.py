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

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"Training XGBoost Model (Multi-output)...")
        
        #  Target Transformation: Log-transform
        y_train_log = np.log1p(y_train)
        
        eval_set = None
        early_stopping_rounds = None # default: no early stopping
        
        if X_val is not None and y_val is not None:
            y_val_log = np.log1p(y_val)
            eval_set = [(X_train, y_train_log), (X_val, y_val_log)]
            early_stopping_rounds = 50 # enable early stopping
            print("Validation set provided. Early stopping enabled.")
        
        # Model Initialization
        n_estimators = 50 if self.debug_mode else 1000
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators, 
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1, 
            random_state=42,
            early_stopping_rounds=early_stopping_rounds 
        )
        
        # train the model
        if eval_set:
            self.model.fit(
                X_train, y_train_log,
                eval_set=eval_set,
                verbose=100
            )
        else:
            self.model.fit(X_train, y_train_log, verbose=True)
            
        print("Training finished.")

    def predict(self, X_test):
        # Predictions are in the log space
        pred_log = self.model.predict(X_test)
        # Convert back to the original space (exp - 1)
        return np.expm1(pred_log)
    
    # -----------------------------
    # Drift Detection
    # -----------------------------
    def detect_drift(self, y_true, y_pred, threshold=12.0):
        error = np.mean(np.abs(y_true - y_pred))
        return error > threshold, error


    # -----------------------------
    # Full Refit 
    # -----------------------------
    def full_refit(self, X_all, y_all):
        print("FULL REFIT triggered")

        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )

        self.model.fit(X_all, np.log1p(y_all))


    # -----------------------------
    # Partial Refit 
    # -----------------------------
    def partial_refit(self, X_new, y_new):
        print("PARTIAL REFIT triggered")

        self.model.fit(
            X_new,
            np.log1p(y_new),
            xgb_model=self.model
        )


    # -----------------------------
    # Retrain 
    # -----------------------------
    def retrain_recent(self, X_all, y_all, window_size=5000):
        print("RECENT-WINDOW RETRAIN triggered")

        X_recent = X_all[-window_size:]
        y_recent = y_all[-window_size:]

        self.model = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )

        self.model.fit(X_recent, np.log1p(y_recent))

    def preprocess(self):      
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # timestamp processing
        if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        else:
            df['date'] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values('date').reset_index(drop=True)
        
        # handle missing vals 
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        # feature engineering
        print("Generating advanced features...")
        
        # time cycle encoding
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

        feature_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                        'roll_mean_6', 'roll_mean_12', 'roll_mean_24',
                        'roll_std_6', 'roll_std_12', 'roll_std_24']
        
        if self.multiple:
            weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            available_weather = [c for c in weather_cols if c in df.columns]
            feature_cols.extend(available_weather)
            print(f"Using weather & extra features: {feature_cols}")

        print(f"Creating Multi-output instances (Horizon={self.horizon})...")
        X, y = create_instances(df, self.max_window_size, self.horizon, 
                                target_column=self.target_column, 
                                feature_cols=feature_cols)
        
        return X, y

def create_instances(df, max_window_size, horizon, target_column, feature_cols):
    X = []
    y = []
    
    data_values = df[target_column].values
    feature_values = df[feature_cols].values
    
    n_samples = len(df)
    
    for i in range(max_window_size, n_samples - horizon + 1):
        current_sample = []
        
        # Historical target values
        current_sample.extend(data_values[i-max_window_size : i])
        
        # Current features
        current_features = feature_values[i-1] 
        current_sample.extend(current_features)
            
        X.append(current_sample)
        
        # Target sequence
        target_seq = data_values[i : i + horizon]
        y.append(target_seq)

    X = np.array(X)
    y = np.array(y)
    print(f"Data shape created: X={X.shape}, y={y.shape}") 
    return X, y