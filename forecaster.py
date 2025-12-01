import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class Forecaster:

    def __init__(self, config):
        self.save_base = config['save_base']
        self.debug_mode = config['debug_mode']
        self.multiple = config['multiple']
        self.max_window_size = config['max_window_size']
        self.horizon = config['horizon']
        self.test_days = config['test_days']
        self.target_column = config['target_column']
        self.data_path = config['data_path']
        self.model = None

    def fit(self, X_train, y_train):
        print(f"Training XGBoost Model...")
        # 针对 MAC M芯片优化参数
        self.model = xgb.XGBRegressor(
            n_estimators=500 if not self.debug_mode else 10, 
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1, # 使用所有核心
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("Training finished.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def preprocess(self):      
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # processing timestamp
        if 'year' in df.columns and 'hour' in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        else:
            df['date'] = pd.to_datetime(df.iloc[:, 0])
            
        df = df.sort_values('date').reset_index(drop=True)
        
        # processing missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    
        target_col = self.target_column
        multiple_columns = None
        
        if self.multiple:
            # Select weather-related columns 
            possible_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            weather_cols = [c for c in possible_cols if c in df.columns]
            if weather_cols:
                multiple_columns = weather_cols
                print(f"Using weather features: {multiple_columns}")

        print("Creating instances (Sliding Window)...")
        X, y = create_instances(df, self.max_window_size, self.horizon, 
                                target_column=target_col, 
                                multiple_columns=multiple_columns)
        
        return X, y

# helpter function to create sliding window instances
def create_instances(df, max_window_size, horizon, target_column, multiple_columns):
    X = []
    y = []
    
    data_values = df[target_column].values
    
    if multiple_columns is not None:
        weather_values = df[multiple_columns].values
    
    n_samples = len(df)
    
    for i in range(max_window_size, n_samples - horizon + 1):
        # Input: past max_window_size hours
        current_sample = []
    
        current_sample.extend(data_values[i-max_window_size : i])
        
        # add weather history
        if multiple_columns is not None:
            weather_window = weather_values[i-max_window_size : i]
            current_sample.extend(weather_window.flatten())
            
        X.append(current_sample)
        
        # Output: future horizon hour
        target_val = data_values[i + horizon - 1]
        y.append(target_val)

    X = np.array(X)
    y = np.array(y)
    print(f"Data shape created: X={X.shape}, y={y.shape}")
    return X, y