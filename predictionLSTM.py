import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import os 
import warnings
# Suppress harmless warnings and TensorFlow verbose output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
TARGET_PRICE = 'target'
TRAIN_SPLIT_RATIO = 0.8
N_STEPS = 60         # Time steps (past days) for LSTM input
N_FUTURE_DAYS = 15   # Constant for the forecast horizon
# ---------------------
def load_and_split_data(filepath):
    """Loads and splits data chronologically."""
    print(f"\n--- Loading Data from '{filepath}' ---")
    df = pd.read_csv(filepath, parse_dates=True, index_col=0).dropna()
    
    X = df.drop(columns=[TARGET_PRICE, 'direction'], errors='ignore').select_dtypes(include=np.number)
    Y = df[TARGET_PRICE]
    
    train_size = int(len(df) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]
    
    print(f"✅ Train size = {len(X_train)}, Test size = {len(X_test)}")
    return X_train, X_test, Y_train, Y_test
def prepare_multi_output_lstm_data(X_train, X_test, Y_train, Y_test, n_steps=N_STEPS, future_steps=N_FUTURE_DAYS):
    print(f"\n--- Preparing Data for LSTM (future steps={future_steps}) ---")
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_all_scaled = np.vstack((X_train_scaled, X_test_scaled))

    Y_full = pd.concat([Y_train, Y_test])
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_full_scaled = scaler_Y.fit_transform(Y_full.values.reshape(-1, 1))
