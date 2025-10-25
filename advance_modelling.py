import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
import warnings
import os 
from tqdm import tqdm

# Suppress harmless warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
PRICE_COLUMN = 'Reliance_Close' 
TARGET_PRICE = 'target'
ADVANCED_RESULTS_FILE = 'advanced_model_results.csv' # Output file name
TRAIN_SPLIT_RATIO = 0.8
N_STEPS = 60 # Time steps (past days) for LSTM input
# ---------------------

def load_and_split_data(filepath):
    """Loads the data and prepares the train/test split (chronological)."""
    print(f"\n--- Loading and Splitting Data from '{filepath}' ---")
    df = pd.read_csv(filepath, parse_dates=True, index_col=0).dropna()

    X = df.drop(columns=[TARGET_PRICE, 'direction'], errors='ignore')
    Y = df[TARGET_PRICE] 

    X = X.select_dtypes(include=np.number)

    # Chronological Split
    train_size = int(len(df) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]

    print(f"✅ Data Split: Train size = {len(X_train)}, Test size = {len(X_test)}")
    return X_train, X_test, Y_train, Y_test, X.columns.tolist()

# ==============================================================================
# 1. LONG SHORT-TERM MEMORY (LSTM) MODEL
# ==============================================================================

def prepare_lstm_data(X_train, X_test, Y_train, Y_test, n_steps=N_STEPS):
    """
    Scales and reshapes data into the 3D format required by LSTM:
    [samples, timesteps, features].
    """
    print(f"\n--- Preparing Data for LSTM (N_STEPS={n_steps}) ---")
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_train_scaled = scaler_Y.fit_transform(Y_train.values.reshape(-1, 1))
    
    def create_sequences(X, Y): 
        Xs = []
        for i in range(len(X) - n_steps):
            Xs.append(X[i:i + n_steps]) # Input: sequence of n_steps
        Y_eval = Y.iloc[n_steps:]
        return np.array(Xs), Y_eval

    X_train_seq, _ = create_sequences(X_train_scaled, Y_train) 
    Y_train_seq = Y_train_scaled[N_STEPS:] 
    
    X_test_seq, Y_test_eval = create_sequences(X_test_scaled, Y_test)
    
    print(f"✅ LSTM Data Ready: X_train_seq shape: {X_train_seq.shape}")
    return X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y

