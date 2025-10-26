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
    def create_multi_sequences(X_scaled, Y_scaled_full):
        Xs, Ys = [], []
        stop_index = len(X_scaled) - n_steps - future_steps + 1
        for i in range(stop_index):
            Xs.append(X_scaled[i:i+n_steps])
            Ys.append(Y_scaled_full[i+n_steps:i+n_steps+future_steps].flatten())
        return np.array(Xs), np.array(Ys)

    X_seq_all, Y_seq_all = create_multi_sequences(X_all_scaled, Y_full_scaled)
    train_end_index = len(X_train) - n_steps - future_steps + 1
    X_train_seq = X_seq_all[:train_end_index]
    Y_train_seq = Y_seq_all[:train_end_index]
    X_test_seq = X_seq_all[train_end_index:]
    Y_full_unscaled = pd.concat([Y_train, Y_test])
    Y_test_eval = []
    start_index = len(Y_train) - n_steps + 1
    for i in range(len(X_test_seq)):
        Y_test_eval.append(Y_full_unscaled.iloc[start_index + i : start_index + i + future_steps].values)
    Y_test_eval = np.array(Y_test_eval)

    return X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y, Y_full.index[-1]
ef build_multi_output_lstm(input_shape, future_steps):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(future_steps))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model
