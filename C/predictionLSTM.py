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
    """Loads the data and prepares the train/test split (chronological)."""
    print(f"\n--- Loading and Splitting Data from '{filepath}' ---")
    df = pd.read_csv(filepath, parse_dates=True, index_col=0).dropna()
    
    # Use only required columns for features (X) and target (Y)
    X = df.drop(columns=[TARGET_PRICE, 'direction'], errors='ignore').select_dtypes(include=np.number)
    Y = df[TARGET_PRICE]
    
    train_size = int(len(df) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]
    
    print(f"✅ Data Split: Train size = {len(X_train)}, Test size = {len(X_test)}")
    return X_train, X_test, Y_train, Y_test

def prepare_multi_output_lstm_data(X_train, X_test, Y_train, Y_test, n_steps=N_STEPS, future_steps=N_FUTURE_DAYS):
    """Prepares data for the multi-output LSTM model."""
    print(f"\n--- Preparing Data for Multi-Output LSTM (Future Steps={future_steps}) ---")
    
    # Scale X
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_all_scaled = np.vstack((X_train_scaled, X_test_scaled))

    # Scale Y (Target is the combination of train and test)
    # FIX 1: Use pd.concat for Y_full scaling
    Y_full = pd.concat([Y_train, Y_test]) 
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_full_scaled = scaler_Y.fit_transform(Y_full.values.reshape(-1, 1))
    
    def create_multi_sequences(X_scaled, Y_scaled_full):
        Xs, Ys = [], []
        stop_index = len(X_scaled) - n_steps - future_steps + 1 
        
        for i in range(stop_index):
            Xs.append(X_scaled[i:i + n_steps])
            Ys.append(Y_scaled_full[i + n_steps: i + n_steps + future_steps].flatten())
        return np.array(Xs), np.array(Ys)

    X_seq_all, Y_seq_all = create_multi_sequences(X_all_scaled, Y_full_scaled)

    # Split the sequences back
    train_end_index = len(X_train) - n_steps - future_steps + 1
    X_train_seq = X_seq_all[:train_end_index]
    Y_train_seq = Y_seq_all[:train_end_index]
    X_test_seq = X_seq_all[train_end_index:]
    
    # Unscaled true target vectors for evaluation
    # FIX 2: Use pd.concat() for Y_full_unscaled
    Y_full_unscaled = pd.concat([Y_train, Y_test])
    Y_test_eval = []
    start_index = len(Y_train) - n_steps + 1
    for i in range(len(X_test_seq)):
        Y_test_eval.append(Y_full_unscaled.iloc[start_index + i : start_index + i + future_steps].values)
    Y_test_eval = np.array(Y_test_eval)

    return X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y, Y_full.index[-1]

def train_and_evaluate_multi_output_lstm(X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y, last_known_date):
    """Trains, evaluates, and performs the final 15-day forecast."""
    future_steps = Y_train_seq.shape[1]
    
    # 1. Build the Model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=future_steps)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    print(f"\n--- Training Multi-Output LSTM Model (75 Epochs) ---")
    model.fit(X_train_seq, Y_train_seq, epochs=75, batch_size=32, verbose=0)

    # 2. Evaluation
    predicted_scaled = model.predict(X_test_seq, verbose=0)
    predicted_price_vectors = scaler_Y.inverse_transform(predicted_scaled)
    actual_flat = Y_test_eval.flatten()
    predicted_flat = predicted_price_vectors.flatten()
    rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))
    r2 = r2_score(actual_flat, predicted_flat)
    
    print("--- Multi-Output LSTM Evaluation (Aggregate 15-Step Error) ---")
    print(f"Aggregate RMSE: {rmse:.2f}")
    print(f"Aggregate R-squared: {r2:.4f}")

    # 3. FINAL 15-Day Forecast
    X_final_forecast = X_test_seq[-1].reshape(1, X_test_seq.shape[1], X_test_seq.shape[2])
    final_scaled_forecast = model.predict(X_final_forecast, verbose=0).flatten()
    final_forecast_prices = scaler_Y.inverse_transform(final_scaled_forecast.reshape(-1, 1)).flatten()

    # Create date index
    # FIX 3: Replaced 'closed' with 'inclusive'
    forecast_dates = pd.date_range(start=last_known_date, periods=N_FUTURE_DAYS + 1, inclusive='neither')
    
    final_forecast_series = pd.Series(final_forecast_prices, index=forecast_dates, name="LSTM_MultiOutput")
    
    return final_forecast_series

def main():
    # Load and prepare data
    X_train, X_test, Y_train, Y_test = load_and_split_data(INPUT_FILE)
    
    X_train_multi, Y_train_multi, X_test_multi, Y_test_multi_eval, scaler_Y_multi, last_known_date = prepare_multi_output_lstm_data(
        X_train, X_test, Y_train, Y_test
    )
    
    # Train, evaluate, and get final forecast
    final_forecast_series = train_and_evaluate_multi_output_lstm(
        X_train_multi, Y_train_multi, X_test_multi, Y_test_multi_eval, scaler_Y_multi, last_known_date
    )

    print("\n" + "="*80)
    print(f"    ✨ FINAL {N_FUTURE_DAYS}-DAY FORECAST VIA MULTI-OUTPUT LSTM (Option B) ✨")
    print("="*80)
    print(final_forecast_series.to_string(float_format='%.2f'))
    print("\n✨ --- FORECAST COMPLETE --- ✨")

if __name__ == "__main__":
    main()