import pandas as pd
import numpy as np
import os 
import warnings
import joblib # For saving scalers

# --- THIS IS THE FIX ---
# All TensorFlow/Keras imports must be at the top level
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# -----------------------

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Suppress harmless warnings and TensorFlow verbose output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
TARGET_PRICE = 'target'
TRAIN_SPLIT_RATIO = 0.8
N_STEPS = 60         # Time steps (past days) for LSTM input
N_FUTURE_DAYS = 15   # Constant for the forecast horizon

MODEL_FILE_NAME = 'lstm_15_day_model.h5'
SCALER_X_FILE_NAME = 'lstm_scaler_X.joblib'
SCALER_Y_FILE_NAME = 'lstm_scaler_Y.joblib'
# ---------------------

def load_and_split_data(filepath):
    """Loads the data and prepares the train/test split (chronological)."""
    print(f"\n--- Loading and Splitting Data from '{filepath}' ---")
    try:
        df = pd.read_csv(filepath, parse_dates=True, index_col=0).dropna(subset=[TARGET_PRICE])
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: '{filepath}'")
        print("Please run '3_feature_engineer.py' first.")
        return None, None, None, None
    
    # Use only required columns for features (X) and target (Y)
    # 'Reliance_Close' is dropped as it's the base for 'target'
    X = df.drop(columns=[TARGET_PRICE, 'Reliance_Close'], errors='ignore').select_dtypes(include=np.number)
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
    Y_full = pd.concat([Y_train, Y_test]) 
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_full_scaled = scaler_Y.fit_transform(Y_full.values.reshape(-1, 1))
    
    # Save scalers
    joblib.dump(scaler_X, SCALER_X_FILE_NAME)
    joblib.dump(scaler_Y, SCALER_Y_FILE_NAME)
    print(f"✅ Scalers saved to '{SCALER_X_FILE_NAME}' and '{SCALER_Y_FILE_NAME}'")
    
    def create_multi_sequences(X_scaled, Y_scaled_full):
        Xs, Ys = [], []
        # Adjust stop index to ensure Y target is available
        stop_index = len(X_scaled) - n_steps - future_steps + 1 
        
        for i in range(stop_index):
            Xs.append(X_scaled[i:i + n_steps])
            Ys.append(Y_scaled_full[i + n_steps: i + n_steps + future_steps].flatten())
        return np.array(Xs), np.array(Ys)

    X_seq_all, Y_seq_all = create_multi_sequences(X_all_scaled, Y_full_scaled)

    # Split the sequences back
    train_end_index = len(X_train) - n_steps - future_steps + 1
    # Ensure index is non-negative
    train_end_index = max(0, train_end_index) 
    
    X_train_seq = X_seq_all[:train_end_index]
    Y_train_seq = Y_seq_all[:train_end_index]
    X_test_seq = X_seq_all[train_end_index:]
    
    # Unscaled true target vectors for evaluation
    Y_full_unscaled = pd.concat([Y_train, Y_test])
    Y_test_eval = []
    
    # Calculate the correct start index for Y_test_eval
    start_index_unscaled_y = train_end_index + n_steps
    
    for i in range(len(X_test_seq)):
        start = start_index_unscaled_y + i
        end = start + future_steps
        if end <= len(Y_full_unscaled):
             Y_test_eval.append(Y_full_unscaled.iloc[start:end].values)
    
    Y_test_eval = np.array(Y_test_eval)

    # Ensure X_test and Y_test_eval have the same number of samples
    min_len = min(len(X_test_seq), len(Y_test_eval))
    X_test_seq = X_test_seq[:min_len]
    Y_test_eval = Y_test_eval[:min_len]

    print(f"✅ Sequence data created: X_train shape {X_train_seq.shape}, Y_train shape {Y_train_seq.shape}")
    print(f"✅ Sequence data created: X_test shape {X_test_seq.shape}, Y_test_eval shape {Y_test_eval.shape}")
    
    return X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y, Y_full.index[-1]

def train_and_evaluate_multi_output_lstm(X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y, last_known_date):
    """Trains, evaluates, and performs the final 15-day forecast."""
    
    if X_train_seq.shape[0] == 0:
        print("❌ ERROR: Not enough data to create training sequences. Try a smaller N_STEPS or get more data.")
        return None
        
    future_steps = Y_train_seq.shape[1]
    
    # 1. Build the Model
    # This is where 'Sequential' is used
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=future_steps)) # Output layer predicts all future steps
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Add an EarlyStopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print(f"\n--- Training Multi-Output LSTM Model (100 Epochs w/ Early Stopping) ---")
    model.fit(
        X_train_seq, Y_train_seq, 
        epochs=100, 
        batch_size=32, 
        verbose=1,
        validation_split=0.1, # Use 10% of training data for validation
        callbacks=[early_stop] # Add the callback
    )
    print("✅ Model training complete.")
    
    # Save the model
    model.save(MODEL_FILE_NAME)
    print(f"✅ Model saved to '{MODEL_FILE_NAME}'")

    # 2. Evaluation
    if X_test_seq.shape[0] > 0:
        predicted_scaled = model.predict(X_test_seq, verbose=0)
        predicted_price_vectors = scaler_Y.inverse_transform(predicted_scaled)
        
        actual_flat = Y_test_eval.flatten()
        predicted_flat = predicted_price_vectors.flatten()
        
        rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))
        r2 = r2_score(actual_flat, predicted_flat)
        
        print("--- Multi-Output LSTM Evaluation (Aggregate 15-Step Error) ---")
        print(f"  Aggregate RMSE: {rmse:.2f} (INR)")
        print(f"  Aggregate R-squared: {r2:.4f}")
    else:
        print("--- Skipping evaluation: Not enough test data to form sequences. ---")


    # 3. FINAL 15-Day Forecast
    # Use the last available sequence from the *entire* dataset
    X_final_forecast_input = np.vstack((X_train_seq, X_test_seq))[-1]
    X_final_forecast_input = X_final_forecast_input.reshape(1, X_final_forecast_input.shape[0], X_final_forecast_input.shape[1])
    
    final_scaled_forecast = model.predict(X_final_forecast_input, verbose=0).flatten()
    final_forecast_prices = scaler_Y.inverse_transform(final_scaled_forecast.reshape(-1, 1)).flatten()

    # Create date index
    forecast_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=N_FUTURE_DAYS)
    
    final_forecast_series = pd.Series(final_forecast_prices, index=forecast_dates, name="LSTM_MultiOutput")
    
    return final_forecast_series

def main():
    # Load and prepare data
    X_train, X_test, Y_train, Y_test = load_and_split_data(INPUT_FILE)
    
    if X_train is None:
        return
        
    X_train_multi, Y_train_multi, X_test_multi, Y_test_multi_eval, scaler_Y_multi, last_known_date = prepare_multi_output_lstm_data(
        X_train, X_test, Y_train, Y_test
    )
    
    # Train, evaluate, and get final forecast
    final_forecast_series = train_and_evaluate_multi_output_lstm(
        X_train_multi, Y_train_multi, X_test_multi, Y_test_multi_eval, scaler_Y_multi, last_known_date
    )

    if final_forecast_series is not None:
        print("\n" + "="*80)
        print(f"    ✨ FINAL {N_FUTURE_DAYS}-DAY FORECAST VIA MULTI-OUTPUT LSTM (Option B) ✨")
        print("="*80)
        print(final_forecast_series.to_string(float_format='%.2f'))
        print("\n✨ --- FORECAST COMPLETE --- ✨")

if __name__ == "__main__":
    main()

