import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings

# Suppress harmless warnings and TensorFlow verbose output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv'
OUTPUT_FILE = 'holdout_period_forecast.csv' # File to save the forecast
TARGET_COLUMN = 'target'         # Should be the price CHANGE (delta)
ACTUAL_PRICE_COL = 'Reliance_Close' # Original price column
TRAIN_SPLIT_RATIO = 0.8
N_STEPS = 60                       # Time steps for LSTM input
# ---------------------

def load_and_split_data(filepath, split_ratio):
    """Loads data and splits into training and prediction sets."""
    print(f"\n--- Loading Data and Splitting ({split_ratio*100:.0f}% Train / {(1-split_ratio)*100:.0f}% Predict) ---")
    if not os.path.exists(filepath):
        print(f"❌ ERROR: Input file not found: '{filepath}'")
        return None, None
    try:
        df_full = pd.read_csv(filepath, parse_dates=True, index_col=0)
        split_index = int(len(df_full) * split_ratio)

        train_df = df_full.iloc[:split_index]
        predict_df = df_full.iloc[split_index:] # Used only for dates and length

        if train_df.empty or predict_df.empty:
             print(f"❌ ERROR: Data split resulted in empty dataframe(s). Check split ratio.")
             return None, None

        last_train_date = train_df.index[-1]
        last_actual_price = train_df[ACTUAL_PRICE_COL].iloc[-1]
        n_future_steps = len(predict_df) # Number of steps to predict

        print(f"✅ Full data loaded ({len(df_full)} rows).")
        print(f"✅ Training data: {len(train_df)} rows (up to {last_train_date.strftime('%Y-%m-%d')}).")
        print(f"✅ Prediction period: {len(predict_df)} days.")
        print(f"   Last actual price before prediction: {last_actual_price:.2f}")

        return train_df, predict_df, last_train_date, last_actual_price, n_future_steps

    except Exception as e:
        print(f"❌ Error loading or splitting data: {e}")
        return None, None, None, None, None


def prepare_multi_output_sequences_and_scalers(train_df, n_steps, n_future):
    """Prepares MULTI-OUTPUT training sequences and fits scalers."""
    print(f"\n--- Preparing MULTI-OUTPUT LSTM Training Sequences (Steps={n_steps}, Future={n_future}) ---")

    feature_cols = train_df.drop(columns=[TARGET_COLUMN, ACTUAL_PRICE_COL], errors='ignore') \
                           .select_dtypes(include=np.number).columns
    X_train = train_df[feature_cols]
    Y_train = train_df[[TARGET_COLUMN]]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_train_scaled_flat = scaler_Y.fit_transform(Y_train) # Scale based on all target values

    X_train_seq, Y_train_seq = [], []
    # Adjust stop index for multi-output target sequence
    stop_index = len(X_train_scaled) - n_steps - n_future + 1
    if stop_index <= 0:
        print(f"❌ ERROR: Training data too short for multi-output sequences (need {n_steps + n_future} rows, have {len(X_train_scaled)}).")
        return None, None, None, None, None

    for i in range(stop_index):
        X_train_seq.append(X_train_scaled[i : i + n_steps])
        # Target is the sequence of scaled future deltas from the training data
        Y_train_seq.append(Y_train_scaled_flat[i + n_steps : i + n_steps + n_future].flatten())

    X_train_seq, Y_train_seq = np.array(X_train_seq), np.array(Y_train_seq)
    last_sequence_scaled = X_train_scaled[-n_steps:] # Last sequence from training data for prediction

    if X_train_seq.size == 0 or Y_train_seq.size == 0:
        print(f"❌ ERROR: Failed to create sequences. Check data length and parameters.")
        return None, None, None, None, None


    print(f"✅ Multi-output sequences created. X shape: {X_train_seq.shape}, Y shape: {Y_train_seq.shape}")
    print(f"✅ Last sequence for forecast extracted.")

    return X_train_seq, Y_train_seq, scaler_X, scaler_Y, last_sequence_scaled

def build_and_train_lstm_multi(X_train_seq, Y_train_seq, n_future):
    """Builds and trains the MULTI-OUTPUT LSTM model."""
    print("\n--- Building MULTI-OUTPUT LSTM Model ---")
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=n_future) # Predict N future deltas
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"\n--- Training MULTI-OUTPUT LSTM Model (on first {TRAIN_SPLIT_RATIO*100:.0f}% data) ---")
    model.fit(
        X_train_seq, Y_train_seq,
        epochs=200, batch_size=32,
        validation_split=0.1, callbacks=[early_stop],
        verbose=1 # Show training progress
    )
    print("\n✅ Multi-Output Model training complete.")
    return model

def generate_forecast_and_save(model, last_sequence_scaled, scaler_Y, n_future,
                             last_train_date, last_actual_price, predict_df_index, output_filepath):
    """Generates the multi-step forecast, reconstructs prices, and saves to CSV."""
    print(f"\n--- Generating {n_future}-Day Forecast for Holdout Period ---")

    # Reshape the last sequence for prediction
    forecast_batch = last_sequence_scaled.reshape((1, N_STEPS, last_sequence_scaled.shape[1]))

    # Predict the sequence of future deltas
    predicted_deltas_scaled = model.predict(forecast_batch, verbose=0)[0] # Get the (n_future,) array

    # Inverse scale the predicted deltas
    predicted_deltas = scaler_Y.inverse_transform(predicted_deltas_scaled.reshape(-1, 1)).flatten()

    # --- Reconstruct Absolute Prices ---
    print("\n--- Reconstructing Absolute Prices for Forecast ---")
    # Use the index from the actual predict_df for the forecast dates
    forecast_dates = predict_df_index

    # Ensure predicted_deltas length matches forecast_dates length
    if len(predicted_deltas) != len(forecast_dates):
         print(f"⚠️ WARNING: Length mismatch! Predicted {len(predicted_deltas)} deltas for {len(forecast_dates)} dates. Truncating forecast.")
         min_len = min(len(predicted_deltas), len(forecast_dates))
         predicted_deltas = predicted_deltas[:min_len]
         forecast_dates = forecast_dates[:min_len]


    forecasted_prices = []
    current_price = last_actual_price
    for delta in predicted_deltas:
        current_price += delta
        forecasted_prices.append(current_price)

    forecast_df = pd.DataFrame({
        'Predicted_Change': predicted_deltas,
        'Forecasted_Price': forecasted_prices
    }, index=forecast_dates)
    forecast_df.index.name = 'Date'

    # --- Save to CSV ---
    try:
        forecast_df.to_csv(output_filepath)
        print(f"\n✅ Forecast saved successfully to '{output_filepath}'")
        print("\n--- Forecast Summary ---")
        print(forecast_df.to_string(float_format='%.2f'))
        print("------------------------")
    except Exception as e:
        print(f"❌ Error saving forecast to CSV: {e}")


def main():
    # Load and split data
    train_df, predict_df, last_train_date, last_actual_price, n_future_steps = load_and_split_data(
        INPUT_FILE, TRAIN_SPLIT_RATIO
    )
    if train_df is None: return

    # Prepare sequences for multi-output training
    X_train_seq, Y_train_seq, scaler_X, scaler_Y, last_sequence_scaled = \
        prepare_multi_output_sequences_and_scalers(train_df, N_STEPS, n_future_steps)
    if X_train_seq is None: return

    # Build and train the multi-output model
    model = build_and_train_lstm_multi(X_train_seq, Y_train_seq, n_future_steps)

    # Generate the forecast for the holdout period and save it
    generate_forecast_and_save(model, last_sequence_scaled, scaler_Y, n_future_steps,
                               last_train_date, last_actual_price, predict_df.index, OUTPUT_FILE)

    print("\n✨ --- HOLDOUT FORECAST GENERATION COMPLETE --- ✨")

# --- Run Main ---
if __name__ == "__main__":
    main()

