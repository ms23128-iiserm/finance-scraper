import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os

# --- Configuration ---
MODEL_PATH = 'stock_prediction_model_target_r2.keras'
SCALER_PATH = 'scaler.joblib'
TICKER = 'RELIANCE.NS'
PREDICTION_START_DATE = '2025-10-20' # Inclusive
PREDICTION_END_DATE = '2025-10-30'   # Inclusive
TIMESTEPS = 60 # IMPORTANT: MUST match the timesteps used for training!

# --- Check if required files exist ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found at '{MODEL_PATH}'")
    exit()
if not os.path.exists(SCALER_PATH):
    print(f"❌ ERROR: Scaler file not found at '{SCALER_PATH}'")
    exit()

# --- 1. Load Model and Scaler ---
print(f"Loading model from '{MODEL_PATH}'...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Failed to load model: {e}")
    exit()

print(f"Loading scaler from '{SCALER_PATH}'...")
try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Failed to load scaler: {e}")
    exit()

# --- 2. Fetch Recent Data ---
# We need TIMESTEPS days of data *before* the PREDICTION_START_DATE
fetch_end_date = pd.to_datetime(PREDICTION_START_DATE)
# Fetch slightly more data to ensure TIMESTEPS business days are available
fetch_start_date = fetch_end_date - timedelta(days=TIMESTEPS + 45) # Increased buffer

print(f"\nFetching recent data for {TICKER} from {fetch_start_date.strftime('%Y-%m-%d')} to {fetch_end_date.strftime('%Y-%m-%d')}...")
try:
    # Use end date + 1 day for yf.download as it's exclusive for the end date sometimes
    stock_data = yf.download(TICKER, start=fetch_start_date, end=fetch_end_date + timedelta(days=1), progress=False)
    # Filter out data strictly before the prediction start date
    stock_data = stock_data[stock_data.index < fetch_end_date]

    if stock_data.empty:
        raise ValueError("No data fetched from yfinance. Check ticker or dates.")
    # Keep only the 'Close' price and ensure we have enough data
    recent_data = stock_data['Close'].values
    if len(recent_data) < TIMESTEPS:
        raise ValueError(f"Not enough historical data. Need {TIMESTEPS}, got {len(recent_data)}. Try fetching data from an earlier start date.")
    # Take the most recent TIMESTEPS days
    recent_data = recent_data[-TIMESTEPS:]
    print(f"✅ Fetched {len(recent_data)} data points for input.")
except Exception as e:
    print(f"❌ ERROR: Failed to fetch recent stock data: {e}")
    exit()

# --- 3. Scale Recent Data ---
try:
    # Scaler expects 2D array: (n_samples, n_features)
    scaled_data = scaler.transform(recent_data.reshape(-1, 1))
except Exception as e:
    print(f"❌ ERROR: Failed to scale recent data: {e}")
    exit()

# --- 4. Iterative Prediction ---
predictions_scaled = []
# Ensure current_batch has the correct starting shape (1, TIMESTEPS, 1)
current_batch = scaled_data.reshape(1, TIMESTEPS, 1)

# Determine the prediction dates (only considering weekdays for simplicity)
prediction_dates = pd.date_range(start=PREDICTION_START_DATE, end=PREDICTION_END_DATE, freq='B') # 'B' = Business Day frequency

print(f"\nPredicting prices for {len(prediction_dates)} business days from {PREDICTION_START_DATE} to {PREDICTION_END_DATE}...")

for i in range(len(prediction_dates)):
    try:
        # Get the prediction (scaled value) - shape (1, 1)
        next_prediction_scaled = model.predict(current_batch, verbose=0)

        # Store the scaled prediction (scalar value)
        predictions_scaled.append(next_prediction_scaled[0, 0])

        # Update the batch for the next prediction
        # Reshape the prediction to (1, 1, 1) to match dimensions for appending
        prediction_reshaped = next_prediction_scaled.reshape(1, 1, 1)

        # Append the new prediction, remove the oldest value from the batch
        current_batch = np.append(current_batch[:, 1:, :], prediction_reshaped, axis=1)

    except Exception as e:
        print(f"❌ ERROR: Failed during prediction loop on day {i+1}: {e}")
        # Fill remaining predictions with NaN or stop
        predictions_scaled.extend([np.nan] * (len(prediction_dates) - len(predictions_scaled)))
        break

print("✅ Prediction loop finished.")

# --- 5. Inverse Transform Predictions ---
if predictions_scaled:
    # Check for NaN values introduced by potential errors
    if np.isnan(predictions_scaled).any():
        print("⚠️ Warning: Some predictions failed and resulted in NaN values.")

    try:
        # Convert list to numpy array and reshape for scaler
        predictions_scaled_array = np.array(predictions_scaled).reshape(-1, 1)
        # Handle potential NaNs before inverse transforming if necessary
        # (Scaler might raise error on NaN) - For now, we let it try
        final_predictions = scaler.inverse_transform(predictions_scaled_array)
        print("✅ Inverse transform successful.")
    except ValueError as ve:
         # Handle case where inverse_transform fails due to NaNs
         print(f"❌ ERROR: Failed to inverse transform predictions, likely due to NaNs: {ve}")
         final_predictions = np.full_like(predictions_scaled_array, np.nan) # Fill with NaN on error
    except Exception as e:
        print(f"❌ ERROR: Failed to inverse transform predictions: {e}")
        final_predictions = np.full_like(predictions_scaled_array, np.nan) # Fill with NaN on error
else:
    print("⚠️ No predictions were generated.")
    final_predictions = np.array([])


# --- 6. Display Results ---
print("\n--- Predicted Stock Prices ---")
if len(final_predictions) == len(prediction_dates):
    # Create DataFrame, handling potential NaNs in predictions
    results_df = pd.DataFrame({
        'Date': prediction_dates.strftime('%Y-%m-%d'),
        f'Predicted {TICKER} Close': final_predictions.flatten() # flatten handles 2D array
    })
    # Optional: format NaN values for printing
    results_df[f'Predicted {TICKER} Close'] = results_df[f'Predicted {TICKER} Close'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "Error")
    print(results_df.to_string(index=False))
else:
    print("Could not generate predictions for all requested dates due to errors.")
    # Optionally print partial results if desired, handling potential length mismatch

print("----------------------------")

