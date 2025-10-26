import pandas as pd
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

# --- Keras/TensorFlow Imports ---
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'features_engineered_data.csv'
MODEL_FILE = 'lstm_15_day_model.h5'
SCALER_X_FILE = 'lstm_scaler_X.joblib'
SCALER_Y_FILE = 'lstm_scaler_Y.joblib'

# Model settings (must match training)
N_STEPS = 60
TARGET_PRICE = 'target'
ACTUAL_PRICE_COL = 'Reliance_Close'

# Report settings
HISTORICAL_START_DATE = '2025-10-05'
# --- THIS IS THE FIX ---
# Set the end date to the last *available* date in your dataset,
# which your log showed was 2025-10-24.
HISTORICAL_END_DATE = '2025-10-24' 
FORECAST_DAYS = 7
# ---------------------

def load_all():
    """Loads the dataset, trained model, and scalers."""
    print("--- Loading Model, Scalers, and Full Data ---")
    try:
        model = load_model(MODEL_FILE)
        scaler_X = joblib.load(SCALER_X_FILE)
        scaler_Y = joblib.load(SCALER_Y_FILE)
        df = pd.read_csv(DATA_FILE, parse_dates=True, index_col=0)
        
        # Define feature columns (all numeric columns except targets/leaks)
        feature_cols = df.drop(columns=[TARGET_PRICE, ACTUAL_PRICE_COL], errors='ignore').select_dtypes(include=np.number).columns
        
        # Scale all features
        df_scaled_features = scaler_X.transform(df[feature_cols])
        
        print("✅ All files loaded successfully.")
        return model, scaler_Y, df, df_scaled_features, feature_cols

    except Exception as e:
        print(f"❌ CRITICAL ERROR loading files: {e}")
        print("Please ensure all model and scaler files from script 4b exist.")
        return None, None, None, None, None

def generate_historical_report(model, scaler_Y, df, df_scaled_features):
    """
    Task 1: Generates a table and graph for the specified historical date range.
    """
    print("\n" + "="*80)
    print(f"    TASK 1: HISTORICAL REPORT ({HISTORICAL_START_DATE} to {HISTORICAL_END_DATE})")
    print("="*80)
    
    try:
        # Get the integer index locations for our date range
        start_iloc = df.index.get_loc(HISTORICAL_START_DATE)
        end_iloc = df.index.get_loc(HISTORICAL_END_DATE)
    except KeyError as e:
        print(f"❌ ERROR: Date not found in dataset: {e}. Cannot generate report.")
        print("  Please check your HISTORICAL_START_DATE and HISTORICAL_END_DATE.")
        # If dates are bad, we can still skip to Task 2
        return

    if (start_iloc - N_STEPS) < 0:
        print(f"❌ ERROR: Start date {HISTORICAL_START_DATE} is too early.")
        print(f"  It does not have {N_STEPS} days of history before it.")
        return

    # --- Prepare data for the specific period ---
    X_sequences_for_period = []
    base_prices_for_period = []
    actual_prices_for_period = []
    dates_for_period = []

    print(f"--- Preparing sequences for {HISTORICAL_START_DATE} to {HISTORICAL_END_DATE} ---")
    
    # Loop from the start date to the end date
    for i in range(start_iloc, end_iloc + 1):
        # Get the sequence of N_STEPS days *before* the current day
        sequence = df_scaled_features[i - N_STEPS : i]
        X_sequences_for_period.append(sequence)
        
        # Get the actual price from the day *before* (T-1)
        base_prices_for_period.append(df[ACTUAL_PRICE_COL].iloc[i-1])
        
        # Get the actual price for the current day (T)
        actual_prices_for_period.append(df[ACTUAL_PRICE_COL].iloc[i])
        
        # Get the date
        dates_for_period.append(df.index[i])

    # Convert to NumPy array for the model
    X_sequences_for_period = np.array(X_sequences_for_period)
    
    # --- Make Predictions ---
    # 1. Predict the *deltas* (price changes)
    predicted_deltas_scaled = model.predict(X_sequences_for_period, verbose=0)
    
    # 2. We only care about the *next day's* prediction
    predicted_next_day_deltas_scaled = predicted_deltas_scaled[:, 0]
    
    # 3. Inverse-scale the deltas
    predicted_next_day_deltas = scaler_Y.inverse_transform(
        predicted_next_day_deltas_scaled.reshape(-1, 1)
    ).flatten()
    
    # 4. Reconstruct the absolute price
    predicted_prices = np.array(base_prices_for_period) + predicted_next_day_deltas

    # --- Generate Table ---
    report_df = pd.DataFrame({
        'Actual_Price': actual_prices_for_period,
        'Predicted_Price': predicted_prices
    }, index=pd.to_datetime(dates_for_period))
    
    report_df['Difference (INR)'] = report_df['Actual_Price'] - report_df['Predicted_Price']
    
    print("\n--- Historical Comparison Table ---")
    print(report_df.to_string(float_format='%.2f'))
    
    # --- Generate Graph ---
    plt.figure(figsize=(15, 7))
    plt.plot(report_df.index, report_df['Actual_Price'], 'b-', label='Actual Price')
    plt.plot(report_df.index, report_df['Predicted_Price'], 'r--', label='Predicted Price')
    plt.title(f'Historical Comparison ({HISTORICAL_START_DATE} to {HISTORICAL_END_DATE})')
    plt.xlabel('Date')
    plt.ylabel('Reliance Price (INR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_file = 'historical_report_graph.png'
    plt.savefig(output_file)
    print(f"\n✅ Graph saved to '{output_file}'")

def generate_future_forecast(model, scaler_Y, df, df_scaled_features):
    """
    Task 2: Generates a new forecast for the next 7 days.
    """
    print("\n" + "="*80)
    print(f"    TASK 2: NEW {FORECAST_DAYS}-DAY FUTURE FORECAST")
    print("="*80)

    # 1. Get the last N_STEPS days of data from the *end* of the dataset
    last_sequence_scaled = df_scaled_features[-N_STEPS:]
    
    # 2. Reshape for the model
    X_final_input = last_sequence_scaled.reshape(1, N_STEPS, last_sequence_scaled.shape[1])
    
    print(f"--- Predicting {FORECAST_DAYS} days based on data up to {df.index[-1].strftime('%Y-%m-%d')} ---")
    
    # 3. Predict the 15-day deltas
    predicted_15_deltas_scaled = model.predict(X_final_input, verbose=0)
    
    # 4. Inverse-scale the deltas
    predicted_15_deltas = scaler_Y.inverse_transform(predicted_15_deltas_scaled).flatten()
    
    # 5. Select only the number of days we want
    forecast_deltas = predicted_15_deltas[:FORECAST_DAYS]
    
    # 6. Get the last known actual price
    last_actual_price = df[ACTUAL_PRICE_COL].iloc[-1]
    last_actual_date = df.index[-1]
    
    print(f"Last Actual Price on {last_actual_date.strftime('%Y-%m-%d')}: {last_actual_price:.2f} INR")

    # 7. Apply deltas cumulatively to get the forecast
    forecast_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)
    
    forecasted_prices = []
    current_price = last_actual_price
    
    for change in forecast_deltas:
        current_price += change
        forecasted_prices.append(current_price)
        
    # --- Generate Table ---
    forecast_df = pd.DataFrame({
        'Forecasted_Price': forecasted_prices,
        'Predicted_Change': forecast_deltas
    }, index=forecast_dates)
    forecast_df.index.name = "Date"
    
    print("\n--- 7-Day Price Forecast ---")
    print(forecast_df.to_string(float_format='%.2f'))

def main():
    model, scaler_Y, df, df_scaled_features, feature_cols = load_all()
    
    if model is None:
        return
        
    # Task 1: Run the historical report
    generate_historical_report(model, scaler_Y, df, df_scaled_features)
    
    # Task 2: Generate the new forecast
    generate_future_forecast(model, scaler_Y, df, df_scaled_features)
    
    print("\n✨ --- CUSTOM ANALYSIS COMPLETE --- ✨")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

# --- Keras/TensorFlow Imports ---
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Scikit-learn Metrics ---
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'features_engineered_data.csv'
MODEL_FILE = 'lstm_15_day_model.h5'
SCALER_X_FILE = 'lstm_scaler_X.joblib'
SCALER_Y_FILE = 'lstm_scaler_Y.joblib'

# These must match the settings used for training
N_STEPS = 60
TRAIN_SPLIT_RATIO = 0.8
TARGET_PRICE = 'target' # This is the 'delta' or 'price change'
ACTUAL_PRICE_COL = 'Reliance_Close' # The original price column
# ---------------------

def load_data_and_model():
    """Loads the dataset, trained model, and scalers."""
    print("--- Loading Model, Scalers, and Data ---")
    
    try:
        model = load_model(MODEL_FILE)
        scaler_X = joblib.load(SCALER_X_FILE)
        scaler_Y = joblib.load(SCALER_Y_FILE)
        df = pd.read_csv(DATA_FILE, parse_dates=True, index_col=0)
        
        # Ensure 'target' exists (it's our delta)
        if TARGET_PRICE not in df.columns:
            print(f"❌ ERROR: Target column '{TARGET_PRICE}' not in {DATA_FILE}.")
            return None, None, None, None
            
        # Ensure 'Reliance_Close' exists (for actuals and base price)
        if ACTUAL_PRICE_COL not in df.columns:
            print(f"❌ ERROR: Actual price column '{ACTUAL_PRICE_COL}' not in {DATA_FILE}.")
            return None, None, None, None

        print("✅ All files loaded successfully.")
        return model, scaler_X, scaler_Y, df

    except Exception as e:
        print(f"❌ CRITICAL ERROR loading files: {e}")
        print("Please ensure all model and scaler files from script 4b exist.")
        return None, None, None, None

def prepare_test_sequences(df, scaler_X, n_steps=N_STEPS):
    """Creates the test set sequences for the model to predict on."""
    print("--- Preparing Test Data Sequences ---")
    
    # Split data chronologically
    train_size = int(len(df) * TRAIN_SPLIT_RATIO)
    test_df = df.iloc[train_size:]

    # Define feature columns (all numeric columns except targets/leaks)
    feature_cols = df.drop(columns=[TARGET_PRICE, ACTUAL_PRICE_COL], errors='ignore').select_dtypes(include=np.number).columns
    
    # Scale the test features
    X_test_scaled = scaler_X.transform(test_df[feature_cols])
    
    # Create sequences
    X_test_seq = []
    for i in range(n_steps, len(X_test_scaled)):
        X_test_seq.append(X_test_scaled[i-n_steps:i])
    
    if not X_test_seq:
        print(f"❌ ERROR: Test set is too small to create sequences with N_STEPS={n_steps}.")
        return None, None, None
        
    X_test_seq = np.array(X_test_seq)
    print(f"✅ Test sequences created. Shape: {X_test_seq.shape}")
    
    # --- Align Actual Prices and Base Prices ---
    # We need to drop the first 'n_steps' rows from the test set
    # to align with the sequences we just made.
    
    # 'actual_prices' are the prices we are trying to predict (T)
    actual_prices = test_df[ACTUAL_PRICE_COL].iloc[n_steps:]
    
    # 'base_prices' are the prices from the day *before* (T-1)
    # We will add our predicted *change* to this base price.
    base_prices = test_df[ACTUAL_PRICE_COL].iloc[n_steps-1:-1]

    return X_test_seq, actual_prices, base_prices

def make_and_plot_predictions(model, scaler_Y, X_test_seq, actual_prices, base_prices):
    """
    Makes predictions, reconstructs the absolute price,
    calculates metrics, and plots the final comparison.
    """
    print("--- Making Predictions on Test Set ---")
    
    # 1. Predict the *deltas* (price changes)
    # This gives [samples, 15_days]
    predicted_deltas_scaled = model.predict(X_test_seq, verbose=0)
    
    # 2. We only care about the *next day's* prediction, not all 15
    # Select the first column: [samples, 0]
    predicted_next_day_deltas_scaled = predicted_deltas_scaled[:, 0]
    
    # 3. Inverse-scale the deltas to get them in Rupees
    predicted_next_day_deltas = scaler_Y.inverse_transform(
        predicted_next_day_deltas_scaled.reshape(-1, 1)
    ).flatten()
    
    # 4. Reconstruct the absolute price
    #    Predicted Price (T) = Actual Price (T-1) + Predicted Delta (T)
    predicted_prices = base_prices.values + predicted_next_day_deltas
    
    print("✅ Price reconstruction complete.")
    
    # --- 5. Calculate Metrics ---
    
    # Use the 'actual_prices' and the reconstructed 'predicted_prices'
    r2 = r2_score(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print("\n" + "="*50)
    print("   HONEST MODEL PERFORMANCE METRICS (on Test Set)")
    print("="*50)
    print(f"R-squared (R²): {r2:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} rupees")
    print(f"MAE (Mean Absolute Error): {mae:.2f} rupees")
    print("="*50)
    print(f"\nThis model's average prediction error is {mae:.2f} rupees.")
    print("You can copy these values into Slide 9 of your presentation.")
    
    # --- Create Plot ---
    
    # Create a results DataFrame
    results_df = pd.DataFrame({
        'Actual_Price': actual_prices.values,
        'Predicted_Price': predicted_prices
    }, index=actual_prices.index)
    
    # Select only the last 30 trading days for the plot
    plot_data = results_df.iloc[-30:]
    
    print(f"\n--- Plotting Last {len(plot_data)} Days of Test Set ---")
    
    plt.figure(figsize=(15, 8))
    plt.plot(plot_data.index, plot_data['Actual_Price'], 'b-', label='Actual Price')
    plt.plot(plot_data.index, plot_data['Predicted_Price'], 'r--', label='Predicted Price (Reconstructed)')
    
    # Add metrics to the title
    plt.title(f'LSTM Model Performance (Predicting Price Changes)\n'
              f'Test Set MAE: {mae:.2f} | R²: {r2:.4f} | RMSE: {rmse:.2f}',
              fontsize=14)
              
    plt.xlabel('Date')
    plt.ylabel('Reliance Price (INR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_file = 'test_set_comparison_last_month.png'
    plt.savefig(output_file)
    print(f"✅ Graph saved to '{output_file}'")

def main():
    model, scaler_X, scaler_Y, df = load_data_and_model()
    
    if model is None:
        return
        
    X_test_seq, actual_prices, base_prices = prepare_test_sequences(df, scaler_X)
    
    if X_test_seq is None:
        return
        
    make_and_plot_predictions(model, scaler_Y, X_test_seq, actual_prices, base_prices)

if __name__ == "__main__":
    main()


