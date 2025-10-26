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
    
    # --- Create Plot ---
    
    # Create a results DataFrame
    results_df = pd.DataFrame({
        'Actual_Price': actual_prices.values,
        'Predicted_Price': predicted_prices
    }, index=actual_prices.index)
    
    # *** THIS IS THE CHANGE ***
    # We are now plotting the *entire* results_df, not just the last 30 days.
    plot_data = results_df
    
    print(f"\n--- Plotting Full Test Set ({len(plot_data)} Days) ---")
    
    plt.figure(figsize=(15, 8))
    plt.plot(plot_data.index, plot_data['Actual_Price'], 'b-', label='Actual Price', linewidth=1.0)
    plt.plot(plot_data.index, plot_data['Predicted_Price'], 'r--', label='Predicted Price (Reconstructed)', linewidth=1.2)
    
    # Add metrics to the title
    plt.title(f'LSTM Model Performance - Full Test Set ({len(plot_data)} days)\n'
              f'Test Set MAE: {mae:.2f} | R²: {r2:.4f} | RMSE: {rmse:.2f}',
              fontsize=14)
              
    plt.xlabel('Date')
    plt.ylabel('Reliance Price (INR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # *** THIS IS THE CHANGE ***
    output_file = 'test_set_comparison_full_year.png'
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

