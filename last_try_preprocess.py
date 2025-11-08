import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from pathlib import Path

# --- Configuration ---
CSV_FILE = 'reliance_data.csv' # Input CSV file from Step 1
LOOK_BACK_PERIOD = 60         # How many previous days the model looks at
TRAIN_SPLIT_RATIO = 0.8       # 80% for training, 20% for testing

# Output files
X_TRAIN_FILE = 'x_train.npy'
Y_TRAIN_FILE = 'y_train.npy'
X_TEST_FILE = 'x_test.npy'
Y_TEST_FILE = 'y_test.npy'
SCALER_FILE = 'scaler.joblib'

def preprocess_data(csv_path, look_back, split_ratio):
    """
    Loads, cleans, scales, and sequences the stock data.

    Args:
        csv_path (str): Path to the input CSV file.
        look_back (int): Number of previous time steps to use as input features.
        split_ratio (float): Proportion of data to use for training.

    Returns:
        None. Saves processed data to .npy files and scaler to .joblib file.
    """
    print("Starting Step 2: Data Preprocessing...")

    # --- 1. Load Data ---
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: Input file '{csv_path}' not found. Please run Step 1 first.")
        return

    try:
        # Load the CSV, ensure 'date' is parsed correctly and set as index
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        print(f"✅ Loaded data from '{csv_path}'. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ ERROR: Could not load or parse CSV file: {e}")
        return

    # --- 2. Isolate and Clean 'Close' Price ---
    if 'reliance_close' not in df.columns:
        print("❌ ERROR: 'reliance_close' column not found in CSV.")
        return

    # Create a DataFrame with only the 'Close' price
    close_prices = df[['reliance_close']].copy()

    # Force the column to be numeric, converting errors to NaN
    close_prices['reliance_close'] = pd.to_numeric(close_prices['reliance_close'], errors='coerce')

    # Remove any rows with NaN values that might have been introduced
    initial_rows = len(close_prices)
    close_prices.dropna(inplace=True)
    cleaned_rows = len(close_prices)
    if initial_rows > cleaned_rows:
        print(f"⚠️ Warning: Removed {initial_rows - cleaned_rows} rows with non-numeric 'reliance_close' values.")

    if close_prices.empty:
        print("❌ ERROR: No valid 'reliance_close' data remaining after cleaning.")
        return

    print(f"Total data points (after cleaning): {cleaned_rows}")
    dataset = close_prices.values # Convert to NumPy array for scaling

    # --- 3. Scale Data ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler only on the training portion to prevent data leakage
    training_data_len = int(np.ceil(len(dataset) * split_ratio))
    train_data_to_scale = dataset[0:int(training_data_len), :]
    scaler.fit(train_data_to_scale)

    # Scale the entire dataset using the fitted scaler
    scaled_data = scaler.transform(dataset)
    print("\nData scaled successfully (values are now between 0 and 1).")

    # --- 4. Create Training and Testing Datasets ---
    train_data = scaled_data[0:int(training_data_len), :]

    # Split data into x_train and y_train sets
    x_train = []
    y_train = []

    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i-look_back:i, 0]) # Take sequence of 'look_back' values
        y_train.append(train_data[i, 0])             # Take the next value as the target

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data for LSTM input [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the testing data set
    test_data = scaled_data[training_data_len - look_back:, :] # Start earlier to have look_back history for first test point

    # Split data into x_test and y_test sets
    x_test = []
    y_test = dataset[training_data_len:, :] # Actual values for y_test (will be scaled later if needed by model eval)

    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i-look_back:i, 0])

    # Convert x_test to numpy array
    x_test = np.array(x_test)

    # Reshape the data for LSTM input [samples, time steps, features]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Note: y_test remains the unscaled actual values from the original dataset.
    # We load the unscaled y_test in step 3b/4 and scale its corresponding portion for loss calculation/comparison if needed,
    # or directly compare inverse_transformed predictions to these unscaled values.
    # For saving y_test, we'll save the *scaled* version corresponding to x_test for consistency with x_train/y_train structure.
    y_test_scaled = scaled_data[training_data_len:, 0]


    print("\nSequence generation complete.")
    print(f"x_train shape: {x_train.shape[:2]}") # Print only samples and timesteps for brevity
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape[:2]}")
    print(f"y_test_scaled shape: {y_test_scaled.shape}") # Corresponds to x_test

    print("\nData reshaped for LSTM.")
    print(f"x_train shape (3D): {x_train.shape}")
    print(f"x_test shape (3D): {x_test.shape}")


    # --- 5. Save Processed Data and Scaler ---
    try:
        np.save(X_TRAIN_FILE, x_train)
        np.save(Y_TRAIN_FILE, y_train)
        np.save(X_TEST_FILE, x_test)
        np.save(Y_TEST_FILE, y_test_scaled) # Save the scaled version for consistency
        joblib.dump(scaler, SCALER_FILE)
        print(f"\n--- Preprocessing Complete! ---")
        print(f"Saved files: '{X_TRAIN_FILE}', '{Y_TRAIN_FILE}', '{X_TEST_FILE}', '{Y_TEST_FILE}', '{SCALER_FILE}'")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to save processed files: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    preprocess_data(CSV_FILE, LOOK_BACK_PERIOD, TRAIN_SPLIT_RATIO)

