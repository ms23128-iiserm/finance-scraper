import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import time
import os

# --- Configuration ---
X_TRAIN_FILE = 'x_train.npy'
Y_TRAIN_FILE = 'y_train.npy'
X_TEST_FILE = 'x_test.npy'
Y_TEST_FILE = 'y_test.npy' # Note: This file should contain SCALED values from step_2
SCALER_FILE = 'scaler.joblib'
MODEL_SAVE_PATH = 'stock_prediction_model_target_r2.keras'
LOSS_PLOT_FILE = 'training_loss_plot_target_r2.png'
PREDICTION_PLOT_FILE = 'test_predictions_plot_target_r2.png'

# Model Hyperparameters (Adjust these to tune R-squared)
LSTM_UNITS = 30     # Keep reduced number of units
DROPOUT_RATE = 0.2
EPOCHS = 50    # Increased training epochs slightly (from 10)
BATCH_SIZE = 32

def build_lstm_model(input_shape):
    """Builds the LSTM model architecture."""
    model = Sequential(name="LSTM_Stock_Predictor_Target_R2")
    model.add(Input(shape=input_shape, name="Input_Layer"))
    # Reintroduce the second LSTM layer but keep units low
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True, name="LSTM_1"))
    model.add(Dropout(DROPOUT_RATE, name="Dropout_1"))
    model.add(LSTM(units=LSTM_UNITS, return_sequences=False, name="LSTM_2")) # Second LSTM layer added back
    model.add(Dropout(DROPOUT_RATE, name="Dropout_2"))
    model.add(Dense(units=LSTM_UNITS // 2, activation='relu', name="Dense_1"))
    model.add(Dense(units=1, name="Output_Layer"))
    return model

def plot_training_history(history, filename):
    """Plots training & validation loss."""
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history: # Check if validation loss exists
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Training history plot saved to '{filename}'")
    plt.close() # Close the plot to avoid displaying it if running in non-interactive mode

def plot_predictions(actual, predicted, filename, r2, mae, rmse):
    """Plots actual vs predicted prices for the test set."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    plt.plot(actual, label='Actual Price', color='blue', linewidth=1.5)
    plt.plot(predicted, label='Predicted Price', color='red', linestyle='--', linewidth=1.5)
    plt.title(f'LSTM Model Performance - Test Set\n$R^2$: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}')
    plt.xlabel('Time (Test Set Days)')
    plt.ylabel('Reliance Price (INR)')
    plt.legend()
    plt.grid(True) # Ensure grid is visible
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Test set prediction plot saved to '{filename}'")
    plt.close() # Close the plot


def train_evaluate_model():
    """Loads data, builds, trains, and evaluates the LSTM model."""
    print("--- Starting Model Training & Evaluation ---")

    # --- 1. Load Training Data ---
    print(f"Loading training data from {X_TRAIN_FILE} and {Y_TRAIN_FILE}...")
    if not (os.path.exists(X_TRAIN_FILE) and os.path.exists(Y_TRAIN_FILE)):
        print(f"❌ ERROR: Training files '{X_TRAIN_FILE}' or '{Y_TRAIN_FILE}' not found. Please run Step 2 first.")
        return
    try:
        x_train = np.load(X_TRAIN_FILE)
        y_train = np.load(Y_TRAIN_FILE)
    except Exception as e:
        print(f"❌ ERROR: Failed to load training data files: {e}")
        return
    print("Training data loaded successfully.")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    if len(x_train.shape) != 3 or x_train.shape[2] != 1:
        print("❌ ERROR: x_train should be 3D (samples, timesteps, features). Did Step 2 run correctly?")
        return

    # --- 2. Build Model ---
    print("\nBuilding LSTM model architecture...")
    input_shape = (x_train.shape[1], x_train.shape[2]) # (timesteps, features)
    model = build_lstm_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model built and compiled successfully.")
    model.summary()

    # --- 3. Train Model ---
    print("\nStarting model training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    start_time = time.time()
    try:
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1, # Use 10% of training data for validation
            verbose=1 # Show progress bar
        )
    except Exception as e:
        print(f"❌ ERROR: An error occurred during model training: {e}")
        return
    end_time = time.time()
    training_time = end_time - start_time
    print("\n--- Training Complete! ---")
    print(f"Total training time: {training_time:.2f} seconds")

    # --- 4. Save Model ---
    try:
        model.save(MODEL_SAVE_PATH)
        print(f"✅ Model saved successfully to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save the model: {e}")
        # Continue to evaluation even if saving fails

    # --- 5. Plot Training History ---
    try:
        plot_training_history(history, LOSS_PLOT_FILE)
    except Exception as e:
        print(f"⚠️ Warning: Could not plot training history: {e}")

    # --- 6. Load Test Data and Scaler ---
    print("\nLoading test data and scaler for evaluation...")
    if not (os.path.exists(X_TEST_FILE) and os.path.exists(Y_TEST_FILE) and os.path.exists(SCALER_FILE)):
        print(f"❌ ERROR: Test files ('{X_TEST_FILE}', '{Y_TEST_FILE}') or scaler ('{SCALER_FILE}') not found. Please ensure Step 2 completed successfully.")
        return
    try:
        x_test = np.load(X_TEST_FILE)
        # IMPORTANT: Load the SCALED y_test values saved by step_2_preprocess_data.py
        y_test_scaled = np.load(Y_TEST_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception as e:
        print(f"❌ ERROR: Failed to load test data files or scaler: {e}")
        return
    print("Test data and scaler loaded successfully.")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test_scaled shape: {y_test_scaled.shape}") # Corrected variable name

    # --- 7. Make Predictions on Test Set ---
    print("\nMaking predictions on the test set...")
    try:
        predictions_scaled = model.predict(x_test)
    except Exception as e:
        print(f"❌ ERROR: Failed to make predictions: {e}")
        return

    # --- 8. Inverse Transform Predictions and Actual Values ---
    print("Inverting scaling to get actual price values...")
    try:
        # Reshape y_test_scaled for the scaler (needs 2D)
        y_test_scaled_reshaped = y_test_scaled.reshape(-1, 1)
        # Inverse transform predictions and the scaled actual values
        predictions = scaler.inverse_transform(predictions_scaled)
        actual = scaler.inverse_transform(y_test_scaled_reshaped) # Use the scaled y_test
    except Exception as e:
        print(f"❌ ERROR: Failed during inverse scaling: {e}")
        return

    # --- 9. Calculate Evaluation Metrics ---
    print("\nCalculating evaluation metrics on the test set...")
    try:
        r2 = r2_score(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        print(f"\n--- Test Set Evaluation Results ---")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print("-----------------------------------")
    except Exception as e:
        print(f"❌ ERROR: Failed to calculate metrics: {e}")
        return

    # --- 10. Plot Test Set Predictions ---
    try:
        plot_predictions(actual, predictions, PREDICTION_PLOT_FILE, r2, mae, rmse)
    except Exception as e:
        print(f"⚠️ Warning: Could not plot predictions: {e}")


    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    train_evaluate_model()

