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

# Suppress harmless warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
PRICE_COLUMN = 'Reliance_Close' 
TARGET_PRICE = 'target'
ADVANCED_RESULTS_FILE = 'advanced_model_results.csv' 
TRAIN_SPLIT_RATIO = 0.8
N_STEPS = 60 # Time steps (past days) for LSTM input
N_FUTURE_DAYS = 15 # Constant for the recursive forecast horizon
# ---------------------

def load_and_split_data(filepath):
    """Loads the data and prepares the train/test split (chronological)."""
    print(f"\n--- Loading and Splitting Data from '{filepath}' ---")
    df = pd.read_csv(filepath, parse_dates=True, index_col=0).dropna()

    # Features (X): All columns EXCEPT the target price and direction
    X = df.drop(columns=[TARGET_PRICE, 'direction'], errors='ignore')
    # Target (Y): Only the next day's price
    Y = df[TARGET_PRICE] 

    # Drop non-numeric/non-predictive columns that may have slipped through
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
    """Scales and reshapes data into the 3D format required by LSTM."""
    print(f"\n--- Preparing Data for LSTM (N_STEPS={n_steps}) ---")
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_train_scaled = scaler_Y.fit_transform(Y_train.values.reshape(-1, 1))
    
    def create_sequences(X, Y): 
        Xs = []
        for i in range(len(X) - n_steps):
            Xs.append(X[i:i + n_steps])
        Y_eval = Y.iloc[n_steps:]
        return np.array(Xs), Y_eval

    X_train_seq, _ = create_sequences(X_train_scaled, Y_train)
    Y_train_seq = Y_train_scaled[N_STEPS:] 
    
    X_test_seq, Y_test_eval = create_sequences(X_test_scaled, Y_test)
    
    print(f"✅ LSTM Data Ready: X_train_seq shape: {X_train_seq.shape}")
    return X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y

def train_and_evaluate_lstm(X_train_seq, Y_train_seq, X_test_seq, Y_test_eval, scaler_Y):
    """Builds, trains, and evaluates the LSTM model."""
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    print("\n--- Training LSTM Model (50 Epochs) ---")
    model.fit(X_train_seq, Y_train_seq, epochs=50, batch_size=32, verbose=0)
    
    predicted_scaled = model.predict(X_test_seq, verbose=0)
    predicted_price = scaler_Y.inverse_transform(predicted_scaled).flatten()
    
    rmse = np.sqrt(mean_squared_error(Y_test_eval, predicted_price))
    r2 = r2_score(Y_test_eval, predicted_price)
    
    print("--- LSTM Model Evaluation ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.4f}")
    
    return Y_test_eval, predicted_price, model

# ==============================================================================
# 2. ARIMA BASELINE MODEL
# ==============================================================================

def train_and_evaluate_arima(Y_train, Y_test):
    """Trains and evaluates the ARIMA model."""
    print("\n--- Training ARIMA Baseline (p=5, d=1, q=0) ---")
    
    model = ARIMA(Y_train, order=(5, 1, 0))
    model_fit = model.fit()
    
    start_index = len(Y_train)
    end_index = len(Y_train) + len(Y_test) - 1
    
    forecast = model_fit.predict(start=start_index, end=end_index, dynamic=False)
    
    forecast.index = Y_test.index
    
    rmse = np.sqrt(mean_squared_error(Y_test, forecast))
    r2 = r2_score(Y_test, forecast)
    
    print("--- ARIMA Model Evaluation ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.4f}")
    
    return Y_test, forecast.values

# ==============================================================================
# 3. XGBOOST OPTIMIZATION & EVALUATION
# ==============================================================================

def optimize_xgboost(X_train, Y_train, feature_names):
    """Performs Hyperparameter Tuning for the XGBoost Regressor using GridSearchCV."""
    print("\n--- Optimizing XGBoost Regressor (GridSearchCV) ---")
    
    tscv = TimeSeriesSplit(n_splits=3) 
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }

    reg_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, use_label_encoder=False)
    
    grid_search = GridSearchCV(
        estimator=reg_model, 
        param_grid=param_grid, 
        scoring='neg_mean_squared_error', 
        cv=tscv, 
        verbose=0,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, Y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ XGBoost Optimization Complete. Best Params: {grid_search.best_params_}")
    
    print("\n--- XGBoost Feature Importance ---")
    importance = best_model.feature_importances_
    feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(10)
    print(feature_importance.to_string())
    
    return best_model

def evaluate_optimized_xgboost(model, X_test, Y_test):
    """Evaluates the final optimized XGBoost model."""
    predicted_price = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(Y_test, predicted_price))
    r2 = r2_score(Y_test, predicted_price)
    
    print("\n--- Optimized XGBoost Model Evaluation ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.4f}")
    
    return Y_test, predicted_price

