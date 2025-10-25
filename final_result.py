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

# ==============================================================================
# 4. RECURSIVE 15-DAY FORECAST
# ==============================================================================

def predict_recursive_xgboost(model, initial_data, feature_names, n_days=N_FUTURE_DAYS):
    """Performs a multi-step, recursive prediction for n_days using the trained XGBoost model."""
    print(f"\n--- Starting Recursive XGBoost Forecast for {n_days} Days ---")
    
    context_size = 30 
    # Use the last 30 days of the price column for initial context
    current_features = initial_data.iloc[-context_size:].copy()

    forecast_prices = []
    last_date = current_features.index[-1]
    
    # Identify relevant lag/rolling windows from feature names
    lags = [int(f.split('lag')[1]) for f in feature_names if 'lag' in f]
    rolls = [int(f.split('mean')[1]) for f in feature_names if 'mean' in f]
    
    for i in range(1, n_days + 1):
        
        next_date = last_date + pd.Timedelta(days=i)
        
        new_features = pd.Series(index=feature_names, dtype=float)
        
        # 1. Populate Time Features
        new_features['year'] = next_date.year
        new_features['month'] = next_date.month
        new_features['day'] = next_date.day
        
        # 2. Populate Lagged and Rolling Features (The Recursive Step)
        history = current_features['Reliance_Close']
        
        # Lag 1
        new_features['Reliance_Closelag1'] = history.iloc[-1]
        
        # Other Lags
        for lag in lags:
            lag_col = f'Reliance_Closelag{lag}'
            if lag_col in feature_names and len(history) >= lag:
                new_features[lag_col] = history.iloc[-lag]
                
        # Rolling Means
        for window in rolls:
            roll_col = f'Reliance_Closeroll_mean{window}'
            if roll_col in feature_names:
                new_features[roll_col] = history.iloc[-window:].mean()
        
        # 3. Predict
        X_predict = pd.DataFrame([new_features], index=[next_date])
        X_predict = X_predict[feature_names].fillna(method='ffill') 
        
        predicted_price = model.predict(X_predict)[0]
        
        # 4. Update Context (Recurse)
        # Create a new row to append to the historical context
        new_price_row = pd.Series(
            [predicted_price], 
            index=[next_date], 
            name='Reliance_Close'
        ).to_frame(name='Reliance_Close')
        
        current_features = pd.concat([current_features, new_price_row])
        
        forecast_prices.append((next_date, predicted_price))

    forecast_series = pd.Series([p for d, p in forecast_prices], 
                                index=[d for d, p in forecast_prices],
                                name='Predicted Price (15-Day Recursive)')
    
    print("✅ Recursive Forecast Complete.")
    return forecast_series

# ==============================================================================
# 5. DATA CONSOLIDATION AND PLOTTING
# ==============================================================================

def save_advanced_model_results(Y_test_full, results_list):
    """Consolidates all model predictions and the actual target into a single DataFrame."""
    final_df = pd.DataFrame({TARGET_PRICE: Y_test_full})
    
    for actual, prediction_series, label in results_list:
        column_name = f'Predicted_{label.replace(" ", "")}'
        final_df[column_name] = np.nan
        final_df.loc[prediction_series.index, column_name] = prediction_series.values
        
    final_df.to_csv(ADVANCED_RESULTS_FILE)
    print(f"\n✅ All predictions consolidated and saved to '{ADVANCED_RESULTS_FILE}'.")
    
def plot_results(results_list, title="Comparative Model Forecasting"):
    """Plots the actual price vs. predictions from all models."""
    plt.figure(figsize=(18, 8))
    
    actual_prices = results_list[0][0]
    plt.plot(actual_prices.index, actual_prices.values, label='Actual Reliance Price (Test Set)', color='black', linewidth=2)
    
    for actual, prediction, label in results_list:
        plt.plot(prediction.index, prediction.values, label=f'Predicted Price ({label})', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Closing Price (INR)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparative_model_forecast.png')
    print("✅ Comparative model forecast plot saved to 'comparative_model_forecast.png'")

# ==============================================================================
#                               MAIN EXECUTION
# ==============================================================================

def main():
    X_train, X_test, Y_train, Y_test, feature_names = load_and_split_data(INPUT_FILE)
    
    # Load the full data to provide context for the recursive forecast
    full_df = pd.read_csv(INPUT_FILE, parse_dates=True, index_col=0).dropna()
    
    all_results = []
    
    # --- 1. XGBoost Optimization ---
    best_xgb_model = optimize_xgboost(X_train, Y_train, feature_names)
    Y_test_xgb, predicted_xgb = evaluate_optimized_xgboost(best_xgb_model, X_test, Y_test)
    all_results.append((Y_test_xgb, pd.Series(predicted_xgb, index=Y_test_xgb.index), "OptimizedXGBoost"))

    # --- 2. ARIMA Baseline ---
    Y_test_arima, predicted_arima = train_and_evaluate_arima(Y_train, Y_test)
    all_results.append((Y_test_arima, pd.Series(predicted_arima, index=Y_test_arima.index), "ARIMABaseline"))
    
    # --- 3. LSTM Deep Learning ---
    X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_lstm_eval, scaler_Y = prepare_lstm_data(X_train, X_test, Y_train, Y_test)
    Y_test_lstm, predicted_lstm, _ = train_and_evaluate_lstm(X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_lstm_eval, scaler_Y)
    all_results.append((Y_test_lstm, pd.Series(predicted_lstm, index=Y_test_lstm.index), "LSTMNetwork"))
    
    # --- 4. Data Consolidation and Plotting ---
    save_advanced_model_results(Y_test, all_results)
    plot_results(all_results)
    
    