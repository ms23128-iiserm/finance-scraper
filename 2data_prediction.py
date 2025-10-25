import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm # Used for progress bar during walk-forward validation

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
OUTPUT_FILE = 'walk_forward_predictions.csv'
TARGET_PRICE = 'target' 
TARGET_DIRECTION = 'direction'
TRAIN_WINDOW_SIZE = 365 # Start training with 365 days of data (approx 1 year)
# ---------------------

def load_data(filepath):
    """Loads the feature-engineered data."""
    print(f"\n--- Loading data from '{filepath}' ---")
    try:
        df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        
        # 1. Create Direction Target
        # Direction is 1 (Up) if next day's price > today's close, else 0 (Down/Flat)
        # We assume 'Reliance_Close' is today's closing price
        df[TARGET_DIRECTION] = np.where(df[TARGET_PRICE] > df['Reliance_Close'], 1, 0)
        
        print(f"✅ Data loaded. Shape: {df.shape}")
        print(f"Direction Target Ratio (0/1): {df[TARGET_DIRECTION].value_counts(normalize=True).to_dict()}")
        
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: '{filepath}'")
        return None

def prepare_X_Y(df, target_cols):
    """Separates features (X) and targets (Y)."""
    # Features are all columns EXCEPT the target columns
    X = df.drop(columns=target_cols).copy()
    Y = df[target_cols].copy()
    
    # Drop non-predictive/non-numeric columns if any remained
    X = X.select_dtypes(include=np.number)
    
    # Critical: Ensure no NaNs remain after final feature set is chosen
    initial_len = len(X)
    combined = pd.concat([X, Y], axis=1).dropna()
    X = combined[X.columns]
    Y = combined[Y.columns]
    if len(X) != initial_len:
         print(f"⚠ Dropped {initial_len - len(X)} rows due to final NaNs.")

    return X, Y

def walk_forward_validation(X, Y):
    """
    Performs Walk-Forward Validation (WFV) using XGBoost models.
    The model is retrained periodically on a rolling window.
    """
    print(f"\n--- Starting Walk-Forward Validation (Train Window: {TRAIN_WINDOW_SIZE} days) ---")

    # The test set starts after the initial training window
    test_start_index = TRAIN_WINDOW_SIZE
    
    # Lists to store WFV results
    price_predictions = []
    direction_predictions = []
    actual_values = []
    
    # Loop over the test set, making one-step prediction each time
    for i in tqdm(range(test_start_index, len(X)), desc="WFV Progress"):
        
        # 1. Define the current training and testing windows
        X_train, Y_train = X.iloc[:i], Y.iloc[:i]
        X_test, Y_test = X.iloc[i:i+1], Y.iloc[i:i+1] # Test window is always the next single day
        
        # 2. Regression Model (Price Prediction)
        reg_model = XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            random_state=42, 
            n_jobs=-1
        )
        reg_model.fit(X_train, Y_train[TARGET_PRICE])
        price_pred = reg_model.predict(X_test)[0]
        price_predictions.append(price_pred)

        # 3. Classification Model (Direction Prediction)
        pos_count = Y_train[TARGET_DIRECTION].sum()
        neg_count = len(Y_train) - pos_count
        scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0

        cls_model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            scale_pos_weight=scale_pos_weight_val, # ADDRESSES IMBALANCE
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42, 
            n_jobs=-1
        )
        cls_model.fit(X_train, Y_train[TARGET_DIRECTION])
        direction_pred = cls_model.predict(X_test)[0]
        direction_predictions.append(direction_pred)

        # Store the actual next day values
        actual_values.append(Y_test.iloc[0].to_dict())

    # 4. Compile Results DataFrame
    results_df = pd.DataFrame(actual_values, index=X.index[test_start_index:])
    results_df['Predicted_Price'] = price_predictions
    results_df['Predicted_Direction'] = direction_predictions
    
    print("\n✅ Walk-Forward Validation Complete.")
    return results_df

def evaluate_wfv_results(results_df):
    """Calculates and prints final WFV metrics."""
    print("\n--- Final Model Evaluation (Walk-Forward Results) ---")
    
    # --- A. Regression Evaluation ---
    actual_price = results_df[TARGET_PRICE]
    pred_price = results_df['Predicted_Price']
    
    # Calculate R-squared and RMSE for the entire out-of-sample period
    rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
    
    # Calculate R2 using the custom formula for time-series (1 - MSE/Variance of Test Data)
    var_test = np.var(actual_price)
    ts_r2 = 1 - (mean_squared_error(actual_price, pred_price) / var_test)
    
    print("\n💰 Price Prediction (XGBoost Regressor):")
    print(f"Walk-Forward Test RMSE: {rmse:.2f}")
    print(f"Walk-Forward Test R-squared: {ts_r2:.4f}")
    
    # --- B. Classification Evaluation ---
    actual_direction = results_df[TARGET_DIRECTION]
    pred_direction = results_df['Predicted_Direction']

    print("\n📈 Direction Prediction (XGBoost Classifier with scale_pos_weight):")
    print(classification_report(actual_direction, pred_direction, target_names=['Down/Flat (0)', 'Up (1)']))
    print(f"Confusion Matrix:\n{confusion_matrix(actual_direction, pred_direction)}")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(actual_price.index, actual_price.values, label='Actual Next Day Price', color='blue', linewidth=1)
    plt.plot(pred_price.index, pred_price.values, label='Predicted Next Day Price', color='red', linestyle='--', linewidth=1)
    plt.title('Walk-Forward Prediction: Actual vs. XGBoost Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('xgboost_walk_forward_plot.png')
    print("✅ Walk-Forward plot saved to 'xgboost_walk_forward_plot.png'")

# ==============================================================================
#                 NEW P&L (PROFIT & LOSS) SIMULATION FUNCTION
# ==============================================================================

def simulate_trading_strategy(results_df, pred_direction_col='Predicted_Direction', target_price_col='target'):
    """
    Simulates a simple trading strategy (Buy when model predicts UP) 
    and calculates cumulative P&L against a Buy & Hold baseline.
    """
    print("\n--- Starting Trading Strategy Simulation (P&L) ---")
    
    df_sim = results_df.copy()
    
    # Load the current day's Close price for the trade entry price
    try:
        df_features = pd.read_csv(INPUT_FILE, parse_dates=True, index_col=0)['Reliance_Close']
        df_sim = df_sim.merge(df_features, left_index=True, right_index=True, how='left')
        close_col = 'Reliance_Close'
    except Exception as e:
        print(f"❌ ERROR: Could not merge Reliance_Close for backtest ({e}). Skipping P&L.")
        return

    # 1. Calculate Daily Returns only on 'Buy' days
    # Daily P&L = (Exit Price - Entry Price) if model predicted 1, else 0.
    df_sim['Buy_Return'] = np.where(
        df_sim[pred_direction_col] == 1, 
        df_sim[target_price_col] - df_sim[close_col], 
        0
    )

    # 2. Calculate Cumulative P&L
    df_sim['Cumulative_P&L'] = df_sim['Buy_Return'].cumsum()
    
    # 3. Calculate Buy-and-Hold Baseline
    initial_price = df_sim[close_col].iloc[0]
    final_price = df_sim[target_price_col].iloc[-1]
    buy_and_hold_return = final_price - initial_price

    # --- Print Results ---
    final_pnl = df_sim['Cumulative_P&L'].iloc[-1]
    
    print("\n--- Trading Strategy Summary ---")
    print(f"Model Prediction P&L: ₹{final_pnl:.2f}")
    print(f"Buy & Hold P&L:       ₹{buy_and_hold_return:.2f}")
    
    if final_pnl > buy_and_hold_return:
        print("🎉 Conclusion: Model OUTPERFORMED the simple Buy & Hold strategy in the test period.")
    else:
        print("😔 Conclusion: Model UNDERPERFORMED the simple Buy & Hold strategy in the test period.")

    # 4. Plotting P&L
    plt.figure(figsize=(14, 6))
    plt.plot(df_sim.index, df_sim['Cumulative_P&L'], label='Model Strategy P&L', color='green', linewidth=2)
    plt.axhline(y=buy_and_hold_return, color='red', linestyle='--', label='Buy & Hold P&L')
    plt.title('Backtesting Result: Model Strategy vs. Buy & Hold (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit/Loss (INR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('backtest_pnl_plot.png')
    print("✅ P&L plot saved to 'backtest_pnl_plot.png'")
    
    return df_sim

# ==============================================================================
#                               MAIN EXECUTION
# ==============================================================================

def main():
    """Main function to run the upgraded pipeline."""
    
    df = load_data(INPUT_FILE)
    if df is None:
        return

    # 1. Prepare X and Y for WFV
    TARGET_COLUMNS = [TARGET_PRICE, TARGET_DIRECTION]
    X, Y = prepare_X_Y(df, TARGET_COLUMNS)
    
    # 2. Run Walk-Forward Validation
    results_df = walk_forward_validation(X, Y)
    
    # 3. Evaluate and Save Results
    evaluate_wfv_results(results_df)
    
    results_df.to_csv(OUTPUT_FILE)
    
    print(f"\nWalk-forward prediction results saved to: {OUTPUT_FILE}")
    
    # 4. Run P&L Simulation
    simulate_trading_strategy(
        results_df, 
        pred_direction_col='Predicted_Direction', 
        target_price_col=TARGET_PRICE
    )
    
    print("\n✨ --- WFV & P&L Analysis Complete --- ✨")

if __name__ == "main":
    main()