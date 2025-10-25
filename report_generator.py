import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
# File from the XGBoost Walk-Forward Validation (WFV)
WFV_RESULTS_FILE = 'walk_forward_predictions.csv' 
# File containing the original features (for Buy & Hold calculation)
FEATURE_DATA_FILE = 'features_engineered_data.csv' 
# File created by 5_advanced_modeling.py
ADVANCED_RESULTS_FILE = 'advanced_model_results.csv' 
TARGET_PRICE_COL = 'target'
# --- END CONFIGURATION ---

def calculate_pnl_from_wfv(wfv_df, features_df):
    """Calculates P&L for the WFV XGBoost model and the Buy & Hold baseline."""
    
    # 1. Merge WFV results with the true 'Reliance_Close' price (Entry Price)
    wfv_df = wfv_df.merge(features_df[['Reliance_Close']], left_index=True, right_index=True, how='left')
    wfv_df = wfv_df.dropna()
    
    # 2. Calculate Model P&L: (Exit Price - Entry Price) if prediction is 1, else 0
    wfv_df['Model_Return'] = np.where(
        wfv_df['Predicted_Direction'] == 1, 
        wfv_df[TARGET_PRICE_COL] - wfv_df['Reliance_Close'], 
        0
    )
    final_model_pnl = wfv_df['Model_Return'].sum()
    
    # 3. Calculate Buy & Hold P&L
    initial_price = wfv_df['Reliance_Close'].iloc[0]
    final_price = wfv_df[TARGET_PRICE_COL].iloc[-1]
    buy_and_hold_pnl = final_price - initial_price
    
    return final_model_pnl, buy_and_hold_pnl

def calculate_metrics(actual, predicted):
    """Calculates RMSE and R2 for regression models."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return rmse, r2

def get_advanced_model_metrics():
    """Loads advanced model results and calculates metrics for the final table."""
    try:
        adv_df = pd.read_csv(ADVANCED_RESULTS_FILE, parse_dates=True, index_col=0).dropna()
    except FileNotFoundError:
        print(f"❌ ERROR: Advanced results file '{ADVANCED_RESULTS_FILE}' not found. Cannot populate TBDs.")
        return {}
    
    actual = adv_df[TARGET_PRICE_COL]
    
    metrics = {}
    
    for col in adv_df.columns:
        if col != TARGET_PRICE_COL and col.startswith('Predicted_'):
            predicted = adv_df[col]
            rmse, r2 = calculate_metrics(actual, predicted)
            
            display_name = col.replace('Predicted_', '')
            metrics[display_name] = {'RMSE': rmse, 'R2': r2}

    return metrics

