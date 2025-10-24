import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_FILE = 'data_cleaned.csv' 
OUTPUT_FILE = 'features_engineered_data.csv'
TARGET_VARIABLE = 'Reliance_Close' 
# ---------------------

def load_data(filepath):
    """Loads the cleaned, merged data."""
    print(f"--- Loading data from '{filepath}' ---")
    try:
        df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        df = df.asfreq('D') # Ensure daily frequency
        
        # We re-apply ffill and bfill to ensure no gaps at all
        df = df.ffill().bfill() 
        
        print(f"✅ Data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: '{filepath}'")
        print("Please make sure you have successfully run '2_data_clean_merge.py' first.")
        return None

def create_target_variable(df, target_col):
    """
    Creates the 'target' column (next day's closing price).
    """
    print(f"--- Engineering Target Variable ('{target_col}') ---")
    # df['target'] for today will be df[target_col] for tomorrow.
    df['target'] = df[target_col].shift(-1)
    
    # Drop the very last row, as it has no target
    df = df.iloc[:-1]
    print("✅ 'target' column created (tomorrow's price).")
    return df

def create_lag_features(df, columns, lags=[1, 3, 7, 14]):
    """
    Creates 'lag' features (e.g., price from 1 day ago, 7 days ago).
    """
    print(f"--- Engineering Lag Features (Lags: {lags}) ---")
    df_lags = df.copy()
    for col in columns:
        for lag in lags:
            col_name = f'{col}lag{lag}'
            df_lags[col_name] = df_lags[col].shift(lag)
    
    # Lag features create NaNs at the start of the dataset, drop them
    original_rows = len(df_lags)
    df_lags = df_lags.dropna()
    print(f"✅ Lag features created. Dropped {original_rows - len(df_lags)} rows with initial NaN values.")
    return df_lags
