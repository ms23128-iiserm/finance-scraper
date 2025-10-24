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

def create_rolling_features(df, columns, windows=[7, 30]):
    """
    Creates rolling mean (trend) and rolling std (volatility) features.
    """
    print(f"--- Engineering Rolling Features (Windows: {windows}) ---")
    df_roll = df.copy()
    for col in columns:
        for window in windows:
            # Rolling Mean (Trend)
            roll_mean_name = f'{col}roll_mean{window}'
            df_roll[roll_mean_name] = df_roll[col].rolling(window=window).mean()
            
            # Rolling Std (Volatility)
            roll_std_name = f'{col}roll_std{window}'
            df_roll[roll_std_name] = df_roll[col].rolling(window=window).std()

    # Rolling features also create NaNs at the start
    original_rows = len(df_roll)
    df_roll = df_roll.dropna()
    print(f"✅ Rolling features created. Dropped {original_rows - len(df_roll)} rows with initial NaN values.")
    return df_roll
def create_technical_indicators(df, price_col='Reliance_Close'):
    """
    Calculates common technical analysis indicators:
    RSI, MACD, and Bollinger Bands (BB).
    """
    print("--- Engineering Advanced Technical Indicators ---")
    df_tech = df.copy()

    # --- 1. Relative Strength Index (RSI) ---
    window = 14
    delta = df_tech[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    # Handle division by zero for initial calculation if avg_loss is 0
    rs = avg_gain / avg_loss.replace(0, np.nan) 
    df_tech['RSI'] = 100 - (100 / (1 + rs))

    # --- 2. Moving Average Convergence Divergence (MACD) ---
    ema_fast = df_tech[price_col].ewm(span=12, adjust=False).mean()
    ema_slow = df_tech[price_col].ewm(span=26, adjust=False).mean()
    
    df_tech['MACD'] = ema_fast - ema_slow
    df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
# --- 3. Bollinger Bands (BB) ---
    window_bb = 20
    df_tech['BB_SMA'] = df_tech[price_col].rolling(window=window_bb).mean()
    df_tech['BB_StdDev'] = df_tech[price_col].rolling(window=window_bb).std()
    
    df_tech['BB_Upper'] = df_tech['BB_SMA'] + (df_tech['BB_StdDev'] * 2)
    df_tech['BB_Lower'] = df_tech['BB_SMA'] - (df_tech['BB_StdDev'] * 2)

    # Drop helper columns
    df_tech = df_tech.drop(columns=['BB_SMA', 'BB_StdDev'])
    
    # Drop NaNs created by the moving windows (RSI, MACD, BB)
    original_rows = len(df_tech)
    df_tech = df_tech.dropna()
    print(f"✅ Technical Indicators created. Dropped {original_rows - len(df_tech)} rows (due to {window} & {window_bb} day windows).")
    return df_tech
def create_date_features(df):
    """Creates features from the date index (e.g., day of week, month)."""
    print("--- Engineering Date Features ---")
    df_date = df.copy()
    df_date['day_of_week'] = df_date.index.dayofweek
    df_date['day_of_month'] = df_date.index.day
    df_date['month'] = df_date.index.month
    df_date['year'] = df_date.index.year
    print("✅ Date features created.")
    return df_date

def run_feature_analysis(df, target_col='target'):
    """
    Performs a correlation analysis to find the most predictive features.
    """
    print("\n--- Running Feature Analysis ---")
    if df.empty or target_col not in df.columns:
        print("❌ Cannot run analysis, DataFrame is empty or target column is missing.")
        return
if __name__ == "__main__":
    main()
