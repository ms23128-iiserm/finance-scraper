import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_FILE = 'data_cleaned.csv'     # The file we just created
OUTPUT_FILE = 'features_engineered_data.csv' # The file this script will create
TARGET_VARIABLE = 'Reliance_Close'  # The column we want to predict
# ---------------------

def load_data(filepath):
    """Loads the cleaned, merged data."""
    print(f"--- Loading data from '{filepath}' ---")
    try:
        df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        df = df.asfreq('D') # Ensure we have a daily frequency for lags
        
        # After asfreq, some new NaNs might be created if days were missing
        # We re-apply ffill and bfill to ensure no gaps at all
        df = df.ffill().bfill() 
        
        print(f"✅ Data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: '{filepath}'")
        print("Please make sure you have successfully run '1b_data_cleaning.py' first.")
        return None

def create_target_variable(df, target_col):
    """
    Creates the 'target' column.
    The target is the next day's closing price.
    """
    print(f"--- Engineering Target Variable ('{target_col}') ---")
    # Shift the target column up by one day.
    # df['target'] for today will be df[target_col] for tomorrow.
    df['target'] = df[target_col].shift(-1)
    
    # We must drop the very last row, as it has no target (we can't know tomorrow's price)
    df = df.iloc[:-1]
    print("✅ 'target' column created (tomorrow's price).")
    return df

def create_lag_features(df, columns, lags=[1, 3, 7, 14]):
    """
    Creates 'lag' features (e.g., price from 1 day ago, 7 days ago).
    This is the most important feature for time-series models.
    """
    print(f"--- Engineering Lag Features (Lags: {lags}) ---")
    df_lags = df.copy()
    for col in columns:
        for lag in lags:
            col_name = f'{col}lag{lag}'
            df_lags[col_name] = df_lags[col].shift(lag)
    
    # Lag features create NaNs at the start of the dataset
    # We drop these rows as they can't be used for training
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
        
    # Calculate correlations
    corr_matrix = df.corr()
    
    # Get correlations with the target variable
    corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
    
    print("--- Top 15 Most Predictive Features (Correlation with Target) ---")
    print(corr_with_target.head(15).to_string())
    
    # Save a heatmap of the most important correlations
    try:
        top_features = corr_with_target.head(15).index
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[top_features].corr(), annot=True, cmap='viridis', fmt='.2f')
        plt.title('Correlation Heatmap of Top 15 Features')
        plt.tight_layout()
        heatmap_file = 'correlation_heatmap.png'
        plt.savefig(heatmap_file)
        print(f"\n✅ Correlation heatmap saved to '{heatmap_file}'")
    except Exception as e:
        print(f"\n⚠ Warning: Could not save heatmap image: {e}")

def main():
    """Main function to run the feature engineering pipeline."""
    
    df = load_data(INPUT_FILE)
    if df is None:
        return

    # Define which columns to create lag/rolling features for
    # We include our target variable and all external data
    feature_cols = [
        'Reliance_Close', 'Gold_Price', 'Petrol_Price', 
        'USD_INR_Rate', 'EUR_INR_Rate', 'Sentiment'
    ]

    # Pipeline
    df = create_target_variable(df, TARGET_VARIABLE)
    
    # We create rolling features FIRST, then lag features
    # This minimizes the number of rows dropped
    df = create_rolling_features(df, feature_cols)
    df = create_lag_features(df, feature_cols)
    
    # Date features don't drop rows, so they can run last
    df = create_date_features(df)
    
    # Re-align frequencies just in case
    df = df.asfreq('D').ffill()

    # Save the final dataset
    df.to_csv(OUTPUT_FILE)
    print(f"\n✨ --- SUCCESS --- ✨")
    print(f"Feature-engineered data saved to: {OUTPUT_FILE}")
    print(f"Final Shape: {df.shape}")

    # Run analysis on the final data
    run_feature_analysis(df)

if _name_ == "_main_":
    main()