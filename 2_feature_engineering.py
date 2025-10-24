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
