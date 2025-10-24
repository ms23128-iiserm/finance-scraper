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





