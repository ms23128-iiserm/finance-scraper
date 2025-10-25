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


        






