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

