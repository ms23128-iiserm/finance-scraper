import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import os 
import warnings
# Suppress harmless warnings and TensorFlow verbose output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
TARGET_PRICE = 'target'
TRAIN_SPLIT_RATIO = 0.8
N_STEPS = 60         # Time steps (past days) for LSTM input
N_FUTURE_DAYS = 15   # Constant for the forecast horizon
# ---------------------
