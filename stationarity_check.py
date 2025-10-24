import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
PRICE_COLUMN = 'Reliance_Close' 
# ---------------------
