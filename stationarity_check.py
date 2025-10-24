import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

# --- Configuration ---
INPUT_FILE = 'features_engineered_data.csv' 
PRICE_COLUMN = 'Reliance_Close' 
# ---------------------

def run_adf_test(series):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.
    """
    print('Running Augmented Dickey-Fuller Test...')
    result = adfuller(series, autolag='AIC')
    
    adf_output = pd.Series(result[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Observations Used'])
    for key, value in result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    
    print('\n--- ADF Test Results ---')
    print(adf_output.to_string())
