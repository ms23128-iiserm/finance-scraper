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

 # Conclusion based on p-value
    p_value = result[1]
    
    print('\n--- Conclusion ---')
    if p_value <= 0.05:
        print(f"✅ The p-value ({p_value:.4f}) is <= 0.05. *Reject the Null Hypothesis (H0)*.")
        print("The time series is likely *STATIONARY*.")
        return True
    else:
        print(f"❌ The p-value ({p_value:.4f}) is > 0.05. *Fail to Reject the Null Hypothesis (H0)*.")
        print("The time series is likely *NON-STATIONARY*.")
        print("You will need to use *differencing* before applying ARIMA/SARIMA models.")
        return False
def main():
    """Main function to load data and run the stationarity check."""
    
    print(f"--- Loading data from '{INPUT_FILE}' for stationarity check ---")
    try:
        # Load the feature-engineered data, which has a clean time index
        df = pd.read_csv(INPUT_FILE, parse_dates=True, index_col=0)
        
        if PRICE_COLUMN not in df.columns:
             print(f"❌ ERROR: Price column '{PRICE_COLUMN}' not found. Check your CSV.")
             return
             
        # Extract the closing price series
        price_series = df[PRICE_COLUMN].copy()
        
        # 1. Run the test on the original series
        print(f"\n--- Checking Stationarity for ORIGINAL {PRICE_COLUMN} Series ---")
        is_stationary = run_adf_test(price_series.dropna())
        
        if not is_stationary:
            # 2. If non-stationary, show the effect of differencing
            # This is the standard fix for non-stationarity
            print(f"\n\n--- Checking Stationarity for DIFFERENCED {PRICE_COLUMN} Series (Lag 1) ---")
            differenced_series = price_series.diff().dropna()
            run_adf_test(differenced_series)

        except FileNotFoundError:
          print(f"❌ ERROR: File not found: '{INPUT_FILE}'. Please run the feature engineering script first.")
        except Exception as e:
          print(f"❌ An unexpected error occurred: {e}")
if _name_ == "_main_":
    main()