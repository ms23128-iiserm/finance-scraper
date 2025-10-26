import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
DATA_FILE = 'iiser_fees.csv'
TARGET_COLUMN = 'Tuition_Fee'  # The fee we want to predict
FORECAST_YEARS = 5
# ---------------------

def load_data(filepath):
    """Loads the manually collected fee data."""
    print(f"--- Loading Data from '{filepath}' ---")
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"❌ ERROR: '{filepath}' is empty.")
            return None
        print("✅ Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: File not found: '{filepath}'")
        print("Please make sure 'iiser_fees.csv' is in the same folder.")
        return None

def train_model(df):
    """Trains a simple linear regression model."""
    print("--- Training Forecasting Model ---")
    
    # --- Define Features (X) and Target (y) ---
    # We will use 'Year' and 'India_CPI' to predict the 'Tuition_Fee'
    features = ['Year', 'India_CPI']
    
    if not all(col in df.columns for col in features):
        print(f"❌ ERROR: Your CSV is missing required columns. It must have: {features}")
        return None, None
        
    X = df[features]
    y = df[TARGET_COLUMN]
    
    if len(X) < 5:
        print(f"❌ ERROR: Not enough data. You need at least 5-10 years of data.")
        return None, None

    model = LinearRegression()
    model.fit(X, y)
    
    # --- Check Model Performance ---
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    print(f"✅ Model training complete. R-squared on historical data: {r2:.4f}")
    if r2 < 0.9:
        print(f"⚠️ WARNING: R-squared is low. The model's predictions may not be accurate.")
        
    return model, features

def make_forecast(model, last_year, last_cpi):
    """Predicts the next 5 years of fees."""
    print(f"\n--- Generating 5-Year Forecast (2025-2029) ---")
    
    # --- Create the "Future" DataFrame ---
    future_years = np.arange(last_year + 1, last_year + 1 + FORECAST_YEARS)
    
    # Assumption: We'll assume future inflation hovers around the last known value (4.95%)
    # This is a simple but necessary assumption for forecasting.
    assumed_future_cpi = np.full(FORECAST_YEARS, last_cpi) 
    
    # Create the future feature set
    X_future = pd.DataFrame({
        'Year': future_years,
        'India_CPI': assumed_future_cpi
    })
    
    # Make the predictions
    future_predictions = model.predict(X_future)
    
    # Format for printing
    forecast_df = pd.DataFrame({
        'Year': future_years,
        'Predicted_Tuition_Fee': np.round(future_predictions) # Round to nearest rupee
    })
    forecast_df.set_index('Year', inplace=True)
    
    print("✅ 5-Year Forecast Generated:")
    print(forecast_df.to_string(float_format='%.0f'))
    
    return forecast_df

def plot_results(df, forecast_df):
    """Plots the historical data and the new forecast."""
    print(f"\n--- Saving Forecast Plot to 'fee_forecast_plot.png' ---")
    
    plt.figure(figsize=(12, 7))
    
    # Plot historical data
    plt.plot(df['Year'], df[TARGET_COLUMN], 'bo-', label='Historical Fees')
    
    # Plot forecasted data
    plt.plot(forecast_df.index, forecast_df['Predicted_Tuition_Fee'], 'r--', label=f'{FORECAST_YEARS}-Year Forecast')
    
    plt.title(f'IISER Mohali {TARGET_COLUMN} Forecast')
    plt.xlabel('Year')
    plt.ylabel('Fee (INR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('fee_forecast_plot.png')
    print("✅ Plot saved successfully.")

def main():
    df = load_data(DATA_FILE)
    if df is None:
        return
        
    model, features = train_model(df)
    if model is None:
        return
        
    # Get the last known values to base our forecast on
    last_year = df['Year'].max()
    last_cpi = df.loc[df['Year'] == last_year, 'India_CPI'].values[0]
    
    forecast_df = make_forecast(model, last_year, last_cpi)
    
    plot_results(df, forecast_df)

if __name__ == "__main__":
    main()

