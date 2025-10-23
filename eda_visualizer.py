import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# --- CONFIGURATION ---
DATABASE_FILE = "market_data.db" 

def perform_financial_eda():
    """
    Loads all financial data from the separate tables,
    merges them, and creates key EDA visualizations.
    """
    print("--- 📊 Starting Exploratory Data Analysis (EDA) ---")

    # --- 1. Load and Prepare Data ---
    try:
        # Create the connection engine for pandas
        engine = create_engine(f'sqlite:///{DATABASE_FILE}')
        
        print(f"Connecting to database: {DATABASE_FILE}")

        # Load each table and select only the 'Close' price, renaming it.
        # We set 'Date' as the index to make merging easy.
        
        reliance_df = pd.read_sql('SELECT "Date", "Close" FROM reliance_stocks', engine, parse_dates=['Date'])
        reliance_df.rename(columns={'Close': 'reliance_price'}, inplace=True)
        reliance_df.set_index('Date', inplace=True)

        gold_df = pd.read_sql('SELECT "Date", "Close" FROM gold_prices', engine, parse_dates=['Date'])
        gold_df.rename(columns={'Close': 'gold_price'}, inplace=True)
        gold_df.set_index('Date', inplace=True)

        petrol_df = pd.read_sql('SELECT "Date", "Close" FROM petrol_prices', engine, parse_dates=['Date'])
        petrol_df.rename(columns={'Close': 'petrol_price'}, inplace=True)
        petrol_df.set_index('Date', inplace=True)

        forex_df = pd.read_sql('SELECT "Date", "Close" FROM forex_rates', engine, parse_dates=['Date'])
        forex_df.rename(columns={'Close': 'forex_rate'}, inplace=True)
        forex_df.set_index('Date', inplace=True)

        print("✅ Data loaded successfully from all tables.")

    except Exception as e:
        print(f"❌ ERROR: Could not read data. Check your table names (e.g., 'reliance_stocks').")
        print(f"Details: {e}")
        return

    
