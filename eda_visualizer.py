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
 # --- 2. Merge Data ---
    # Combine all DataFrames side-by-side using their common 'Date' index
    df = pd.concat([reliance_df, gold_df, petrol_df, forex_df], axis=1)
    
    # Handle missing values (e.g., market holidays) by filling with the last known price
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True) # Drop any remaining empty rows (from the start)

    print("✅ All financial data merged successfully.")
    print("\n--- Sample of Merged Data ---")
    print(df.head())

    # --- 3. Generate Visualizations ---
    print("\n📈 Generating plots... (Close plot windows to see the next one)")

    # Plot 1: Time-Series of Reliance Stock Price
    plt.figure(figsize=(14, 7))
    df['reliance_price'].plot()
    plt.title('Reliance Stock Price (Last 5 Years)')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.grid(True)
    plt.show()

    # Plot 2: Histograms of All Features
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle('Histograms of Feature Distributions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot 3: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix (Stocks, Gold, Petrol, Forex)')
    plt.show()
    
    print("\n--- ✅ EDA Complete ---")

if __name__ == '__main__':
    # You may need to install these libraries
    # pip install matplotlib seaborn
    perform_financial_eda()
    
