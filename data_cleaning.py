import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
DB_NAME = 'market_data.db'
CLEANED_CSV_NAME = 'data_cleaned.csv'

# ==============================================================================
#  THIS IS THE FIX:
#  The config now uses the exact (and strange) string-tuple column names 
#  that we discovered in your database from the debug_db.py output.
# ==============================================================================
TABLE_CONFIG = {
    'reliance_stocks': {
        'date_col': "('Date', '')",
        'rename_cols': {
            "('stock_price', 'RELIANCE.NS')": 'Reliance_Close'
        }
    },
    'gold_prices': {
        'date_col': "('Date', '')",
        'rename_cols': {
            "('Gold Price (USD/oz)', 'GC=F')": 'Gold_Price'
        }
    },
    'petrol_prices': {
        'date_col': "('Date', '')",
        'rename_cols': {
            "('oil_price', 'CL=F')": 'Petrol_Price'
        }
    },
    'forex_rates': {
        'date_col': "('date', '')",
        'rename_cols': {
            "('USDINR=X', 'USDINR=X')": 'USD_INR_Rate',
            "('EURINR=X', 'EURINR=X')": 'EUR_INR_Rate'
        }
    },
    'market_news': {
        'date_col': 'date', # This table was correct, no change needed
        'rename_cols': {
            'headline': 'Headline'
        }
    }
}
# ==============================================================================

def check_db_exists():
    """Checks if the database file exists before trying to read it."""
    if not os.path.exists(DB_NAME):
        print(f"❌ CRITICAL ERROR: Database file '{DB_NAME}' not found.")
        print(f"Please run your main scraper script (e.g., 'python main.py') first to create it.")
        return False
    return True

def load_and_clean_table(conn, table_name, config):
    """
    Loads a single table, cleans it, and sets the date as the index.
    """
    print(f"\n   -> Loading table: {table_name}")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except sqlite3.OperationalError:
        print(f"   ❌ ERROR: No such table '{table_name}'. Skipping.")
        return None

    if df.empty:
        print(f"   ⚠ WARNING: Table '{table_name}' is empty. Skipping.")
        return None

    # --- Robustness Check ---
    if config['date_col'] not in df.columns:
        print(f"   ❌ ERROR: Date column '{config['date_col']}' not found in '{table_name}'.")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
        
    for old_col in config['rename_cols'].keys():
        if old_col not in df.columns:
            print(f"   ❌ ERROR: Expected column '{old_col}' not found in '{table_name}'.")
            print(f"   Available columns: {df.columns.tolist()}")
            return None
    # --- End Check ---

    # 1. Convert date column to datetime objects
    # Use errors='coerce' to handle any problematic date formats
    df[config['date_col']] = pd.to_datetime(df[config['date_col']], errors='coerce')
    
    # 2. Rename columns for standardization
    df = df.rename(columns=config['rename_cols'])
    
    # 3. Keep only the columns we need (the new names + the date col)
    required_cols = [config['date_col']] + list(config['rename_cols'].values())
    df = df[required_cols]
    
    # 4. Handle duplicates: Keep only the first entry for any given day
    original_rows = len(df)
    df = df.drop_duplicates(subset=[config['date_col']], keep='first')
    if original_rows > len(df):
        print(f"   -> Dropped {original_rows - len(df)} duplicate date entries.")
        
    # 5. Set the date as the index (crucial for merging)
    df = df.set_index(config['date_col'])
    
    print(f"   -> Loaded and cleaned {len(df)} rows.")
    return df

def process_news_sentiment(df):
    """
    Analyzes sentiment of headlines and aggregates to a daily score.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=pd.to_datetime([])) # Return empty DF with datetime index
        
    print("\n2. Processing News Sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    
    if 'Headline' not in df.columns:
        print("   ❌ ERROR: 'Headline' column not found in news data. Skipping sentiment analysis.")
        return pd.DataFrame(index=pd.to_datetime([]))
        
    # Ensure headline is a string
    df['Headline'] = df['Headline'].astype(str)
    df['compound_score'] = df['Headline'].apply(lambda h: analyzer.polarity_scores(h)['compound'])
    
    # Group by date and calculate the mean sentiment for that day
    # We reset the index to get 'date' back as a column, then set it again.
    daily_sentiment = df.reset_index().groupby(pd.Grouper(key=df.index.name, freq='D'))['compound_score'].mean()
    
    daily_sentiment_df = daily_sentiment.to_frame().rename(columns={'compound_score': 'Sentiment'})
    
    print(f"   -> Analyzed {len(df)} headlines into {len(daily_sentiment_df)} daily scores.")
    return daily_sentiment_df

def merge_and_clean_data(dataframes):
    """
    Merges all DataFrames, forward-fills missing data, and saves to CSV.
    """
    if not dataframes:
        print("❌ No dataframes to merge. Exiting.")
        return

    print("\n3. Merging All Datasets...")
    
    master_df = pd.concat(dataframes, axis=1, join='outer')
    master_df = master_df.sort_index()
    
    print(f"   -> Combined Data Shape (before cleaning): {master_df.shape}")
    print("   -> Null values BEFORE cleaning:\n", master_df.isnull().sum())
    
    # ==========================================================================
    #  THIS IS THE FIX:
    #  We fill missing data in a smarter way to prevent data loss.
    # ==========================================================================
    
    # 1. Forward-fill all data. This fills weekends for prices
    #    and carries the last known sentiment score forward.
    master_df = master_df.ffill()
    
    # 2. Handle remaining NaNs for Sentiment.
    #    Any NaNs left are at the beginning of the dataset, before our
    #    first news article. We'll fill these with 0.0 (neutral).
    master_df['Sentiment'] = master_df['Sentiment'].fillna(0.0)
    
    # 3. Drop any remaining nulls from the price data.
    #    This will only drop rows at the very start if all data was missing.
    master_df = master_df.dropna()
    
    # ==========================================================================
    
    print(f"\n   -> Null values AFTER cleaning:\n", master_df.isnull().sum())
    
    # Save the final, clean dataset
    master_df.to_csv(CLEANED_CSV_NAME)
    print(f"\n✨ --- SUCCESS --- ✨")
    print(f"Clean, merged data saved to: {CLEANED_CSV_NAME}")
    print(f"Final Shape: {master_df.shape}")

def main():
    print("--- Starting Data Merge & Clean Script ---")
    if not check_db_exists():
        return

    all_dfs = []
    news_df = None

    try:
        conn = sqlite3.connect(DB_NAME)
        print(f"✅ Database connection successful: {DB_NAME}")

        for table, config in TABLE_CONFIG.items():
            if table == 'market_news':
                continue # Handle news separately
            df = load_and_clean_table(conn, table, config)
            if df is not None:
                all_dfs.append(df)
        
        # Load and process news
        news_df_raw = load_and_clean_table(conn, 'market_news', TABLE_CONFIG['market_news'])
        if news_df_raw is not None:
            news_sentiment_df = process_news_sentiment(news_df_raw)
            all_dfs.append(news_sentiment_df)

        conn.close()

    except Exception as e:
        print(f"❌ CRITICAL ERROR during database processing: {e}")
        if conn:
            conn.close()
        return
    
    merge_and_clean_data(all_dfs)
if  __name__ == "__main__":
     main()