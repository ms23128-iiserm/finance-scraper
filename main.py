from datetime import datetime
import pandas as pd
import sqlite3 

# --- CONFIGURATION PARAMETERS ---
# Define the date range for your time-series scrapers
TODAY = datetime.now().strftime('%Y-%m-%d')

# The calculation uses 5 * 365 days as a simple approximation for 5 years.
START_DATE = (datetime.now() - pd.Timedelta(days=5 * 365)).strftime('%Y-%m-%d') 

# >> ACTION REQUIRED: REPLACE 'PLACEHOLDER' WITH YOUR ACTUAL KEY <<
NEWS_API_KEY = "eab586f731354326a3c2b38a2833be78" 

# Database configuration
DATABASE_FILE = "market_data.db" 
# --------------------------------

# --- Import all your specific scraping functions ---
from scraper_reliance import scrape_reliance_data
from scraper_gold import scrape_gold_data
from scraper_petrol import scrape_petrol_data
from scraper_forex import scrape_forex
from scraper_news import scrape_news_data


def standardize_data_format(data):
    """
    Converts data from Pandas DataFrame/Series to a list of dictionaries, 
    which is the safe format for insertion via to_sql, handling cases where 
    data might already be a list or None.
    """
    if isinstance(data, pd.DataFrame):
        # Reset index if it's named (like 'Date' from yfinance) and convert to list of dicts
        if data.index.name:
            return data.reset_index().to_dict('records')
        return data.to_dict('records')
    elif isinstance(data, pd.Series):
        # Convert Series to a list of dictionaries
        return data.to_frame().reset_index().to_dict('records')
    elif data is None:
        return []
    return data # Assume it's already a list of dicts or an empty list


def store_data_to_db_sqlite(data_list, table_name, connection):
    """
    Stores a clean list of dictionaries into the SQLite DB via Pandas.
    """
    if not data_list:
        print(f"   [INFO] No data to store for table: {table_name}")
        return

    try:
        df = pd.DataFrame(data_list)
        # Write the DataFrame to the specified table
        df.to_sql(table_name, connection, if_exists='append', index=False)
        print(f"   [SUCCESS] Stored {len(df)} records in table: {table_name}")
        
    except Exception as e:
        print(f"   [ERROR] Failed to store data to DB for table {table_name}: {e}")


def run_data_pipeline():
    """
    Executes the entire data collection (scraping) and storage pipeline.
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"--- Data Pipeline Started at {current_time} ---")
    
    all_scraped_data = {} 
    
    # 1. SCRAPE & COLLECT DATA
    print("\n--- 1. Starting Scraping Jobs ---")
    
    # RELIANCE
    try:
        print("1/5. Scraping Reliance Stock...")
        # *** MODIFICATION 2: Using the new START_DATE variable ***
        data = scrape_reliance_data(start_date=START_DATE, end_date=TODAY)
        all_scraped_data['reliance'] = standardize_data_format(data)
        print(f"  [SUCCESS] Collected {len(all_scraped_data['reliance'])} Reliance records.")
    except Exception as e:
        print(f"  [ERROR] Reliance scraping failed: {e}")
        all_scraped_data['reliance'] = []

    # GOLD
    try:
        print("2/5. Scraping Gold...")
        # *** MODIFICATION 2: Using the new START_DATE variable ***
        data = scrape_gold_data(start_date=START_DATE, end_date=TODAY)
        all_scraped_data['gold'] = standardize_data_format(data)
        print(f"  [SUCCESS] Collected {len(all_scraped_data['gold'])} Gold records.")
    except Exception as e:
        print(f"  [ERROR] Gold scraping failed: {e}")
        all_scraped_data['gold'] = []

    # PETROL
    try:
        print("3/5. Scraping Petrol...")
        # *** MODIFICATION 2: Using the new START_DATE variable ***
        data = scrape_petrol_data(start_date=START_DATE, end_date=TODAY)
        all_scraped_data['petrol'] = standardize_data_format(data)
        print(f"  [SUCCESS] Collected {len(all_scraped_data['petrol'])} Petrol records.")
    except Exception as e:
        print(f"  [ERROR] Petrol scraping failed: {e}")
        all_scraped_data['petrol'] = []
    
    # FOREX
    try:
        print("4/5. Scraping Forex...")
        # *** MODIFICATION 2: Using the new START_DATE variable ***
        data = scrape_forex(start_date=START_DATE, end_date=TODAY)
        all_scraped_data['forex'] = standardize_data_format(data)
        print(f"  [SUCCESS] Collected {len(all_scraped_data['forex'])} Forex records.")
    except Exception as e:
        print(f"  [ERROR] Forex scraping failed: {e}")
        all_scraped_data['forex'] = []
        
    # NEWS (No change here, as news usually only fetches recent articles)
    try:
        print("5/5. Scraping Latest News...")
        data = scrape_news_data(api_key=NEWS_API_KEY)
        all_scraped_data['news'] = standardize_data_format(data)
        print(f"  [SUCCESS] Collected {len(all_scraped_data['news'])} News items.")
    except Exception as e:
        # NOTE: This error often happens due to the placeholder API key.
        print(f"  [ERROR] News scraping failed: {e}")
        all_scraped_data['news'] = []

    print("\n--- Scraping and Collection Complete. ---")
    
    
    # 2. STORE DATA TO DATABASE
    
    print("\n--- 2. Storing Data to Database (SQLite) ---")
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            print(f"Database connection established (File: {DATABASE_FILE}).")
            
            # Store each dataset 
            store_data_to_db_sqlite(all_scraped_data['reliance'], table_name="reliance_stocks", connection=conn)
            store_data_to_db_sqlite(all_scraped_data['gold'], table_name="gold_prices", connection=conn)
            store_data_to_db_sqlite(all_scraped_data['petrol'], table_name="petrol_prices", connection=conn)
            store_data_to_db_sqlite(all_scraped_data['forex'], table_name="forex_rates", connection=conn)
            store_data_to_db_sqlite(all_scraped_data['news'], table_name="market_news", connection=conn)
            
    except Exception as e:
        # If the DB connection itself fails (e.g., file permissions), it's caught here.
        print(f"\n[CRITICAL ERROR] Failed to connect to or process data with the Database: {e}")
        
    print("\n--- Data Pipeline Finished. All tasks completed. ---")


if __name__ == "__main__":
    run_data_pipeline()
   
