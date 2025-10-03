from datetime import datetime
import pandas as pd # Required for Timedelta functionality

# --- CONFIGURATION PARAMETERS ---
# Define the date range for your time-series scrapers
TODAY = datetime.now().strftime('%Y-%m-%d')
SEVEN_DAYS_AGO = (datetime.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')

# Replace this placeholder with your actual key
NEWS_API_KEY = "eab586f731354326a3c2b38a2833be78" 
# --------------------------------

# --- Import all your specific scraping functions ---
from scraper_reliance import scrape_reliance_data
from scraper_gold import scrape_gold_data
from scraper_petrol import scrape_petrol_data
from scraper_forex import scrape_forex
from scraper_news import scrape_news_data


def run_data_pipeline():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"--- Data Pipeline Started at {current_time} ---")
    
    all_scraped_data = {} 
    print("\n--- Starting Scraping Jobs ---")
    
    # 1. RELIANCE (Now passing start_date and end_date)
    try:
        print("1/5. Scraping Reliance Stock...")
        data = scrape_reliance_data(start_date=SEVEN_DAYS_AGO, end_date=TODAY) # <--- FIXED
        all_scraped_data['reliance'] = data
        print(f"  [SUCCESS] Collected {len(data)} Reliance records.")
    except Exception as e:
        print(f"  [ERROR] Reliance scraping failed: {e}")
        all_scraped_data['reliance'] = []

    # 2. GOLD (Now passing start_date and end_date)
    try:
        print("2/5. Scraping Gold...")
        data = scrape_gold_data(start_date=SEVEN_DAYS_AGO, end_date=TODAY) # <--- FIXED
        all_scraped_data['gold'] = data
        print(f"  [SUCCESS] Collected {len(data)} Gold records.")
    except Exception as e:
        print(f"  [ERROR] Gold scraping failed: {e}")
        all_scraped_data['gold'] = []

    # 3. PETROL (Now passing start_date and end_date)
    try:
        print("3/5. Scraping Petrol...")
        data = scrape_petrol_data(start_date=SEVEN_DAYS_AGO, end_date=TODAY) # <--- FIXED
        all_scraped_data['petrol'] = data
        print(f"  [SUCCESS] Collected {len(data)} Petrol records.")
    except Exception as e:
        print(f"  [ERROR] Petrol scraping failed: {e}")
        all_scraped_data['petrol'] = []
    
    # 4. FOREX (Now passing start_date and end_date)
    try:
        print("4/5. Scraping Forex...")
        data = scrape_forex(start_date=SEVEN_DAYS_AGO, end_date=TODAY) # <--- FIXED
        all_scraped_data['forex'] = data
        print(f"  [SUCCESS] Collected {len(data)} Forex records.")
    except Exception as e:
        print(f"  [ERROR] Forex scraping failed: {e}")
        all_scraped_data['forex'] = []
        
    # 5. NEWS (Now passing api_key)
    try:
        print("5/5. Scraping Latest News...")
        data = scrape_news_data(api_key=NEWS_API_KEY) # <--- FIXED
        all_scraped_data['news'] = data
        print(f"  [SUCCESS] Collected {len(data)} News items.")
    except Exception as e:
        print(f"  [ERROR] News scraping failed: {e}")
        all_scraped_data['news'] = []

    print("\n--- Scraping and Collection Complete. ---")
    
    return all_scraped_data


if __name__ == "__main__":
    combined_market_data = run_data_pipeline()
    
    # If successful, 'combined_market_data' now holds all your scraped lists/DataFrames.