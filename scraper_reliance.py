# scraper_reliance.py
import yfinance as yf
import pandas as pd
import datetime

def scrape_reliance_data(start_date, end_date):
    """
    Scrapes 5 years of historical stock data for Reliance Industries (RELIANCE.NS).
    """
    STOCK_TICKER = 'RELIANCE.NS'
    print(f"Scraping historical data for {STOCK_TICKER}...")
    
    try:
        # Download historical data from Yahoo Finance using the yfinance library
        data = yf.download(STOCK_TICKER, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"❌ No data found for ticker {STOCK_TICKER}.")
            return pd.DataFrame()
            
        print("✅ Reliance historical data scraped successfully.")
        
        # We only need the 'Close' price for our project
        reliance_price = data[['Close']].rename(columns={'Close': 'stock_price'})
        return reliance_price

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return pd.DataFrame()

# This block allows you to test this script directly
if __name__ == '__main__':
    # Define the 5-year date range
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=1825)
    
    print(f"--- Running Test for scraper_reliance.py ---")
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Call your function
    reliance_df = scrape_reliance_data(start_date, end_date)
    
    # Print the first and last 5 rows to confirm it works
    if not reliance_df.empty:
        print("\n--- Sample of Scraped Historical Data (Oldest) ---")
        print(reliance_df.head())
        print("\n--- Sample of Scraped Historical Data (Most Recent) ---")
        print(reliance_df.tail())
        print("\nTest finished successfully! 🚀")