# scraper_petrol.py
# This script scrapes 5 years of historical data for Petrol (Crude Oil).
import yfinance as yf
import pandas as pd
import datetime

def scrape_petrol_data(start_date, end_date):
    """
    Scrapes historical 'Close' price data for Crude Oil (CL=F) 
    for a given date range.
    """
    # This is the official Yahoo Finance ticker for Crude Oil
    OIL_TICKER = 'CL=F'
    
    print(f"Scraping data for Petrol/Oil ({OIL_TICKER})...")
    try:
        # Download the historical data from Yahoo Finance
        data = yf.download(OIL_TICKER, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"❌ No data found for ticker {OIL_TICKER}.")
            return pd.DataFrame()

        # We only need the 'Close' price. We select it and rename the column.
        petrol_price = data[['Close']].rename(columns={'Close': 'oil_price'})
        
        print("✅ Petrol data scraped successfully.")
        return petrol_price

    except Exception as e:
        print(f"❌ An error occurred while scraping data: {e}")
        return pd.DataFrame()
    