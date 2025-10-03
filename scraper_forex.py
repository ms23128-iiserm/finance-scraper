# File: scraper_forex_nocsv.py
import yfinance as yf
import pandas as pd
import datetime

def scrape_forex(ticker: str, years: int = 5):
    """
    Scrape historical forex data for a given ticker over the last `years` years
    and return it as a pandas DataFrame (no CSV saved).
    
    Args:
        ticker (str): Forex ticker symbol (e.g., "INR=X" for USD/INR).
        years (int): Number of years of historical data to fetch (default=5).
    
    Returns:
        pd.DataFrame: The historical adjusted close prices.
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=years*365)
    
    print(f"Fetching data for {ticker} from {start} to {end}...")
    
    data = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False)
    
    # Use 'Adj Close' if available, otherwise fallback to 'Close'
    if 'Adj Close' in data.columns:
        df = data['Adj Close']
    else:
        df = data['Close']
    
    print(f"Data fetched for {ticker}, {len(df)} rows.")
    return df

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    tickers = ["INR=X", "EURINR=X"]  # USD/INR and EUR/INR
    
    usd_inr = scrape_forex("INR=X", years=5)
    eur_inr = scrape_forex("EURINR=X", years=5)
    
    print(usd_inr.head())
    print(eur_inr.head())

