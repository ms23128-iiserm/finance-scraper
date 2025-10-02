# scraper_gold.py
import yfinance as yf
import pandas as pd
import datetime

def scrape_gold_data(start_date, end_date):
    """
    Scrapes historical 'Close' price data for gold
    for a given date range.
    """
    GOLD_TICKER = 'GC=F'  # Gold Futures
    
    print(f"Scraping data for gold ({GOLD_TICKER}) from {start_date} to {end_date}...")
    data = yf.download(GOLD_TICKER, start=start_date, end=end_date, interval="1d")
    df = data[["Close"]].reset_index()
    df.rename(columns={"Close": "Gold Price (USD/oz)"}, inplace=True)
    return df

if __name__ == "__main__":
    # Define date range (last 5 years)
    end = datetime.date.today()
    start = end - datetime.timedelta(days=5*365)

    # Scrape gold data
    df = scrape_gold_data(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(df.head())  # print first few rows

    # Save to CSV
    df.to_csv("gold_data.csv", index=False)
    print("Data saved to gold_data.csv")
