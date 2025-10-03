# File: scraper_forex_nocsv.py
import yfinance as yf
import pandas as pd
import datetime

import yfinance as yf

def scrape_forex(start_date, end_date):
    tickers = {"USDINR": "INR=X", "EURINR": "EURINR=X"}  # Yahoo Finance symbols

    result = {}
    for key, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Use 'Adj Close' if available, otherwise 'Close'
        if "Adj Close" in df.columns:
            result[key] = df["Adj Close"].iloc[-1]
        else:
            result[key] = df["Close"].iloc[-1]
    return result

