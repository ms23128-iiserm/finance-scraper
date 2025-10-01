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
    