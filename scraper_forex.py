import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def scrape_forex(start_date, end_date):
    """
    Scrapes 5 years of historical forex data:
    USD→INR and EUR→INR using Yahoo Finance.
    Returns a Pandas DataFrame with columns:
    [date, USDINR=X, EURINR=X]
    """
    print("Scraping historical forex data (USD→INR, EUR→INR)...")

    # Define currency pairs
    tickers = ['USDINR=X', 'EURINR=X']
    data_frames = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            df = df[['Close']].rename(columns={'Close': ticker})
            data_frames.append(df)
        except Exception as e:
            print(f"[ERROR] Could not fetch data for {ticker}: {e}")

    # Merge both forex datasets
    if data_frames:
        combined = pd.concat(data_frames, axis=1).dropna()
        combined.reset_index(inplace=True)
        combined.rename(columns={'Date': 'date'}, inplace=True)
        print(f"[SUCCESS] Fetched {len(combined)} forex records.")

        # Save to CSV for backup/logging
        combined.to_csv("forex_data.csv", index=False)
        print("Saved forex data to 'forex_data.csv'")

        return combined

    print("[WARNING] No forex data fetched.")
    return pd.DataFrame(columns=['date', 'USDINR=X', 'EURINR=X'])


# --- Standalone Test Section ---
if __name__ == "__main__":
    # Calculate last 5 years date range
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Run test
    df = scrape_forex(start_date, end_date)
    print("\n--- Last 5 Rows ---")
    print(df.tail())
    print(f"\nTotal records fetched: {len(df)}")
