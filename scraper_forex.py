import yfinance as yf
import pandas as pd
import datetime

# -----------------------------
# Forex data using yfinance
# -----------------------------

# Define tickers (Yahoo Finance symbols for forex pairs)
# INR=X means USD/INR, EURUSD=X means EUR/USD
tickers = ["INR=X", "EURUSD=X"]

# Fetch last 5 years of daily data
end = datetime.date.today()
start = end - datetime.timedelta(days=5*365)

# Use Close prices (forex does not have 'Adj Close')
data = yf.download(tickers, start=start, end=end, interval="1d")['Close']

# Rename columns for clarity
data = data.rename(columns={
    "INR=X": "USD_INR",
    "EURUSD=X": "EUR_USD"
})

# Reset index to have 'date' as a column
df_currency = data.reset_index()

# Create new column: EUR to INR
df_currency["EUR_INR"] = df_currency["EUR_USD"] * df_currency["USD_INR"]

# Display first 10 rows
print(df_currency.head(1297))
print(f"Total rows: {len(df_currency)}")

