import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Function to scrape Reliance stock price
def get_reliance_stock_price():
    url = "https://finance.yahoo.com/quote/RELIANCE.NS"  # Reliance Industries (NSE)
    headers = {"User-Agent": "Mozilla/5.0"}  # to avoid blocking
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Yahoo finance keeps the price inside <fin-streamer> tag
    price_tag = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
    
    if price_tag:
        price = price_tag.text
        print("Reliance Stock Price:", price)
        return price
    else:
        print("Could not fetch price.")
        return None

# Save to CSV
def save_to_csv(price):
    df = pd.DataFrame([[datetime.now(), price]], columns=["Time", "Reliance Price"])
    df.to_csv("reliance_stock.csv", mode="a", header=not pd.io.common.file_exists("reliance_stock.csv"), index=False)
    print("Saved to reliance_stock.csv")

if __name__ == "__main__":
    price = get_reliance_stock_price()
    if price:
        save_to_csv(price)
