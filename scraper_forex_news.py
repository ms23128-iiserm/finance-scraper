import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# --- Reliance Price ---
def get_reliance_price():
    try:
        url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=RELIANCE.BO"
        data = requests.get(url, timeout=10).json()
        price = data["quoteResponse"]["result"][0]["regularMarketPrice"]
        return price
    except Exception as e:
        print("Error fetching Reliance price:", e)
        return None

# --- BBC News (RSS via requests + XML parsing) ---
def get_bbc_headlines(limit=5):
    try:
        url = "http://feeds.bbci.co.uk/news/rss.xml"
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)

        headlines = []
        for item in root.findall("./channel/item")[:limit]:
            title = item.find("title").text or ""
            headlines.append(title)
        return headlines
    except Exception as e:
        print("Error fetching BBC News:", e)
        return []

# --- Currency Rates (USD & EUR → INR) ---
def get_currency_rates():
    try:
        url = "https://api.frankfurter.app/latest?from=USD&to=INR,EUR"
        data = requests.get(url, timeout=10).json()
        usd_inr = data["rates"]["INR"]      # 1 USD = ? INR
        usd_eur = data["rates"]["EUR"]      # 1 USD = ? EUR
        eur_inr = usd_inr / usd_eur         # 1 EUR = ? INR
        return usd_inr, eur_inr
    except Exception as e:
        print("Error fetching currency rates:", e)
        return None, None

# --- Main Run ---
if __name__ == "_main_":
    print("Fetching latest data...\n")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reliance_price = get_reliance_price()
    headlines = get_bbc_headlines()
    usd_inr, eur_inr = get_currency_rates()

    print("=== Latest Update @", timestamp, "===")
    if reliance_price:
        print(f"Reliance Price: ₹{reliance_price}")
    if usd_inr and eur_inr:
        print(f"USD → INR: {usd_inr:.2f}")
        print(f"EUR → INR: {eur_inr:.2f}")
    if headlines:
        print("\nTop BBC Headlines:")
        for i, h in enumerate(headlines, 1):
            print(f"{i}. {h}")
