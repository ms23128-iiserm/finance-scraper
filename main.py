import sqlite3
import datetime

# Import scraper functions
from scraper_reliance import scrape_reliance_data
from scraper_gold import scrape_gold_data
from scraper_petrol import scrape_petrol_data
from scraper_forex import get_forex_rates
from scraper_news import scrape_news_data

DB_FILE = "market_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            reliance REAL,
            gold REAL,
            petrol REAL,
            usd_inr REAL,
            eur_usd REAL,
            news TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_data(data):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO market_data (timestamp, reliance, gold, petrol, usd_inr, eur_usd, news)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        data["reliance"],
        data["gold"],
        data["petrol"],
        data["forex"]["USDINR"],
        data["forex"]["EURUSD"],
        "; ".join(data["news"])
    ))
    conn.commit()
    conn.close()

def main():
    data = {
        "reliance": scrape_reliance_data(),
        "gold": scrape_gold_data(),
        "petrol": scrape_petrol_data(),
        "forex": get_forex_rates(),
        "news": scrape_news_data()
    }
    store_data(data)
    print("✅ Data stored successfully:", data)

if __name__ == "__main__":
    init_db()
    main()
