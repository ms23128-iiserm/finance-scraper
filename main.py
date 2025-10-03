import sqlite3
import datetime

# Import scraper functions
from scraper_reliance import scrape_reliance_data
from scraper_gold import scrape_gold_data
from scraper_petrol import scrape_petrol_data
from scraper_forex import scrape_forex
from scraper_news import scrape_news_data

DB_FILE = "market_data.db"

# -------------------- Database --------------------
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
            eur_inr REAL,
            news TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_data(data):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO market_data (timestamp, reliance, gold, petrol, usd_inr, eur_inr, news)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        data["reliance"],
        data["gold"],
        data["petrol"],
        data["forex"]["USDINR"],
        data["forex"]["EURINR"],
        "; ".join(data["news"])
    ))
    conn.commit()
    conn.close()

# -------------------- Main --------------------
def main():
    # Define date range (last 5 years)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5*365)

    # Your news API key
    NEWS_API_KEY = "eab586f731354326a3c2b38a2833be78"  #

    # Scrape all data
    data = {
        "reliance": scrape_reliance_data(start_date, end_date),
        "gold": scrape_gold_data(start_date, end_date),
        "petrol": scrape_petrol_data(start_date, end_date),
        "forex": scrape_forex(start_date, end_date),
        "news": scrape_news_data(api_key =NEWS_API_KEY)
    }

    # Store in DB
    store_data(data)
    print("✅ Data stored successfully:", data)

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    init_db()
    main()
