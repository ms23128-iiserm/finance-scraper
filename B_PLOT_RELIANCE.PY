# plot_reliance.py
import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Local import (same folder)
from scraper_reliance import scrape_reliance_data, save_to_csv

OUT_CSV = Path("reliance_data.csv")
OUT_PNG = Path("reliance_price_5y.png")

def plot_reliance_last_5y():
    # 5-year window
    end = datetime.date.today()
    start = end - datetime.timedelta(days=5 * 365)

    # Fetch & store CSV
    df = scrape_reliance_data(start, end)
    save_to_csv(df, OUT_CSV)

    if df.empty:
        print("❌ Nothing to plot.")
        return

    # Ensure datetime for plotting
    df["date"] = pd.to_datetime(df["date"])

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["reliance_close"], linewidth=1.5)
    plt.title("Reliance Industries – Adjusted Close (Last 5 Years)")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")

    # Nice date formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,4,7,10)))
    plt.grid(True, which="both", axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=150)
    print(f"🖼️ Saved plot -> {OUT_PNG.resolve()}")
    # If you want to show interactively:
    # plt.show()

if __name__ == "__main__":
    plot_reliance_last_5y()
