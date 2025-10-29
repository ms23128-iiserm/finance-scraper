# B_core_reliance_external.py
# Correlation between Reliance, Gold, Crude, USD/INR with robust loading + plotting.

from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ------------------ Config ------------------
OUT_DIR = Path.cwd() / "artifacts_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If you already saved CSVs with the earlier scrapers, we’ll use them:
CSV_RELIANCE = Path("artifacts_reliance/reliance_data_5y.csv")  # columns: date, reliance_close
CSV_GOLD     = Path("artifacts_gold/gold_data.csv")              # columns: date, Gold_Price_USD_oz
CSV_CRUDE    = Path("artifacts_petrol/petrol_data.csv")          # columns: date, Oil_Price_USD_bbl
CSV_FOREX    = Path("artifacts_forex/forex_data.csv")            # columns: date, USDINR=X, EURINR=X

YEARS = 5
ROLL_WIN = 60  # trading days

# Yahoo fallback tickers
TICKERS = {
    "reliance": "RELIANCE.NS",
    "gold":     "GC=F",
    "crude":    "CL=F",
    "usdinr":   "INR=X",  # will retry USDINR=X if needed
}

# ------------------ Helpers ------------------
def logret(s: pd.Series) -> pd.Series:
    return np.log(s).diff()

def _series_from_csv(path: Path, date_col: str, value_col: str, new_name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[date_col])
    s = df[[date_col, value_col]].dropna().set_index(date_col)[value_col].sort_index()
    s.name = new_name                              # ✅ correct way to set series name
    return s

def _fetch_series_yf(ticker: str, start: datetime.date, end: datetime.date, label: str) -> pd.Series:
    """Download close prices; try date-range first, then period=5y fallback."""
    if not HAVE_YF:
        raise RuntimeError("yfinance not available and local CSV missing.")
    # Attempt 1: date-range
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        # Attempt 2: fallback by period
        df = yf.download(ticker, period=f"{YEARS}y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        # Special retry for USD/INR if INR=X failed
        if ticker == "INR=X":
            df = yf.download("USDINR=X", period=f"{YEARS}y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for ticker {ticker}")
    s = df["Close"].copy()
    s.name = label                                   # ✅ don’t call rename(name)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

def load_or_download_prices() -> pd.DataFrame:
    """
    Return aligned price levels with columns: reliance, gold, crude, usdinr.
    Prefer local CSVs; fill missing via Yahoo.
    """
    pieces = {}

    # 1) Try local CSVs
    if CSV_RELIANCE.exists():
        pieces["reliance"] = _series_from_csv(CSV_RELIANCE, "date", "reliance_close", "reliance")
    if CSV_GOLD.exists():
        # gold could be Gold_Price_USD_oz or "Gold Price (USD/oz)" depending on your earlier script
        cols = pd.read_csv(CSV_GOLD, nrows=0).columns.tolist()
        gold_col = "Gold_Price_USD_oz" if "Gold_Price_USD_oz" in cols else "Gold Price (USD/oz)"
        pieces["gold"] = _series_from_csv(CSV_GOLD, "date", gold_col, "gold")
    if CSV_CRUDE.exists():
        cols = pd.read_csv(CSV_CRUDE, nrows=0).columns.tolist()
        crude_col = "Oil_Price_USD_bbl" if "Oil_Price_USD_bbl" in cols else ("oil_price" if "oil_price" in cols else "Close")
        pieces["crude"] = _series_from_csv(CSV_CRUDE, "date", crude_col, "crude")
    if CSV_FOREX.exists():
        pieces["usdinr"] = _series_from_csv(CSV_FOREX, "date", "USDINR=X", "usdinr")

    # 2) Download any missing series
    need = [k for k in ["reliance", "gold", "crude", "usdinr"] if k not in pieces]
    if need:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=YEARS * 365)
        print(f"[INFO] Downloading missing: {need}  (from {start} to {end})")
        for name in need:
            ticker = TICKERS[name]
            pieces[name] = _fetch_series_yf(ticker, start, end, name)

    # 3) Align on common dates (inner join)
    df = pd.concat([pieces[k] for k in ["reliance", "gold", "crude", "usdinr"]], axis=1)
    df = df.dropna(how="any").sort_index()
    return df

# ------------------ Main ------------------
def main():
    levels = load_or_download_prices()
    print("[INFO] aligned rows:", len(levels))
    levels.to_csv(OUT_DIR / "levels_aligned.csv")
    print("[SAVE]", (OUT_DIR / "levels_aligned.csv").resolve())

    returns = levels.apply(logret).dropna(how="any")
    returns.to_csv(OUT_DIR / "returns_log_aligned.csv")
    print("[SAVE]", (OUT_DIR / "returns_log_aligned.csv").resolve())

    corr_levels = levels.corr()
    corr_returns = returns.corr()
    corr_levels.to_csv(OUT_DIR / "corr_levels.csv")
    corr_returns.to_csv(OUT_DIR / "corr_returns.csv")
    print("[SAVE]", (OUT_DIR / "corr_returns.csv").resolve())

    # ---- Plot 1: Heatmap (returns) ----
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr_returns.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_returns.columns)))
    ax.set_yticks(range(len(corr_returns.index)))
    ax.set_xticklabels(corr_returns.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_returns.index)
    for i in range(corr_returns.shape[0]):
        for j in range(corr_returns.shape[1]):
            ax.text(j, i, f"{corr_returns.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Correlation (Log Returns)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    heatmap_path = OUT_DIR / "heatmap_corr_returns.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print("[SAVE]", heatmap_path.resolve())

    # ---- Plot 2: Rolling correlations (Reliance vs others) ----
    win = ROLL_WIN
    rel = returns["reliance"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["gold", "crude", "usdinr"]:
        roll = rel.rolling(win).corr(returns[col])
        ax.plot(roll.index, roll.values, label=f"reliance vs {col} ({win}d)")
    ax.set_title(f"Rolling {win}-Day Correlation with Reliance (Log Returns)")
    ax.set_xlabel("Date"); ax.set_ylabel("Correlation")
    ax.legend()
    plt.tight_layout()
    roll_path = OUT_DIR / "rolling_corr.png"
    plt.savefig(roll_path, dpi=150)
    plt.close()
    print("[SAVE]", roll_path.resolve())

if __name__ == "__main__":
    main()
