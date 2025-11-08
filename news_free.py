import requests
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from newspaper import Article
import time

# =============================
# CONFIGURATION
# =============================

API_KEY = "pub_cdf3525735a74828a7e5589d292b9313"  # ✅ Your NewsData.io API key
QUERY = "Reliance OR Jio OR Ambani OR war OR election"
LANGUAGE = "en"
COUNTRY = "in"   # restrict to India
MAX_PAGES = 10   # adjust as needed (each page ~10-20 results)
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# =============================
# STEP 1️⃣  FETCH NEWS ARTICLES
# =============================

base_url = "https://newsdata.io/api/1/news"
all_articles = []

print(f"Fetching news for query: {QUERY}")

for page in tqdm(range(1, MAX_PAGES + 1)):
    params = {
        "apikey": API_KEY,
        "q": QUERY,
        "language": LANGUAGE,
        "country": COUNTRY,
        "from_date": START_DATE,
        "to_date": END_DATE,
        "page": page
    }
    
    res = requests.get(base_url, params=params)
    if res.status_code != 200:
        print(f"Error {res.status_code}: {res.text}")
        break
    
    data = res.json()
    if "results" not in data or not data["results"]:
        print("No more articles found or empty result.")
        break
    
    all_articles.extend(data["results"])
    
    # respect API rate limits
    time.sleep(2)
    
    if not data.get("nextPage"):
        break

df = pd.DataFrame(all_articles)
print(f"\n✅ Collected {len(df)} articles from NewsData.io")

# Check if dataframe is empty before continuing
if df.empty:
    print("⚠️ No data fetched from NewsData API. Please check your API key, query, or date range.")
    exit()

# Keep only available columns safely
available_cols = [c for c in ["title", "description", "link", "pubDate", "source_id"] if c in df.columns]
df = df[available_cols].drop_duplicates()

# =============================
# STEP 2️⃣  EXTRACT FULL TEXT (Optional)
# =============================

def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

if "link" in df.columns:
    tqdm.pandas(desc="Extracting full article text")
    df["text"] = df["link"].progress_apply(get_article_text)
else:
    df["text"] = None

df["combined_text"] = (
    df.get("title", "").fillna('') + " " +
    df.get("description", "").fillna('') + " " +
    df.get("text", "").fillna('')
)

# =============================
# STEP 3️⃣  RUN FINBERT SENTIMENT ANALYSIS
# =============================

print("\nLoading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_sentiment(text):
    try:
        if not text or len(text.strip()) == 0:
            return "neutral"
        result = finbert(text[:512])[0]   # limit to 512 tokens
        return result["label"]
    except:
        return "neutral"

tqdm.pandas(desc="Running FinBERT")
df["sentiment"] = df["combined_text"].progress_apply(get_sentiment)

# =============================
# STEP 4️⃣  SAVE RESULTS
# =============================

df.to_csv("reliance_news_finbert.csv", index=False)
print("\n✅ Saved results to reliance_news_finbert.csv")

# Quick sentiment summary
print("\nSentiment Summary:")
print(df["sentiment"].value_counts())
