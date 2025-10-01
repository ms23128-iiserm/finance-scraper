# scraper_news.py
# This script is responsible for scraping news headlines.
import pandas as pd
from newsapi import NewsApiClient
import datetime

def scrape_news_data(api_key):
    """
    Scrapes news headlines for a list of predefined keywords using NewsAPI.
    
    Args:
        api_key (str): The API key for authenticating with NewsAPI.

    Returns:
        pandas.DataFrame: A DataFrame containing the date and headline of each article,
                          or an empty DataFrame if an error occurs.
    """
    print("Scraping news headlines...")
    
    # Check if the API key has been provided to avoid errors
    if not api_key or api_key == 'YOUR_NEWS_API_KEY':
        print("❌ ERROR: News API Key is missing. Cannot fetch news.")
        return pd.DataFrame() # Return an empty DataFrame

    try:
        # Initialize the NewsAPI client with the provided key
        newsapi = NewsApiClient(api_key=api_key)
        
        # Keywords relevant to our project
        keywords = ['reliance', 'ambani', 'jio', 'war', 'election', 'disaster']
        all_articles = []
        
        print(f"Fetching news for keywords: {keywords}")
        for query in keywords:
            # Fetch the top headlines for each keyword
            articles_response = newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=100  # Max articles per request from the free plan
            )
            # Extract the title and publication date for each article
            for article in articles_response['articles']:
                all_articles.append({
                    'date': pd.to_datetime(article['publishedAt']).date(),
                    'headline': article['title']
                })
        
        print(f"✅ News scraping complete. Found {len(all_articles)} articles.")
        # Convert the list of articles into a pandas DataFrame
        return pd.DataFrame(all_articles)

    except Exception as e:
        # Catch any errors during the API request
        print(f"❌ An error occurred while scraping news data: {e}")
        return pd.DataFrame() # Return an empty DataFrame on failure

# This special block allows you to test this script directly
if __name__ == '__main__':
    # --- Configuration for the Test ---
    # IMPORTANT: Paste your real News API Key here to run the test
    TEST_NEWS_API_KEY = ''
    
    print("--- Running Test for News Scraper ---")
    
    # Call the function to get the news data
    news_df = scrape_news_data(TEST_NEWS_API_KEY)
    
    # If the function returned data, print the first 5 rows to check
    if not news_df.empty:
        print("\n--- Sample of Scraped News Headlines ---")
        print(news_df.head())
        print("\nTest finished successfully! 🚀")
    else:
        print("\nTest finished, but no data was fetched. Please check your API key.")
