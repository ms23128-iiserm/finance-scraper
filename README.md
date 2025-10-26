Financial Forecasting Project: Reliance Stock Price

This repository contains a forecasting model to predict the next-day price movement (delta) of Reliance (RELIANCE.NS) stock using an LSTM model.

Project Overview: Reliance Stock Price Forecast

* Model:   LSTM (Long Short-Term Memory) neural network.
* Data:   Trained on 5+ years of daily data, including market prices, commodities (gold, oil), and news sentiment.
* Method:  This model predicts the *price change (delta)*, not the absolute price. This is a more advanced method to avoid the "Lag-1 Trap" (where a model simply predicts today's price as tomorrow's price).

Honest Model Performance (on Test Set)

1.R-squared (R²):** 0.9743
2.MAE (Mean Absolute Error): 11.18 rupees
    (This means the model's average prediction error on the daily price change was ~11 rupees)

 How to Run

1. Setup

```bash
# Clone this repository
git clone <your-repo-url>
cd finance-scraper

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all required libraries
pip install -r requirements.txt

You should have all these files in your repository
scraper_petrol.py

scraper_gold.py

scraper_forex.py

scraper_reliance.py

2. Run Reliance Project
Step 1: Scrape All Data
(This runs all your scrapers and saves the data to the database)

python main.py


Step 2: Clean and Merge Data
(This reads from the database and creates data_cleaned.csv)

python data_cleaning.py


Step 3: Engineer Features
(This reads data_cleaned.csv and creates features_engineered_data.csv)

python 2_feature_engineering.py


Step 4: Train the LSTM Model
(This trains the main model and saves lstm_15_day_model.h5)

python 4b_model_train_lstm.py


Step 5: Generate the Final Forecast
(This loads the trained model and predicts the next 7-15 days)

python final_forecast.py


Additional Analysis Scripts

You can run these scripts after Step 4 to analyze the model's performance:

To see the performance metrics (R², MAE, RMSE):

python plot_test_result.py


To see the 1-year performance graph:

python plot_2.py


To run the alternate XGBoost model (our first test):

python prediction_xgboost.p
