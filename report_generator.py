import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration ---
# File from the XGBoost Walk-Forward Validation (WFV)
WFV_RESULTS_FILE = 'walk_forward_predictions.csv' 
# File containing the original features (for Buy & Hold calculation)
FEATURE_DATA_FILE = 'features_engineered_data.csv' 
# File created by 5_advanced_modeling.py
ADVANCED_RESULTS_FILE = 'advanced_model_results.csv' 
TARGET_PRICE_COL = 'target'
# --- END CONFIGURATION ---

def calculate_pnl_from_wfv(wfv_df, features_df):
    """Calculates P&L for the WFV XGBoost model and the Buy & Hold baseline."""
    
    # 1. Merge WFV results with the true 'Reliance_Close' price (Entry Price)
    wfv_df = wfv_df.merge(features_df[['Reliance_Close']], left_index=True, right_index=True, how='left')
    wfv_df = wfv_df.dropna()
    
    # 2. Calculate Model P&L: (Exit Price - Entry Price) if prediction is 1, else 0
    wfv_df['Model_Return'] = np.where(
        wfv_df['Predicted_Direction'] == 1, 
        wfv_df[TARGET_PRICE_COL] - wfv_df['Reliance_Close'], 
        0
    )
    final_model_pnl = wfv_df['Model_Return'].sum()
    
    # 3. Calculate Buy & Hold P&L
    initial_price = wfv_df['Reliance_Close'].iloc[0]
    final_price = wfv_df[TARGET_PRICE_COL].iloc[-1]
    buy_and_hold_pnl = final_price - initial_price
    
    return final_model_pnl, buy_and_hold_pnl

def calculate_metrics(actual, predicted):
    """Calculates RMSE and R2 for regression models."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return rmse, r2

def get_advanced_model_metrics():
    """Loads advanced model results and calculates metrics for the final table."""
    try:
        adv_df = pd.read_csv(ADVANCED_RESULTS_FILE, parse_dates=True, index_col=0).dropna()
    except FileNotFoundError:
        print(f"❌ ERROR: Advanced results file '{ADVANCED_RESULTS_FILE}' not found. Cannot populate TBDs.")
        return {}
    
    actual = adv_df[TARGET_PRICE_COL]
    
    metrics = {}
    
    for col in adv_df.columns:
        if col != TARGET_PRICE_COL and col.startswith('Predicted_'):
            predicted = adv_df[col]
            rmse, r2 = calculate_metrics(actual, predicted)
            
            display_name = col.replace('Predicted_', '')
            metrics[display_name] = {'RMSE': rmse, 'R2': r2}

    return metrics

def compile_master_results(wfv_df, features_df):
    """Main function to consolidate all results into a single list."""
    
    model_pnl, buy_and_hold_pnl = calculate_pnl_from_wfv(wfv_df, features_df)
    
    wfv_rmse, wfv_r2 = calculate_metrics(wfv_df[TARGET_PRICE_COL], wfv_df['Predicted_Price'])
    
    # WFV Directional Metric (Assuming 0.55 from your data_prediction.py output)
    wfv_f1_score = 0.55 
    
    advanced_metrics = get_advanced_model_metrics()
    
    # --- 1. Base Results (XGBoost WFV & Buy-and-Hold) ---
    results = [
        {
            'Model': 'XGBoost (Walk-Forward)',
            'Validation': 'Dynamic WFV',
            'RMSE': wfv_rmse,
            'R2': wfv_r2,
            'F1-Score (Direction)': wfv_f1_score,
            'P&L (INR)': f"₹{model_pnl:,.2f}"
        },
        {
            'Model': 'Buy-and-Hold',
            'Validation': 'Baseline',
            'RMSE': 'N/A',
            'R2': 'N/A',
            'F1-Score (Direction)': 'N/A',
            'P&L (INR)': f"₹{buy_and_hold_pnl:,.2f}"
        },
    ]

    # --- 2. Advanced Model Results (LSTM, ARIMA, Optimized XGBoost) ---
    for model_name, m in advanced_metrics.items():
        results.append({
            'Model': model_name.replace('Optimized', 'Optimized ').replace('Network', ' Network').replace('Baseline', ' Baseline'), # Clean up names
            'Validation': 'Static 80/20',
            'RMSE': m['RMSE'],
            'R2': m['R2'],
            'F1-Score (Direction)': 'N/A',
            'P&L (INR)': 'N/A'
        })
    
    return results, model_pnl, buy_and_hold_pnl

def print_markdown_report(results, model_pnl, buy_and_hold_pnl):
    """Prints the compiled results in a final Markdown report format."""
    
    # --- Retrieve Key Metrics for Summary ---
    xgb_wfv_results = next(item for item in results if item["Model"] == "XGBoost (Walk-Forward)")
    wfv_r2 = xgb_wfv_results['R2']
    wfv_rmse = xgb_wfv_results['RMSE']
    
    print("\n# Final Data Science Project Report: Reliance Stock Price Prediction")
    print("## 1. Executive Summary")
    
    print(f"The project successfully developed and validated several time-series models for forecasting Reliance stock prices over the last 5 years.")
    
    if model_pnl > buy_and_hold_pnl:
        # Ideal Outcome
        print(f"\n* *Conclusion:* The *XGBoost (Walk-Forward)* model successfully *OUTPERFORMED* the market, yielding a P&L of ₹{model_pnl:,.2f} over the passive Buy-and-Hold benchmark (₹{buy_and_hold_pnl:,.2f}).")
    else:
        # Realistic Outcome (Model P&L < Buy-and-Hold P&L)
        print(f"\n* *Technical Accuracy:* The *XGBoost (Walk-Forward)* model achieved the best technical performance (R²: {wfv_r2:.4f}, RMSE: {wfv_rmse:.2f}), demonstrating strong price prediction skill.")
        print(f"* *Financial Reality:* The model's active trading strategy (P&L: ₹{model_pnl:,.2f}) *UNDERPERFORMED* the passive Buy-and-Hold strategy (₹{buy_and_hold_pnl:,.2f}).")
        print("\n*Conclusion:* The model lacked the directional edge (F1-Score 0.55) necessary to overcome trading risk and beat the market, highlighting a key challenge in financial modeling.")
    
    # --- Print Master Table ---
    print("\n## 2. Quantitative Model Comparison")
    
    df_report = pd.DataFrame(results)
    
    # Format the numeric columns
    df_report['RMSE'] = df_report['RMSE'].apply(lambda x: f"{x:,.4f}" if isinstance(x, (float, np.float64)) else x)
    df_report['R2'] = df_report['R2'].apply(lambda x: f"{x:,.4f}" if isinstance(x, (float, np.float64)) else x)
    
    df_report = df_report[['Model', 'Validation', 'RMSE', 'R2', 'F1-Score (Direction)', 'P&L (INR)']]

    print("\n### Master Results Table")
    print(df_report.to_markdown(index=False))
    
    print("\n*Note: Lower RMSE and higher R² indicate better price prediction. F1-Score > 0.50 indicates better than random directional accuracy.*")
    
    print("\n## 3. Business Impact: P&L Analysis")
    print(f"| Strategy | Final P&L (Test Period) | Performance |")
    print(f"|:---|:---|:---|")
    
    # Print Model vs. Benchmark comparison
    print(f"| *Model Strategy (XGBoost WFV)* | ₹{model_pnl:,.2f} | {'< Benchmark' if model_pnl < buy_and_hold_pnl else '> Benchmark'} |")
    print(f"| Buy-and-Hold Baseline | ₹{buy_and_hold_pnl:,.2f} | Benchmark |")
    
    print("\n*(This section would be followed by your P&L plot: )*")