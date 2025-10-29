import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_anomalies(csv_file, column_name, threshold=2.0):
    """
    Finds and plots anomalies in a time series using the Z-score method.
    
    1. Calculates daily percentage returns.
    2. Calculates the Z-score for each day's return.
    3. Identifies "outliers" (anomalies) above a certain threshold.
    4. Prints the dates of these outliers.
    5. Plots the Z-scores and highlights the outliers.
    """
    print(f"--- Starting Anomaly Detection for '{column_name}' ---")
    
    data_path = Path(csv_file)
    if not data_path.exists():
        print(f"❌ ERROR: Cannot find '{csv_file}'.")
        print("Please run 'run_correlation_analysis.py' first to generate it.")
        return

    # Load the merged and cleaned data
    try:
        # --- FIX: Load by column index (0) instead of column name 'Date' ---
        # This is more robust if the saved CSV's index column has no name.
        df = pd.read_csv(data_path, parse_dates=[0], index_col=0)
        df.index.name = 'Date' # Assign a name to the index for clarity
        
        if column_name not in df.columns:
            print(f"❌ ERROR: Column '{column_name}' not found in '{csv_file}'.")
            return
    except Exception as e:
        print(f"❌ ERROR: Could not load data. {e}")
        return

    # 1. Calculate daily percentage returns
    # We drop the first day (NaN) as it has no prior day to compare to
    df['daily_return'] = df[column_name].pct_change().dropna()
    
    # 2. Calculate Z-score
    # Get the mean and standard deviation of the daily returns
    mean_return = df['daily_return'].mean()
    std_return = df['daily_return'].std()
    
    df['z_score'] = (df['daily_return'] - mean_return) / std_return
    
    # 3. Identify outliers
    df_ups = df[df['z_score'] > threshold]
    df_downs = df[df['z_score'] < -threshold]
    
    print("\n--- Analysis Complete ---")
    print(f"Average daily change (mean): {mean_return: .4f}%")
    print(f"Normal change range (std dev): {std_return: .4f}%")
    print(f"Found {len(df_ups)} 'sudden up' events (>{threshold} sigma).")
    print(f"Found {len(df_downs)} 'sudden down' events (< -{threshold} sigma).")
    
    # 4. Print the dates of these events
    print("\n--- 'Sudden Up' Dates ---")
    for date, row in df_ups.iterrows():
        print(f"{date.date()}: {row['daily_return']: .2f}% change (Z-score: {row['z_score']: .2f})")
        
    print("\n--- 'Sudden Down' Dates ---")
    for date, row in df_downs.iterrows():
        print(f"{date.date()}: {row['daily_return']: .2f}% change (Z-score: {row['z_score']: .2f})")

    # 5. Plot the graph
    print("\nGenerating anomaly plot...")
    plt.figure(figsize=(16, 6))
    
    # Plot all Z-scores
    plt.plot(df.index, df['z_score'], label=f'{column_name} Z-Score', color='goldenrod', zorder=1)
    
    # Plot the outliers
    plt.scatter(
        df_ups.index, 
        df_ups['z_score'], 
        color='green', 
        marker='^', 
        s=100, # size
        label=f'Sudden Up (>{threshold} sigma)',
        zorder=2
    )
    plt.scatter(
        df_downs.index, 
        df_downs['z_score'], 
        color='red', 
        marker='v', 
        s=100, # size
        label=f'Sudden Down (< -{threshold} sigma)',
        zorder=2
    )
    
    # Style the plot like your example
    plt.title(f"Z-Return Anomalies for {column_name} (Last 5 Years)", fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Z-Return (Standard Deviations)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5) # Add a line for the average
    plt.axhline(threshold, color='red', linestyle=':', alpha=0.5)
    plt.axhline(-threshold, color='red', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    output_path = f"{column_name.lower()}_anomalies_plot.png"
    plt.savefig(output_path)
    
    print(f"✅ Successfully saved plot to '{output_path}'")


if __name__ == "__main__":
    # Ensure 'all_data_merged_clean.csv' exists by running run_correlation_analysis.py first
    
    # We will analyze the 'reliance_close' column
    # We set the 'exaggeration' threshold to 2.0 (same as your example)
    find_anomalies(
        csv_file="all_data_merged_clean.csv", 
        column_name="reliance_close", 
        threshold=2.0
    )


