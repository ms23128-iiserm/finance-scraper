import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler # For normalization

def run_correlation_analysis():
    """
    Loads all financial CSVs, merges them, handles missing data,
    plots a normalized time-series, and then calculates
    and plots a correlation matrix.
    """
    print("--- Starting Correlation Analysis ---")
    
    # --- 1. DEFINE FILE PATHS ---
    base_path = Path('.') # Assumes files are in the same directory
    
    file_info = {
        'reliance': {
            'path': base_path / 'reliance_data.csv',
            'date_col': 'date',
            'value_col': 'reliance_close'
        },
        'gold': {
            'path': base_path / 'gold_data.csv',
            'date_col': 'Date',
            'value_col': 'Gold Price (USD/oz)'
        },
        'petrol': {
            'path': base_path / 'petrol_data.csv',
            'date_col': 'Date',
            'value_col': 'oil_price'
        },
        'forex': {
            'path': base_path / 'forex_data.csv',
            'date_col': 'date',
            'value_cols': ['USDINR=X', 'EURINR=X']
        }
    }
    
    all_dataframes = []
    missing_files = []

    # --- 2. LOAD & PREPARE EACH DATAFRAME ---
    try:
        # Load Reliance
        df_rel = pd.read_csv(file_info['reliance']['path'])
        df_rel = df_rel[[file_info['reliance']['date_col'], file_info['reliance']['value_col']]]
        # --- FIX: Force value column to be numeric, coercing errors to NaN ---
        df_rel[file_info['reliance']['value_col']] = pd.to_numeric(df_rel[file_info['reliance']['value_col']], errors='coerce')
        df_rel[file_info['reliance']['date_col']] = pd.to_datetime(df_rel[file_info['reliance']['date_col']])
        df_rel = df_rel.set_index(file_info['reliance']['date_col'])
        all_dataframes.append(df_rel)
        print(f"✅ Loaded {file_info['reliance']['path'].name}")

        # Load Gold
        df_gold = pd.read_csv(file_info['gold']['path'])
        df_gold = df_gold[[file_info['gold']['date_col'], file_info['gold']['value_col']]]
        # --- FIX: Force value column to be numeric, coercing errors to NaN ---
        df_gold[file_info['gold']['value_col']] = pd.to_numeric(df_gold[file_info['gold']['value_col']], errors='coerce')
        df_gold[file_info['gold']['date_col']] = pd.to_datetime(df_gold[file_info['gold']['date_col']])
        df_gold = df_gold.set_index(file_info['gold']['date_col'])
        all_dataframes.append(df_gold)
        print(f"✅ Loaded {file_info['gold']['path'].name}")

        # Load Petrol
        df_petrol = pd.read_csv(file_info['petrol']['path'])
        df_petrol = df_petrol[[file_info['petrol']['date_col'], file_info['petrol']['value_col']]]
        # --- FIX: Force value column to be numeric, coercing errors to NaN ---
        df_petrol[file_info['petrol']['value_col']] = pd.to_numeric(df_petrol[file_info['petrol']['value_col']], errors='coerce')
        df_petrol[file_info['petrol']['date_col']] = pd.to_datetime(df_petrol[file_info['petrol']['date_col']])
        df_petrol = df_petrol.set_index(file_info['petrol']['date_col'])
        all_dataframes.append(df_petrol)
        print(f"✅ Loaded {file_info['petrol']['path'].name}")

        # Load Forex
        df_forex = pd.read_csv(file_info['forex']['path'])
        df_forex = df_forex[[file_info['forex']['date_col']] + file_info['forex']['value_cols']]
        # --- FIX: Force value columns to be numeric, coercing errors to NaN ---
        for col in file_info['forex']['value_cols']:
            df_forex[col] = pd.to_numeric(df_forex[col], errors='coerce')
        df_forex[file_info['forex']['date_col']] = pd.to_datetime(df_forex[file_info['forex']['date_col']])
        df_forex = df_forex.set_index(file_info['forex']['date_col'])
        all_dataframes.append(df_forex)
        print(f"✅ Loaded {file_info['forex']['path'].name}")

    except Exception as e:
        # Check which file is missing
        for key, info in file_info.items():
            if not info['path'].exists():
                missing_files.append(info['path'].name)
                
        print(f"\n--- ERROR ---")
        print(f"An error occurred: {e}")
        print("One or more CSV files could not be loaded. Please run all scrapers first.")
        if missing_files:
            print(f"Missing files: {', '.join(missing_files)}")
        return # Stop the script

    # --- 3. MERGE & CLEAN DATA ---
    print("\nMerging all dataframes...")
    
    # Merge all dataframes on their date index
    # 'outer' join keeps all dates from all files
    merged_df = pd.concat(all_dataframes, axis=1, join='outer')
    
    # Sort by date
    merged_df = merged_df.sort_index()
    
    # Handle missing values (e.g., market holidays)
    # ffill() = "forward fill" -> uses the last known price
    # dropna() will drop any rows that still have NaN (e.g., from the 'coerce' fix)
    merged_df_clean = merged_df.ffill().dropna()
    
    if merged_df_clean.empty:
        print("\n--- ERROR ---")
        print("No data remained after cleaning. Check your CSV files for major data gaps.")
        return

    print(f"Data merged and cleaned. Total shared trading days: {len(merged_df_clean)}")
    
    # Save the cleaned master dataset
    merged_df_clean.to_csv("all_data_merged_clean.csv")
    print("💾 Saved combined, cleaned data to 'all_data_merged_clean.csv'")
    
    # --- 4. NORMALIZE DATA FOR PLOTTING ---
    print("\nNormalizing data for time-series plot...")
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(merged_df_clean)
    normalized_df = pd.DataFrame(
        normalized_data, 
        columns=merged_df_clean.columns, 
        index=merged_df_clean.index
    )
    
    # --- 5. PLOT NORMALIZED TIME-SERIES (The "Superimposed" Graph) ---
    print("Generating normalized time-series plot...")
    plt.figure(figsize=(14, 8))
    for column in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[column], label=column)
    
    plt.title('Normalized Price Movements Over Time (5 Years)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (0 to 1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    ts_plot_path = "normalized_timeseries_plot.png"
    plt.savefig(ts_plot_path)
    print(f"✅ Successfully saved time-series plot to '{ts_plot_path}'")
    plt.close() # Close the figure to free up memory

    # --- 6. CALCULATE CORRELATION ---
    print("\nCalculating correlation matrix...")
    
    # Calculate Pearson correlation coefficient
    correlation_matrix = merged_df_clean.corr(method='pearson')
    
    # Save the correlation matrix (the "parameters")
    correlation_matrix.to_csv("correlation_matrix.csv")
    print("💾 Saved correlation parameters to 'correlation_matrix.csv'")
    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)

    # --- 7. VISUALIZE HEATMAP ---
    print("\nGenerating heatmap visualization...")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True,     # Show the numbers in the squares
        cmap='coolwarm',# Color scheme
        fmt='.2f'       # Format numbers to 2 decimal places
    )
    plt.title('Correlation Heatmap of Reliance vs. External Factors')
    plt.tight_layout()
    
    # Save the plot as an image
    output_image_path = "correlation_heatmap.png"
    plt.savefig(output_image_path)
    plt.close() # Close the figure
    
    print(f"✅ Successfully saved heatmap to '{output_image_path}'")
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    run_correlation_analysis()

