import pandas as pd
import sqlite3

# --- Configuration ---
DATABASE_FILE = "market_data.db" 
# ---------------------

def prepare_master_timeseries_data(db_file: str = DATABASE_FILE) -> pd.DataFrame:
    """
    Finalized function for Step 1: Data Preparation.
    Handles the complex tuple-based column names found during debugging.
    """
    
    # Define table names and the EXACT column name for the primary value
    # The actual column names are complex strings found from the debug output:
    
    # NOTE: The value column names must match the exact string format from your DEBUG output.
    table_map = {
        "reliance_stocks": {
            "final_col": "Reliance_Close", 
            "expected_cols": ["('stock_price', 'RELIANCE.NS')", 'stock_price', 'Close']
        }, 
        "gold_prices": {
            "final_col": "Gold_Price", 
            "expected_cols": ["('Gold Price (USD/oz)', 'GC=F')", 'Price', 'Rate']
        },
        "petrol_prices": {
            "final_col": "Petrol_Price", 
            "expected_cols": ["('oil_price', 'CL=F')", 'oil_price', 'Close']
        }, 
        "forex_rates": {
            "final_col": "USD_INR_Rate", 
            # Forex has TWO value columns, we must handle both separately
            "expected_cols": ["('USDINR=X', 'USDINR=X')", "('EURINR=X', 'EURINR=X')", 'Rate', 'Price']
        }
    }
    
    all_dfs = []
    
    try:
        # A. LOAD AND INSPECT DATA
        with sqlite3.connect(db_file) as conn:
            print(f"✅ Database connection successful: {db_file}")
            
            for table, mapping in table_map.items():
                print(f"\n   -> Loading table: {table}")
                
                # 1. Fetch all columns without setting an index yet
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                
                # 🛑 DEBUG: Columns loaded: (Keep this here for reference)
                print(f"   🛑 DEBUG: Columns loaded: {df.columns.tolist()}")
                
                # --- FIX 1: HIGHLY ROBUST Date/Index Column Identification (SUCCESSFUL) ---
                potential_date_names = ['date', 'index', 'level_0', df.columns[0].lower()]
                date_cols = [col for col in df.columns if col.lower() in potential_date_names]
                
                if not date_cols:
                    print(f"   [WARNING] Could not find a suitable Date/Index column in {table}. Skipping.")
                    continue
                
                date_col_name = date_cols[0]
                df = df.rename(columns={date_col_name: 'Date'})
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                
                
                # --- FIX 2: Identify and Standardize the Value Columns (THE NEW FIX) ---
                
                # Container for all columns we successfully find for this table
                found_value_cols = {} 

                # Handle Forex (which has two value columns)
                if table == 'forex_rates':
                    # Look for USDINR
                    usd_col = "('USDINR=X', 'USDINR=X')"
                    if usd_col in df.columns:
                        found_value_cols[usd_col] = 'USD_INR_Rate' # Use the final column name
                    
                    # Look for EURINR
                    eur_col = "('EURINR=X', 'EURINR=X')"
                    if eur_col in df.columns:
                        found_value_cols[eur_col] = 'EUR_INR_Rate' # New column for EUR
                    
                    if not found_value_cols:
                        print(f"   [WARNING] No primary value columns found in {table}. Skipping.")
                        continue
                        
                # Handle Reliance, Gold, Petrol (which have one primary value column)
                else:
                    value_col = None
                    # Search for the expected columns in the order defined in table_map
                    for expected_col in mapping['expected_cols']:
                        if expected_col in df.columns:
                            value_col = expected_col
                            found_value_cols[value_col] = mapping['final_col']
                            break
                    
                    if not value_col:
                        print(f"   [WARNING] No primary value column found in {table}. Skipping.")
                        continue
                
                # Clean the DataFrame and prepare for merge
                print(f"   -> Identified value columns: {list(found_value_cols.keys())}")
                
                # Create a temporary DataFrame with only the found columns and their standardized names
                df_cleaned = df[list(found_value_cols.keys())].rename(columns=found_value_cols)
                all_dfs.append(df_cleaned)

    except Exception as e:
        print(f"❌ CRITICAL ERROR during database connection/loading: {e}")
        return None

    if not all_dfs:
        print("❌ No valid dataframes were loaded. Cannot proceed.")
        return None

    # C. COMBINE TIME-SERIES DATA
    print("\n\n2. Combining and Cleaning Time-Series Data...")
    
    # Start with the first DataFrame
    master_df = all_dfs[0].copy()
    
    # Merge the rest using the common datetime index 
    for df in all_dfs[1:]:
        # Use merge for alignment; outer merge ensures we don't drop dates
        master_df = master_df.merge(df, how='outer', left_index=True, right_index=True)

    print(f"   -> Combined Data Shape (before cleaning): {master_df.shape}")

    # B. HANDLE MISSING DATA (IMPUTATION)
    # 1. Reindex to include all dates between the min and max date (Daily frequency 'D')
    idx = pd.date_range(start=master_df.index.min(), end=master_df.index.max(), freq='D')
    master_df = master_df.reindex(idx)
    
    # 2. Forward-fill the missing values (fills weekends/holidays with the last valid price)
    print("   -> Imputing missing data using Forward-Fill (ffill)...")
    master_df = master_df.ffill()
    
    # Drop any rows that remain NaN
    master_df = master_df.dropna()
    
    print(f"\n✨ FINAL MASTER DATAFRAME CREATED ✨")
    print(f"   -> Final Shape: {master_df.shape}")
    print("   -> Columns:", master_df.columns.tolist())
    
    return master_df

# --- EXECUTE THE FUNCTION ---
if __name__ == "__main__":
    master_data = prepare_master_timeseries_data()
    
    if master_data is not None:
        print("\n--- Head of Master Data ---")
        print(master_data.head())
        print("\n--- Tail of Master Data ---")
        print(master_data.tail())