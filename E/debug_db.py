import sqlite3
import pandas as pd
import os

DB_NAME = 'market_data.db'

def inspect_database():
    """
    Connects to the DB, lists all tables, and prints the columns 
    and first 3 rows of each table.
    """
    if not os.path.exists(DB_NAME):
        print(f"❌ ERROR: Database file '{DB_NAME}' not found.")
        print("Please run 'python main.py' first to create it.")
        return

    print(f"--- Inspecting Database: {DB_NAME} ---")
    
    try:
        conn = sqlite3.connect(DB_NAME)
        
        # 1. Get all table names
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("\n❌ ERROR: The database is empty. No tables found.")
            conn.close()
            return
            
        print(f"Tables found: {[table[0] for table in tables]}")

        # 2. For each table, print its schema and head
        for table in tables:
            table_name = table[0]
            print(f"\n--- Table: '{table_name}' ---")
            
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                
                if df.empty:
                    print("Status: This table is EMPTY.")
                    continue
                
                # Print the exact column names
                print(f"COLUMNS: {df.columns.tolist()}")
                
                # Print the first 3 rows
                print("--- First 3 Rows ---")
                print(df.head(3).to_string())
                
            except Exception as e:
                print(f"Could not read table '{table_name}': {e}")
                
        conn.close()

    except Exception as e:
        print(f"An error occurred while connecting to the database: {e}")

if __name__ == "__main__":
    inspect_database()
