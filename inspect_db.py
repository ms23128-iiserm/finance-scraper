import sqlite3
import pandas as pd

# Path to your SQLite database file
DB_FILE = "market_data.db"

def inspect_database():
    """Displays all table names and their record counts, then shows first 5 rows from each."""
    with sqlite3.connect(DB_FILE) as conn:
        # Get all table names
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table_names = tables['name'].tolist()

        print(f"\n--- Found {len(table_names)} Tables in {DB_FILE} ---\n")
        for name in table_names:
            # Count rows in each table
            count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {name};", conn)['count'][0]
            print(f"Table: {name}  →  {count} records")

        print("\n--- Preview (first 5 rows per table) ---\n")
        for name in table_names:
            print(f"\n🧩 {name.upper()} — first 5 rows:")
            df = pd.read_sql_query(f"SELECT * FROM {name} LIMIT 5;", conn)
            print(df)
            print("-" * 80)

if __name__ == "__main__":
    inspect_database()




