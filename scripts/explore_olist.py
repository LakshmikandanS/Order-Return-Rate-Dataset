import sqlite3
import pandas as pd
import os

db_path = 'data/olist.sqlite'
if os.path.isdir(db_path):
    print("Trying nested path...")
    db_path = 'data/olist.sqlite/olist.sqlite'
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    res = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    tables = res['name'].tolist()
    print("Tables:", tables)
    for t in tables:
        df = pd.read_sql(f"PRAGMA table_info('{t}');", conn)
        print(f"\n--- Table: {t} ---")
        print(df[['name', 'type']])
        try:
            sample = pd.read_sql(f"SELECT * FROM `{t}` LIMIT 2;", conn)
            print("Sample data:")
            print(sample)
        except Exception as e:
            print("Failed to read sample", e)
    conn.close()