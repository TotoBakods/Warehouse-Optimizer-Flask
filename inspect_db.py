import sqlite3
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def inspect():
    conn = sqlite3.connect('warehouse.db')
    
    print("Tables:")
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(tables)
    
    print("Warehouses:")
    whs = pd.read_sql("SELECT * FROM warehouse_config WHERE id=1", conn)
    print(whs)
    
    print("\nItems for Warehouse 1:")
    items = pd.read_sql("SELECT count(*) as count, avg(length) as avg_len, avg(width) as avg_wid, avg(height) as avg_hgt FROM items WHERE warehouse_id=1", conn)
    print(items)
    
    conn.close()

if __name__ == '__main__':
    inspect()
