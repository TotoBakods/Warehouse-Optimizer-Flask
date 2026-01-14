import sqlite3
import json
from datetime import datetime
import pandas as pd
import os
import uuid

DB_PATH = 'warehouse.db'

# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS items
                 (id TEXT PRIMARY KEY, name TEXT, length REAL, width REAL, 
                  height REAL, weight REAL, category TEXT, priority INTEGER,
                  fragility INTEGER, stackable INTEGER, access_freq INTEGER,
                  can_rotate INTEGER, x REAL, y REAL, z REAL, rotation INTEGER,
                  warehouse_id INTEGER DEFAULT 1)''')

    c.execute('''CREATE TABLE IF NOT EXISTS warehouse_config
                 (id INTEGER PRIMARY KEY, name TEXT, length REAL, width REAL, height REAL,
                  grid_size REAL, levels INTEGER, walkway_width REAL, layer_heights_json TEXT,
                  is_active INTEGER DEFAULT 1)''')

    c.execute('''CREATE TABLE IF NOT EXISTS exclusion_zones
                 (id INTEGER PRIMARY KEY, name TEXT, x1 REAL, y1 REAL, 
                  x2 REAL, y2 REAL, zone_type TEXT, warehouse_id INTEGER DEFAULT 1)''')

    c.execute('''CREATE TABLE IF NOT EXISTS optimization_results
                 (id INTEGER PRIMARY KEY, algorithm TEXT, fitness REAL,
                  space_utilization REAL, accessibility REAL, stability REAL,
                  grouping REAL, execution_time REAL, timestamp DATETIME, 
                  solution_data TEXT, warehouse_id INTEGER DEFAULT 1)''')

    # Insert default warehouse if it doesn't exist
    c.execute('''INSERT OR IGNORE INTO warehouse_config 
                 (id, name, length, width, height, grid_size, levels, walkway_width, layer_heights_json)
                 VALUES (1, 'Default Warehouse', 10.0, 8.0, 5.0, 0.5, 1, 1.0, '[]')''')

    conn.commit()
    conn.close()


def migrate_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('SELECT layer_heights_json FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding layer_heights_json column.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN layer_heights_json TEXT')
        c.execute("UPDATE warehouse_config SET layer_heights_json = '[]' WHERE layer_heights_json IS NULL")
        conn.commit()

    try:
        c.execute('SELECT grouping FROM optimization_results LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating optimization_results table: adding grouping column.")
        c.execute('ALTER TABLE optimization_results ADD COLUMN grouping REAL DEFAULT 0')
        conn.commit()

    try:
        c.execute('SELECT levels FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding levels, grid_size columns.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN levels INTEGER DEFAULT 1')
        c.execute('ALTER TABLE warehouse_config ADD COLUMN grid_size REAL DEFAULT 0.5')
        conn.commit()

    try:
        c.execute('SELECT warehouse_id FROM items LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating items table: adding warehouse_id column.")
        c.execute('ALTER TABLE items ADD COLUMN warehouse_id INTEGER DEFAULT 1')
        conn.commit()

    try:
        c.execute('SELECT warehouse_id FROM exclusion_zones LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating exclusion_zones table: adding warehouse_id column.")
        c.execute('ALTER TABLE exclusion_zones ADD COLUMN warehouse_id INTEGER DEFAULT 1')
        conn.commit()

    try:
        c.execute('SELECT warehouse_id FROM optimization_results LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating optimization_results table: adding warehouse_id column.")
        c.execute('ALTER TABLE optimization_results ADD COLUMN warehouse_id INTEGER DEFAULT 1')
        conn.commit()

    try:
        c.execute('SELECT name FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding name column.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN name TEXT')
        c.execute("UPDATE warehouse_config SET name = 'Default Warehouse' WHERE name IS NULL")
        conn.commit()

    try:
        c.execute('SELECT is_active FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding is_active column.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN is_active INTEGER DEFAULT 1')
        conn.commit()

    try:
        c.execute('SELECT zone_metadata FROM exclusion_zones LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating exclusion_zones table: adding zone_metadata column.")
        c.execute('ALTER TABLE exclusion_zones ADD COLUMN zone_metadata TEXT')
        conn.commit()

    try:
        c.execute('SELECT z1 FROM exclusion_zones LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating exclusion_zones table: adding z1, z2 columns.")
        c.execute('ALTER TABLE exclusion_zones ADD COLUMN z1 REAL DEFAULT 0')
        c.execute('ALTER TABLE exclusion_zones ADD COLUMN z2 REAL DEFAULT 100')
        conn.commit()

    try:
        c.execute('SELECT door_x FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding door_x, door_y columns.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN door_x REAL DEFAULT 0')
        c.execute('ALTER TABLE warehouse_config ADD COLUMN door_y REAL DEFAULT 0')
        conn.commit()

    try:
        c.execute('SELECT time_to_best FROM optimization_results LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating optimization_results table: adding time_to_best column.")
        c.execute('ALTER TABLE optimization_results ADD COLUMN time_to_best REAL DEFAULT 0')
        conn.commit()

    conn.close()

# Database helper functions
def get_all_items(warehouse_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if warehouse_id:
        c.execute('SELECT * FROM items WHERE warehouse_id = ?', (warehouse_id,))
    else:
        c.execute('SELECT * FROM items')

    rows = c.fetchall()
    conn.close()

    items = []
    for row in rows:
        items.append({
            'id': row[0], 'name': row[1], 'length': row[2], 'width': row[3],
            'height': row[4], 'weight': row[5], 'category': row[6], 'priority': row[7],
            'fragility': row[8], 'stackable': bool(row[9]), 'access_freq': row[10],
            'can_rotate': bool(row[11]), 'x': row[12], 'y': row[13], 'z': row[14],
            'rotation': row[15], 'warehouse_id': row[16] if len(row) > 16 else 1
        })
    return items


def get_item_by_id(item_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM items WHERE id = ?', (item_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {
            'id': row[0], 'name': row[1], 'length': row[2], 'width': row[3],
            'height': row[4], 'weight': row[5], 'category': row[6], 'priority': row[7],
            'fragility': row[8], 'stackable': bool(row[9]), 'access_freq': row[10],
            'can_rotate': bool(row[11]), 'warehouse_id': row[16] if len(row) > 16 else 1
        }
    return None


def get_warehouse_config(warehouse_id=1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM warehouse_config WHERE id = ?', (warehouse_id,))
    row = c.fetchone()
    conn.close()

    if row:
        config = {
            'id': row[0],
            'name': row[1],
            'length': row[2], 'width': row[3], 'height': row[4],
            'grid_size': row[5], 'levels': row[6], 'walkway_width': row[7],
            'door_x': row[9] if len(row) > 9 else 0,
            'door_y': row[10] if len(row) > 10 else 0
        }
        layer_heights_json = row[8] if len(row) > 8 else None
        if layer_heights_json:
            try:
                config['layer_heights'] = json.loads(layer_heights_json)
            except (json.JSONDecodeError, TypeError):
                config['layer_heights'] = []
        else:
            config['layer_heights'] = []
        return config
    return None


def get_all_warehouses():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM warehouse_config')
    rows = c.fetchall()
    conn.close()

    warehouses = []
    for row in rows:
        warehouse = {
            'id': row[0],
            'name': row[1],
            'length': row[2], 'width': row[3], 'height': row[4],
            'grid_size': row[5], 'levels': row[6], 'walkway_width': row[7],
            'is_active': row[9] if len(row) > 9 else 1
        }
        layer_heights_json = row[8] if len(row) > 8 else None
        if layer_heights_json:
            try:
                warehouse['layer_heights'] = json.loads(layer_heights_json)
            except (json.JSONDecodeError, TypeError):
                warehouse['layer_heights'] = []
        else:
            warehouse['layer_heights'] = []
        warehouses.append(warehouse)
    return warehouses


def get_exclusion_zones(warehouse_id=1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM exclusion_zones WHERE warehouse_id = ?', (warehouse_id,))
    rows = c.fetchall()
    conn.close()

    zones = []
    for row in rows:
        metadata = {}
        if len(row) > 8 and row[8]:
            try:
                metadata = json.loads(row[8])
            except Exception:
                pass
        
        zones.append({
            'id': row[0], 'name': row[1], 'x1': row[2], 'y1': row[3],
            'x2': row[4], 'y2': row[5], 'zone_type': row[6],
            'warehouse_id': row[7] if len(row) > 7 else 1,
            'metadata': metadata,
            'z1': row[9] if len(row) > 9 else 0,
            'z2': row[10] if len(row) > 10 else 100
        })
    return zones


def save_solution(solution, algorithm, fitness, space_util, accessibility, stability, grouping, exec_time,
                  warehouse_id=1, time_to_best=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if solution:
        for item_sol in solution:
            c.execute('''UPDATE items SET x = ?, y = ?, z = ?, rotation = ? 
                         WHERE id = ? AND warehouse_id = ?''',
                      (item_sol['x'], item_sol['y'], item_sol['z'], item_sol['rotation'],
                       item_sol['id'], warehouse_id))

    c.execute('''INSERT INTO optimization_results 
                 (algorithm, fitness, space_utilization, accessibility, stability, 
                  grouping, execution_time, timestamp, solution_data, warehouse_id, time_to_best)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (algorithm, fitness, space_util, accessibility, stability, grouping,
               exec_time, datetime.now(), json.dumps(solution), warehouse_id, time_to_best))

    conn.commit()
    conn.close()


def add_warehouse(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        name = data.get('name', 'New Warehouse')
        length = data.get('length', 10.0)
        width = data.get('width', 8.0)
        height = data.get('height', 5.0)
        grid_size = data.get('grid_size', 0.5)
        levels = data.get('levels', 1)
        walkway_width = data.get('walkway_width', 1.0)
        layer_heights = data.get('layer_heights', [])
        layer_heights_json = json.dumps(layer_heights)

        c.execute('''INSERT INTO warehouse_config 
                     (name, length, width, height, grid_size, levels, walkway_width, layer_heights_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (name, length, width, height, grid_size, levels, walkway_width, layer_heights_json))

        warehouse_id = c.lastrowid
        conn.commit()
        return warehouse_id
    finally:
        conn.close()


def delete_warehouse(warehouse_id):
    if warehouse_id == 1:
        raise ValueError('Cannot delete default warehouse')

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # Delete warehouse config
        c.execute('DELETE FROM warehouse_config WHERE id = ?', (warehouse_id,))
        # Delete associated items
        c.execute('DELETE FROM items WHERE warehouse_id = ?', (warehouse_id,))
        # Delete associated exclusion zones
        c.execute('DELETE FROM exclusion_zones WHERE warehouse_id = ?', (warehouse_id,))
        # Delete associated optimization results
        c.execute('DELETE FROM optimization_results WHERE warehouse_id = ?', (warehouse_id,))

        conn.commit()
    finally:
        conn.close()


def update_warehouse_config(warehouse_id, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    levels = data.get('levels', 1)
    height = data.get('height', 1)
    default_layer_heights = [i * (height / levels) for i in range(levels)] if levels > 0 else [0]
    layer_heights = data.get('layer_heights', default_layer_heights)
    layer_heights_json = json.dumps(layer_heights)

    try:
        c.execute('''UPDATE warehouse_config SET 
                     name = ?, length = ?, width = ?, height = ?, grid_size = ?, 
                     levels = ?, walkway_width = ?, layer_heights_json = ?,
                     door_x = ?, door_y = ? WHERE id = ?''',
                  (data.get('name', 'Warehouse'), data['length'], data['width'], data['height'],
                   data.get('grid_size', 0.5), data.get('levels', 1),
                   data.get('walkway_width', 1.0), layer_heights_json,
                   data.get('door_x', 0), data.get('door_y', 0), warehouse_id))

        conn.commit()
    finally:
        conn.close()


def add_item(data, warehouse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute('''INSERT OR REPLACE INTO items 
                     (id, name, length, width, height, weight, category, priority,
                      fragility, stackable, access_freq, can_rotate, x, y, z, rotation, warehouse_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (data['id'], data.get('name', ''), data['length'], data['width'],
                   data['height'], data.get('weight', 0), data.get('category', ''),
                   data.get('priority', 1), data.get('fragility', 0),
                   int(data.get('stackable', 1)), data.get('access_freq', 1),
                   int(data.get('can_rotate', 1)), data.get('x', 0), data.get('y', 0),
                   data.get('z', 0), data.get('rotation', 0), warehouse_id))

        conn.commit()
    finally:
        conn.close()


def update_item(item_id, data, warehouse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:

        # Whitelist columns to prevent injection
        valid_columns = {'name', 'length', 'width', 'height', 'weight', 'category', 
                         'priority', 'fragility', 'stackable', 'access_freq', 'can_rotate', 
                         'x', 'y', 'z', 'rotation'}
        
        fields = []
        values = []
        for key, value in data.items():
            if key in valid_columns:
                fields.append(f"{key} = ?")
                values.append(value)

        if not fields:
             return # Nothing to update

        values.append(item_id)
        values.append(warehouse_id)
        query = f"UPDATE items SET {', '.join(fields)} WHERE id = ? AND warehouse_id = ?"
        c.execute(query, values)

        conn.commit()
    finally:
        conn.close()


def delete_item(item_id, warehouse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute('DELETE FROM items WHERE id = ? AND warehouse_id = ?', (item_id, warehouse_id))
        conn.commit()
    finally:
        conn.close()


def clear_data(warehouse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute('DELETE FROM items WHERE warehouse_id = ?', (warehouse_id,))
        c.execute('DELETE FROM optimization_results WHERE warehouse_id = ?', (warehouse_id,))
        conn.commit()
    finally:
        conn.close()


def add_exclusion_zone(data, warehouse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        metadata = json.dumps(data.get('metadata', {}))
        c.execute('''INSERT INTO exclusion_zones (name, x1, y1, x2, y2, zone_type, warehouse_id, zone_metadata, z1, z2)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (data['name'], data['x1'], data['y1'], data['x2'], data['y2'],
                   data.get('zone_type', 'exclusion'), warehouse_id, metadata,
                   data.get('z1', 0), data.get('z2', 100)))

        conn.commit()
        return c.lastrowid
    finally:
        conn.close()


def update_exclusion_zone(zone_id, data, warehouse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        metadata = json.dumps(data.get('metadata', {}))
        c.execute('''UPDATE exclusion_zones 
                     SET name = ?, x1 = ?, y1 = ?, x2 = ?, y2 = ?, z1 = ?, z2 = ?, 
                         zone_type = ?, zone_metadata = ?
                     WHERE id = ? AND warehouse_id = ?''',
                  (data['name'], data['x1'], data['y1'], data['x2'], data['y2'], 
                   data.get('z1', 0), data.get('z2', 100),
                   data.get('zone_type', 'exclusion'), metadata, zone_id, warehouse_id))
        conn.commit()
    finally:
        conn.close()


def delete_exclusion_zone(zone_id, warehouse_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute('DELETE FROM exclusion_zones WHERE id = ? AND warehouse_id = ?', (zone_id, warehouse_id))
        conn.commit()
    finally:
        conn.close()


def create_default_sample_data():
    try:
        sample_data = [
            ['BOX001', 'Small Box', 1.0, 0.8, 0.6, 12.0, 'General', 2, 0, 1, 3, 1, 0, 0, 0, 0],
            ['BOX002', 'Medium Box', 1.2, 1.0, 0.8, 18.5, 'General', 2, 0, 1, 2, 1, 0, 0, 0, 0],
            ['BOX003', 'Large Box', 1.5, 1.2, 1.0, 25.0, 'General', 3, 0, 1, 1, 1, 0, 0, 0, 0],
            ['ELEC001', 'Electronics', 0.8, 0.6, 0.4, 8.2, 'Electronics', 1, 1, 0, 5, 1, 0, 0, 0, 0],
            ['HEAVY001', 'Heavy Item', 2.0, 1.5, 1.0, 45.0, 'Heavy', 3, 0, 1, 1, 0, 0, 0, 0, 0]
        ]

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Only delete generic items if needed, but the original code deleted ALL items. 
        # Assuming this is for initialization or fallback.
        c.execute('DELETE FROM items')

        for item in sample_data:
            c.execute('''INSERT INTO items 
                         (id, name, length, width, height, weight, category, priority,
                          fragility, stackable, access_freq, can_rotate, x, y, z, rotation)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', item)

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating default sample data: {e}")
        return False


def load_sample_data(warehouse_id=1):
    try:
        if os.path.exists('datasets.csv'):
            conn = sqlite3.connect(DB_PATH)
            
            # Clear existing items first to avoid duplicates if replace isn't working as expected per chunk
            # But the original code used 'replace' which drops the table. 
            # We can't drop table per chunk. We should drop items for this warehouse first.
            # However, pandas `to_sql(if_exists='replace')` drops the *table*, which is bad for other warehouses!
            # The original code `to_sql(..., if_exists='replace')` DROPPED THE ITEMS TABLE completely!
            # That was a bug in the original code if multiple warehouses existed. 
            # We should correct this to `delete from items where warehouse_id=...` and then append.
            
            c = conn.cursor()
            c.execute('DELETE FROM items WHERE warehouse_id = ?', (warehouse_id,))
            conn.commit()

            # Define all possible columns
            potential_columns = {
                'id': 'id',
                'name': 'name', 
                'length': 'length',
                'width': 'width',
                'height': 'height',
                'weight': 'weight',
                'category': 'category',
                'priority': 'priority',
                'fragility': 'fragility',
                'stackable': 'stackable',
                'access_freq': 'access_freq',
                'can_rotate': 'can_rotate',
                'x': 'x',
                'y': 'y',
                'z': 'z',
                'rotation': 'rotation'
            }
            
            chunk_size = 10000
            for chunk in pd.read_csv('datasets.csv', chunksize=chunk_size):
                # Only map columns that actually exist in the CSV
                column_mapping = {k: v for k, v in potential_columns.items() if k in chunk.columns}
                
                db_df = chunk[list(column_mapping.keys())].rename(columns=column_mapping)

                # Ensure all required DB columns exist, defaulting to 0/empty
                for col, default_val in [
                    ('x', 0), ('y', 0), ('z', 0), ('rotation', 0), 
                    ('name', ''), ('category', 'General')
                ]:
                    if col not in db_df.columns:
                        db_df[col] = default_val
                
                db_df = db_df.fillna({
                    'x': 0, 'y': 0, 'z': 0, 'rotation': 0,
                    'name': '', 'category': 'General'
                })
                
                # Set warehouse_id
                db_df['warehouse_id'] = warehouse_id
                
                # Append each chunk
                db_df.to_sql('items', conn, if_exists='append', index=False)
            
            conn.close()
            return True
        else:
            return create_default_sample_data()
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return create_default_sample_data()


def load_generated_data(warehouse_id=1):
    generated_file = os.path.join('gan', 'generated_items.csv')
    try:
        if os.path.exists(generated_file):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            # Clear existing items for this warehouse
            c.execute('DELETE FROM items WHERE warehouse_id = ?', (warehouse_id,))
            conn.commit()

            total_items = 0
            chunk_size = 10000

            for chunk in pd.read_csv(generated_file, chunksize=chunk_size):
                # Define all possible columns
                potential_columns = {
                    'id': 'id',
                    'name': 'name', 
                    'length': 'length',
                    'width': 'width',
                    'height': 'height',
                    'weight': 'weight',
                    'category': 'category',
                    'priority': 'priority',
                    'fragility': 'fragility',
                    'stackable': 'stackable',
                    'access_freq': 'access_freq',
                    'can_rotate': 'can_rotate',
                    'x': 'x',
                    'y': 'y',
                    'z': 'z',
                    'rotation': 'rotation'
                }
                
                # Only map columns that actually exist in the CSV
                column_mapping = {k: v for k, v in potential_columns.items() if k in chunk.columns}

                db_df = chunk[list(column_mapping.keys())].rename(columns=column_mapping)

                # Ensure all required DB columns exist, defaulting to 0/empty
                for col, default_val in [
                    ('x', 0), ('y', 0), ('z', 0), ('rotation', 0), 
                    ('name', 'Synthetic Item'), ('category', 'General')
                ]:
                    if col not in db_df.columns:
                        db_df[col] = default_val
                
                db_df = db_df.fillna({
                    'x': 0, 'y': 0, 'z': 0, 'rotation': 0,
                    'name': 'Synthetic Item', 'category': 'General'
                })
                
                # Set warehouse_id
                db_df['warehouse_id'] = warehouse_id

                # Append chunk
                db_df.to_sql('items', conn, if_exists='append', index=False)
                total_items += len(db_df)

            conn.close()
            return True, total_items
        else:
            return False, "File not found"
    except Exception as e:
        print(f"Error loading generated data: {e}")
        return False, str(e)


def get_metrics_history(warehouse_id=1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT algorithm, fitness, space_utilization, accessibility, 
                 stability, execution_time, timestamp, time_to_best FROM optimization_results 
                 WHERE warehouse_id = ? ORDER BY timestamp DESC LIMIT 50''', (warehouse_id,))
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            'algorithm': row[0],
            'fitness': row[1],
            'space_utilization': row[2],
            'accessibility': row[3],
            'stability': row[4],
            'execution_time': row[5],
            'timestamp': row[6],
            'time_to_best': row[7] if len(row) > 7 else 0
        })
    return history


def get_item_stats_by_category(warehouse_id=1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT category, COUNT(*), SUM(length * width * height)
                 FROM items WHERE warehouse_id = ? GROUP BY category''', (warehouse_id,))
    rows = c.fetchall()
    conn.close()

    # Return format expected by Chart.js frontend
    categories = []
    counts = []
    volumes = []
    
    for row in rows:
        categories.append(row[0] if row[0] else 'Uncategorized')
        counts.append(row[1])
        volumes.append(row[2] if row[2] else 0)
    
    return {
        'categories': categories,
        'counts': counts,
        'volumes': volumes
    }
