import json
import csv
import os

INPUT_FILE = '../bed-bpp_v1.json'
OUTPUT_FILE = '../datasets.csv'

def convert():
    input_path = os.path.join(os.path.dirname(__file__), INPUT_FILE)
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)

    print(f"Reading from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    items = []
    
    print("Processing items...")
    for order_id, order_data in data.items():
        if "item_sequence" in order_data:
            for seq_key, item_data in order_data["item_sequence"].items():
                
                try:
                    # dimensions in mm -> m
                    length = item_data.get("length/mm", 0) / 1000.0
                    width = item_data.get("width/mm", 0) / 1000.0
                    height = item_data.get("height/mm", 0) / 1000.0
                    weight = item_data.get("weight/kg", 0)
                    
                    # Basic validation
                    if length <= 0 or width <= 0 or height <= 0:
                        continue

                    items.append({
                        'id': item_data.get('id', f"{order_id}-{seq_key}"),
                        'name': item_data.get('article', 'Unknown Item'),
                        'length': round(length, 3),
                        'width': round(width, 3),
                        'height': round(height, 3),
                        'weight': round(weight, 3),
                        'category': item_data.get('product_group', 'General'),
                        'priority': 1,
                        'fragility': 0,
                        'stackable': 1,
                        'access_freq': 1,
                        'can_rotate': 1
                    })
                except Exception as e:
                    print(f"Skipping item due to error: {e}")
                    continue

    print(f"Found {len(items)} valid items.")
    
    print(f"Writing to {output_path}...")
    fieldnames = ['id', 'name', 'length', 'width', 'height', 'weight', 'category', 
                  'priority', 'fragility', 'stackable', 'access_freq', 'can_rotate']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)

    print("Conversion complete.")

if __name__ == "__main__":
    convert()
