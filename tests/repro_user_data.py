import sys
import os
import numpy as np
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact

def check_overlap(box1, box2):
    x1, y1, z1, l1, w1, h1 = box1
    x2, y2, z2, l2, w2, h2 = box2
    
    # Epsilon for float comparison
    eps = 1e-4
    
    # Overlap if intersects in ALL 3 dimensions
    # Strict inequality > eps to avoid touching
    overlap_x = (x1 < x2 + l2 - eps) and (x1 + l1 > x2 + eps)
    overlap_y = (y1 < y2 + w2 - eps) and (y1 + w1 > y2 + eps)
    overlap_z = (z1 < z2 + h2 - eps) and (z1 + h1 > z2 + eps)
    
    if overlap_x and overlap_y and overlap_z:
        return True
    return False

def test_user_data_collisions():
    print("Testing Collision with User Data (Fractional Dimensions)...")
    
    # Warehouse: 10x10x10 (Small warehouse for these small items?)
    # User items are ~1.0 unit. Warehouse likely 100?
    # If warehouse is huge and items tiny, density is low.
    wh_dims = (100, 100, 100, 0, 0)
    layer_heights = [0.0]
    allocation_zones = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100}]
    
    # Load first 50 items from CSV manually (mocking them)
    # Based on the file view: 0.78, 0.51, 0.58 etc.
    num_items = 50
    items_props = np.zeros((num_items, 8), dtype=np.float32)
    
    # Hardcode a few problematic small ones from the CSV
    # Row 2: 0.78, 0.51, 0.58
    # Row 3: 1.1, 0.48, 0.5
    
    np.random.seed(42)
    for i in range(num_items):
        # Generate random fractional dims similar to users
        l = round(np.random.uniform(0.5, 1.2), 2)
        w = round(np.random.uniform(0.3, 0.8), 2)
        h = round(np.random.uniform(0.2, 0.6), 2)
        items_props[i] = [l, w, h, 1, 1, 0, 10, 0]
        
    solution = np.zeros((num_items, 4))
    
    print("Running repair...")
    new_sol = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, layer_heights)
    
    overlaps = 0
    boxes = []
    for i in range(num_items):
        x, y, z, rot = new_sol[i]
        l, w, h = items_props[i, 0:3]
        if int(rot) % 180 == 0:
            dx, dy = l, w
        else:
            dx, dy = w, l
        
        # Center to Corner
        boxes.append((x - dx/2, y - dy/2, z, dx, dy, h))
        
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if check_overlap(boxes[i], boxes[j]):
                print(f"[FAIL] Overlap Item {i} vs {j}")
                print(f" I{i}: {boxes[i]}")
                print(f" I{j}: {boxes[j]}")
                overlaps += 1
                
    if overlaps == 0:
        print(f"[PASS] No overlaps with fractional items.")
    else:
        print(f"[FAIL] {overlaps} overlaps detected.")

if __name__ == "__main__":
    test_user_data_collisions()
