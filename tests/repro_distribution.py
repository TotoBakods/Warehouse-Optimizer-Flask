import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact

def test_distribution():
    print("Testing Zone Distribution...")
    
    # Warehouse: 100x100x100
    wh_dims = (100, 100, 100, 0, 0)
    layer_heights = [0.0]
    
    # Two identical zones: Left and Right
    # Zone A: 0-50, 0-100
    # Zone B: 50-100, 0-100
    # Both have area 5000.
    allocation_zones = [
        {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 100, 'z1': 0, 'z2': 100}, # Zone A
        {'x1': 50, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100} # Zone B
    ]
    
    # 20 items. 10x10x10.
    num_items = 20
    items_props = np.zeros((num_items, 8), dtype=np.float32)
    for i in range(num_items):
        items_props[i] = [10, 10, 10, 0, 1, 0, 10, 0]
        
    solution = np.zeros((num_items, 4))
    
    print("Running repair...")
    new_sol = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, layer_heights)
    
    zone_a_count = 0
    zone_b_count = 0
    
    for i in range(num_items):
        x = new_sol[i, 0]
        if x < 50:
            zone_a_count += 1
        else:
            zone_b_count += 1
            
    print(f"Zone A Count: {zone_a_count}")
    print(f"Zone B Count: {zone_b_count}")
    
    # Expect roughly equal distribution (e.g. 10 +/- 5)
    # If logic was First-Fit deterministic, Zone A would accept ALL 20 items because they fit.
    # So if Zone A has 20 and Zone B has 0 -> FAIL.
    # If Zone A has ~10 and Zone B has ~10 -> PASS.
    
    if abs(zone_a_count - zone_b_count) < 10:
        print("[PASS] Items distributed across zones.")
    else:
        print("[FAIL] Distribution imbalance detected.")

if __name__ == "__main__":
    test_distribution()
