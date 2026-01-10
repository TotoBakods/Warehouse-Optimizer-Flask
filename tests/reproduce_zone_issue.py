import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer import repair_solution_compact

def test_multiple_allocation_zones():
    print("Testing Multiple Allocation Zones...")
    
    # 1. Setup Warehouse and Items
    # Warehouse: 100x100x100
    warehouse_dims = (100, 100, 100, 0, 0)
    
    # Define 2 Zones:
    # Zone 1: Left side (0-40)
    # Zone 2: Right side (60-100)
    # Gap in middle (40-60) ensures distinct separation
    allocation_zones = [
        {'x1': 0, 'y1': 0, 'x2': 40, 'y2': 100, 'z1': 0, 'z2': 100},
        {'x1': 60, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100}
    ]
    
    # Create 10 identical items that fit in either zone
    # Size: 10x10x10
    num_items = 20
    items_props = np.zeros((num_items, 8))
    items_props[:, 0] = 10 # Length
    items_props[:, 1] = 10 # Width
    items_props[:, 2] = 10 # Height
    items_props[:, 3] = 1  # Can rotate (doesn't matter for cube)
    
    # Initial random solution
    solution = np.zeros((num_items, 4))
    solution[:, 0] = 50 # Start in the middle (invalid/gap)
    solution[:, 1] = 50
    solution[:, 2] = 0
    
    # 2. Run Repair
    print("Running repair_solution_compact...")
    repaired_solution = repair_solution_compact(
        solution, items_props, warehouse_dims, allocation_zones, layer_heights=[0.0]
    )
    
    # 3. Analyze Results
    in_zone_1 = 0
    in_zone_2 = 0
    outside_zones = 0
    
    print("\nItem Positions:")
    for i in range(num_items):
        x, y, z = repaired_solution[i, 0], repaired_solution[i, 1], repaired_solution[i, 2]
        
        # Check Zone 1 (allowing for item half-width 5)
        # Center in 0-40? Actually bounds check uses min/max.
        # Item 10x10, so min_x >= 0, max_x <= 40
        # center x: 5 to 35
        is_z1 = (x >= 5) and (x <= 35)
        
        # Check Zone 2
        # center x: 65 to 95
        is_z2 = (x >= 65) and (x <= 95)
        
        if is_z1:
            in_zone_1 += 1
        elif is_z2:
            in_zone_2 += 1
        else:
            outside_zones += 1
            # print(f"  Item {i}: ({x:.1f}, {y:.1f}, {z:.1f}) - OUTSIDE")
            
    print(f"\nSummary:")
    print(f"Items in Zone 1: {in_zone_1}")
    print(f"Items in Zone 2: {in_zone_2}")
    print(f"Items Outside:   {outside_zones}")
    
    if in_zone_1 > 0 and in_zone_2 > 0:
        print("\nSUCCESS: Items distributed across both zones.")
    elif in_zone_1 > 0 and in_zone_2 == 0:
        print("\nFAILURE: All items in Zone 1. Zone 2 ignored.")
    elif in_zone_1 == 0 and in_zone_2 > 0:
        print("\nFAILURE: All items in Zone 2. Zone 1 ignored.")
    else:
        print("\nFAILURE: Items handling unclear.")

if __name__ == "__main__":
    test_multiple_allocation_zones()
