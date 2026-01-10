
import numpy as np
import optimizer
from optimizer import repair_solution_compact

def test_zone_distribution():
    print("Testing Zone Distribution...")
    
    # 2 Zones:
    # Zone A: 0,0 to 20,20. Height 0-10.
    # Zone B: 50,0 to 70,20. Height 0-10.
    allocation_zones = [
        {'zone_type': 'allocation', 'x1': 0, 'y1': 0, 'x2': 20, 'y2': 20, 'z1': 0, 'z2': 10},
        {'zone_type': 'allocation', 'x1': 50, 'y1': 0, 'x2': 70, 'y2': 20, 'z1': 0, 'z2': 10}
    ]
    
    # Warehouse large enough
    warehouse_dims = (100, 100, 100, 0, 0)
    
    # 8 Items of size 10x10x10.
    # Zone A can fit exactly 4 items on floor (20x20 vs 10x10 items -> 2x2 grid).
    # Since height is 10, it can fit 1 layer. Total capacity = 4 items.
    # We have 8 items.
    # We expect 4 items in Zone A, 4 items in Zone B.
    # If bug exists, 8 items will be in Zone A (stacked to Z=20, exceeding limit, or just overlapping if fallback old logic, but we fixed overlap so now they stack).
    
    num_items = 8
    items_props = np.zeros((num_items, 8))
    for i in range(num_items):
        items_props[i] = [10, 10, 10, 0, 1, 0, 0, 0] # 10x10x10
        
    solution = np.zeros((num_items, 4))
    # Init random places
    for i in range(num_items):
        solution[i] = [50, 50, 0, 0] 
        
    repaired = repair_solution_compact(solution, items_props, warehouse_dims, allocation_zones)
    
    print("Repaired Solution:")
    items_in_A = 0
    items_in_B = 0
    items_outside = 0
    
    for i in range(num_items):
        x, y, z = repaired[i, 0], repaired[i, 1], repaired[i, 2]
        print(f"Item {i}: {x}, {y}, {z}")
        
        in_A = (x >= 0 and x <= 20)
        in_B = (x >= 50 and x <= 70)
        
        if in_A: items_in_A += 1
        elif in_B: items_in_B += 1
        else: items_outside += 1
        
    print(f"Items in Zone A: {items_in_A}")
    print(f"Items in Zone B: {items_in_B}")
    
    if items_in_B == 0:
        print("FAILURE: No items in Zone B! Distribution failed.")
    elif items_in_A > 4:
         print(f"WARNING: Zone A Overfilled? ({items_in_A} items). Capacity should be 4 if strictly respecting height.")
    else:
        print("SUCCESS: Items distributed.")

if __name__ == "__main__":
    test_zone_distribution()
