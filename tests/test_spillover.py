
import numpy as np
import optimizer
from optimizer import repair_solution_compact

def test_fallback_chimney():
    print("Testing Fallback Chimney vs Spillover...")
    
    # Warehouse 30x30x30
    warehouse_dims = (30, 30, 30, 0, 0)
    
    # Zone A: Small. 0,0 to 12,12. Height 0-10.
    # Fit: 10x10 item fits.
    # 0,0 to 12,12 -> Width 12. Item 10. Fits 1 item loosely.
    allocation_zones = [
        {'zone_type': 'allocation', 'x1': 0, 'y1': 0, 'x2': 12, 'y2': 12, 'z1': 0, 'z2': 10}
    ]
    
    # 5 items.
    num_items = 5
    items_props = np.zeros((num_items, 8))
    for i in range(num_items):
        items_props[i] = [10, 10, 10, 0, 1, 0, 0, 0] # 10x10x10
    
    # Start all at 50,50 (Outside warehouse, or general location)
    solution = np.zeros((num_items, 4))
    for i in range(num_items):
        solution[i] = [50, 50, 0, 0] 
        
    repaired = repair_solution_compact(
        solution, items_props, warehouse_dims, allocation_zones
    )
    
    print("Repaired Coordinates:")
    for i in range(num_items):
        print(repaired[i])
        
    # Analyze
    # Item 0 should be in Zone A (approx 5,5 or 6,6).
    # Item 1-4:
    # If Chimney: They will be at approx 6,6, stacked Z=10, 20, 30, 40.
    # If Spillover: They will be outside Zone A (e.g. > 12 X or Y) and on Z=0.
    
    chimney_count = 0
    spillover_count = 0
    
    for i in range(num_items):
        x, y, z = repaired[i, 0], repaired[i, 1], repaired[i, 2]
        if x < 13 and y < 13:
            # Inside Zone A (or Fallback clamped to Zone A)
            if z > 0.1:
                chimney_count += 1
        elif z < 1.0:
            spillover_count += 1
            
    print(f"Chimney Items (In Zone A, Stacked): {chimney_count}")
    print(f"Spillover Items (Outside Zone A, Floor): {spillover_count}")
    
    if spillover_count > 0:
        print("SUCCESS: Items spilled over to open space.")
    else:
        print("FAILURE: Items formed a chimney in the full zone.")

if __name__ == "__main__":
    test_fallback_chimney()
