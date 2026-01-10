
import numpy as np
import optimizer
from optimizer import repair_solution_compact, calculate_z_for_item

def test_overlap_fallback_forced():
    print("Testing Collision Fallback with constrained height...")
    
    # Setup
    # Warehouse 10x10 x 15 (Height 15)
    warehouse_dims = (10, 10, 15, 0, 0)
    
    num_items = 2
    items_props = np.zeros((num_items, 8))
    # Item 0: 10x10x10
    items_props[0] = [10, 10, 10, 1, 1, 0, 0, 0]
    # Item 1: 10x10x10
    items_props[1] = [10, 10, 10, 1, 1, 0, 0, 0]
    
    # Initial solution: Both at 5, 5, 0 (Center)
    solution = np.zeros((num_items, 4))
    solution[0] = [5, 5, 0, 0]
    solution[1] = [5, 5, 0, 0]
    
    # Run repair
    # Logic expectation:
    # Item 1 place at Z=0. Occupies 0-10.
    # Item 2 tries to place.
    # Grid search: Best Z = 10.
    # Check: 10 + 10 = 20 > 15 (Warehouse Height).
    # Grid search fails to find fit.
    # Trigger Fallback.
    # Fallback X,Y = 5,5.
    # Fallback Z = ?
    # OLD CODE: max(0, min(15-10, 0)) = 0. Or clamped to 5. --> Overlap.
    # NEW CODE: calculate_z_for_item -> 10. --> Stack (exceeds height, but no overlap).
    
    repaired = repair_solution_compact(solution.copy(), items_props, warehouse_dims)
    
    print("Repaired Solution:")
    print(repaired)
    
    z0 = repaired[0, 2]
    z1 = repaired[1, 2]
    
    print(f"Item 0 Z: {z0}")
    print(f"Item 1 Z: {z1}")
    
    # Check overlap
    # If Z1 == 0 or Z1 == 5, it overlaps with Item 0 (0-10)
    if z1 < 10:
        print("FAILURE: Item 1 is inside Item 0! (Z < 10)")
    elif z1 >= 10:
        print("SUCCESS: Item 1 is stacked on Item 0 (Z >= 10).")
        
if __name__ == "__main__":
    test_overlap_fallback_forced()
