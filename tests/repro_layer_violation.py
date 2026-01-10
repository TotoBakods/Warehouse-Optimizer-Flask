import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact

def test_layer_violation():
    print("Testing Layer Violations...")
    
    # Warehouse: 100x100x100
    # Two layers: 0-50, 50-100
    wh_dims = (100, 100, 100, 0, 0)
    layer_heights = [0.0, 50.0]
    allocation_zones = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100}]
    
    # Check 1: Item too tall for any layer
    # Item: 60 height.
    # Layer 1: 50 height. Layer 2: 50 height.
    # Should NOT fit.
    items_props = np.zeros((1, 8), dtype=np.float32)
    items_props[0] = [10, 10, 60.0, 0, 1, 0, 0, 0] 
    
    solution = np.zeros((1, 4))
    solution[0] = [50, 50, 0, 0] # Initial guess
    
    print("\nCase 1: Item Height 60 in 50-height layers")
    new_sol = repair_solution_compact(solution.copy(), items_props, wh_dims, allocation_zones, layer_heights)
    z = new_sol[0, 2]
    h = 60.0
    print(f"Placed at Z={z}, Top={z+h}")
    
    if (z < 0.1 and z+h > 50.1) or (z > 49.9 and z+h > 100.1):
         print("[FAIL] Item violates layer ceiling!")
         if z < 0.1: print("  - Violated Layer 1 ceiling (50)")
         if z > 49.9: print("  - Violated Layer 2 ceiling (100)")
    else:
         # It might be unplaced (z=0 but maybe handled?) or placed weirdly?
         # If it's placed at 0 and violates, it fails.
         # If it returns expected behavior (unplaced/error), good.
         # But repair usually forces placement.
         pass
         
    # Check 2: Stacking Violation in Fallback
    # 2 Items. Item 1 (40h) forced to 0,0. Item 2 (20h).
    # Layer 1 (0-50).
    # Item 1 takes 0-40.
    # Item 2 on top: 40-60. Violates Layer 1 ceiling (50).
    # Item 2 should go to Layer 2 (50-70).
    
    print("\nCase 2: Stacking causing violation")
    items_props_2 = np.zeros((2, 8), dtype=np.float32)
    items_props_2[0] = [100, 100, 40.0, 0, 1, 0, 0, 0] # Big base item filling floor
    items_props_2[1] = [10, 10, 20.0, 0, 1, 0, 0, 0]  # Small item on top
    
    sol2 = np.zeros((2, 4))
    sol2[0] = [50, 50, 0, 0]
    sol2[1] = [50, 50, 10, 0]
    
    new_sol2 = repair_solution_compact(sol2.copy(), items_props_2, wh_dims, allocation_zones, layer_heights)
    
    z1 = new_sol2[0, 2]
    z2 = new_sol2[1, 2]
    print(f"Item 1 (40h): Z={z1}, Top={z1+40}")
    print(f"Item 2 (20h): Z={z2}, Top={z2+20}")
    
    if z2 > z1 + 39: # If stacked on top
        if z2 + 20 > 50.1 and z2 < 49.9:
            print("[FAIL] Item 2 stacked on Item 1 violates Layer 1 ceiling (40+20 > 50)")
        elif z2 >= 50.0:
            print("[PASS] Item 2 correctly jumped to Layer 2")
        else:
             print("[PASS] Item 2 placed elsewhere safely")
    else:
         print("[PASS] Items not stacked")

if __name__ == "__main__":
    test_layer_violation()
