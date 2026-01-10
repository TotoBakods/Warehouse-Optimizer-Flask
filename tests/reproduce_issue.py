import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact

def test_stagnation_and_zone_violation():
    print("Testing Stagnation and Zone Violation...")
    
    # Setup
    num_items = 5
    # items_props: len, wid, hgt, can_rotate, stackable, access_freq, weight, category
    items_props = np.zeros((num_items, 8), dtype=np.float32)
    items_props[:, 0] = 10.0 # Length
    items_props[:, 1] = 10.0 # Width
    items_props[:, 2] = 10.0 # Height
    items_props[:, 3] = 1.0  # Can Rotate
    items_props[:, 4] = 1.0  # Stackable
    
    # Warehouse: 100x100x100
    wh_dims = (100, 100, 100, 0, 0)
    
    # Allocation Zone: Small corner 0-20, 0-20 (Cannot fit all 5 items if they are 10x10 each and placed loosely, but should fit easily packed)
    # Actually, 5 items of 10x10 is 500 area. Zone 20x20 is 400 area.
    # So one item MUST fail to fit in the zone.
    allocation_zones = [{'x1': 0, 'y1': 0, 'x2': 20, 'y2': 20, 'z1': 0, 'z2': 100}]
    
    # Create two different random inputs
    input1 = np.random.rand(num_items, 4) * 100
    input2 = np.random.rand(num_items, 4) * 100
    
    # Run repair
    repaired1 = repair_solution_compact(input1.copy(), items_props, wh_dims, allocation_zones)
    repaired2 = repair_solution_compact(input2.copy(), items_props, wh_dims, allocation_zones)
    
    # Check Stagnation (Are outputs identical despite different inputs?)
    # Since repair sorts by volume (all equal here?) 
    # If volumes are equal, sort might be stable or index based.
    # Let's vary volumes slightly to force a sort order if it depends on volume.
    items_props[0, 0] = 11.0
    items_props[1, 0] = 12.0
    # Now volumes are different.
    
    repaired1 = repair_solution_compact(input1.copy(), items_props, wh_dims, allocation_zones)
    repaired2 = repair_solution_compact(input2.copy(), items_props, wh_dims, allocation_zones)
    
    if np.allclose(repaired1, repaired2):
        print("[FAIL] Stagnation Detected: Repair function is deterministic and ignores input solution structure.")
    else:
        print("[PASS] Repair function produces different outputs for different inputs.")
        print(repaired1)
        print(repaired2)

    # Check Zone Violation
    # All items should be within 0-20, 0-20.
    # If item 1 failed to fit (area 400 vs total item area 121+144+100*3 > 400?), it might spill over.
    # Zone area: 400. Items: 12.1*10 + 10... wait dims.
    # Item 0: 11x10 = 110
    # Item 1: 12x10 = 120
    # Item 2,3,4: 10x10 = 100 each.
    # Total area: 110+120+300 = 530.
    # Zone Area = 20x20 = 400.
    # So items MUST overflow.
    
    print("\nChecking Zone Violations...")
    violations = 0
    for i in range(num_items):
        r = repaired1[i]
        x, y = r[0], r[1]
        
        # Check against zone [0, 20]
        # x is center. Dims are in items_props.
        # But repair might rotate.
        # We just check center for simplicity, or bounds?
        # repair places center x,y.
        if x > 25 or y > 25: # Generous buffer
            print(f"Item {i} at {x},{y} is OUTSIDE zone (0-20). Violation!")
            violations += 1
            
    if violations > 0:
        print(f"[FAIL] Zone Violation Detected: {violations} items placed outside restricted zone.")
    else:
        print("[PASS] All items contained in zone (or unplaced).")

if __name__ == "__main__":
    test_stagnation_and_zone_violation()
