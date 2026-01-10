import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact

def check_overlap(box1, box2):
    # box: x, y, z, l, w, h
    x1, y1, z1, l1, w1, h1 = box1
    x2, y2, z2, l2, w2, h2 = box2
    
    # Overlap if intersects in ALL 3 dimensions
    if (x1 < x2 + l2 and x1 + l1 > x2) and \
       (y1 < y2 + w2 and y1 + w1 > y2) and \
       (z1 < z2 + h2 and z1 + h1 > z2):
        return True
    return False

def test_collisions():
    print("Testing Item Collisions...")
    
    # Warehouse: 100x100x100
    wh_dims = (100, 100, 100, 0, 0)
    # Single layer 0-100
    layer_heights = [0.0] 
    allocation_zones = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100}]
    
    # Create 50 mixed items
    num_items = 50
    items_props = np.zeros((num_items, 8), dtype=np.float32)
    np.random.seed(42) # Deterministic
    for i in range(num_items):
        l = np.random.randint(5, 15)
        w = np.random.randint(5, 15)
        h = np.random.randint(5, 15)
        items_props[i] = [l, w, h, 1, 1, 0, 10, 0] # Enable Rotation
        
    # Start with ALL items at 0,0,0
    solution = np.zeros((num_items, 4))
    
    print("Running repair on 50 mixed rotatable items...")
    new_sol = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, layer_heights)
    
    # Check for overlaps
    overlaps = 0
    
    boxes = []
    for i in range(num_items):
        x, y, z, rot = new_sol[i]
        l, w, h = items_props[i, 0:3]
        
        # Adjust dimensions if rotated
        if int(rot) % 180 == 0:
            dx, dy = l, w
        else:
            dx, dy = w, l
            
        # x,y is center
        boxes.append((x - dx/2, y - dy/2, z, dx, dy, h))
        
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if check_overlap(boxes[i], boxes[j]):
                print(f"[FAIL] Overlap detected between Item {i} and {j}!")
                print(f"  Item {i}: {boxes[i]}")
                print(f"  Item {j}: {boxes[j]}")
                overlaps += 1
                
    if overlaps == 0:
        print("[PASS] No overlaps detected.")
    else:
        print(f"[FAIL] Total Overlaps: {overlaps}")

if __name__ == "__main__":
    test_collisions()
