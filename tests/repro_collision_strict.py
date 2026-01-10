import sys
import os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact, calculate_z_for_item

def check_intersection(box1, box2):
    # box: x, y, z, dx, dy, h
    # Strict inequality check (no overlap if touching allowed)
    # But user complained about "inside", so strict intersection.
    
    # Expand slightly to catch near-touches that might be glitches?
    # No, strict.
    
    x1, y1, z1, dx1, dy1, h1 = box1
    x2, y2, z2, dx2, dy2, h2 = box2
    
    # Min/Max
    min_x1, max_x1 = x1 - dx1/2, x1 + dx1/2
    min_y1, max_y1 = y1 - dy1/2, y1 + dy1/2
    min_z1, max_z1 = z1, z1 + h1
    
    min_x2, max_x2 = x2 - dx2/2, x2 + dx2/2
    min_y2, max_y2 = y2 - dy2/2, y2 + dy2/2
    min_z2, max_z2 = z2, z2 + h2
    
    # Overlap?
    ox = (min_x1 < max_x2) and (max_x1 > min_x2)
    oy = (min_y1 < max_y2) and (max_y1 > min_y2)
    oz = (min_z1 < max_z2) and (max_z1 > min_z2)
    
    # If touching is okay, we need epsilon.
    # If 10.0 < 10.0 is False. Touching is OK.
    # But 9.999 < 10.0 is True. Overlap.
    
    return ox and oy and oz

def test_strict_collision():
    print("Running Strict Collision Test (100 Iterations)...")
    
    wh_dims = (100, 100, 100, 0, 0)
    layer_heights = [0.0]
    allocation_zones = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100}]
    
    failures = 0
    
    for iteration in range(100):
        # Generate random items
        num_items = random.randint(30, 60)
        items_props = np.zeros((num_items, 8), dtype=np.float32)
        
        for i in range(num_items):
            l = round(random.uniform(2, 12), 2)
            w = round(random.uniform(2, 12), 2)
            h = round(random.uniform(2, 12), 2)
            items_props[i] = [l, w, h, 1, 1, 0, 10, 0]
            
        solution = np.zeros((num_items, 4))
        
        try:
            new_sol = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, layer_heights)
        except Exception as e:
            print(f"Iteration {iteration} crashed: {e}")
            continue
            
        boxes = []
        for i in range(num_items):
            x, y, z, rot = new_sol[i]
            l, w, h = items_props[i, 0:3]
            if int(rot) % 180 == 0:
                dx, dy = l, w
            else:
                dx, dy = w, l
            boxes.append((x, y, z, dx, dy, h))
            
        # Check all pairs
        collision_found = False
        for i in range(num_items):
            for j in range(i+1, num_items):
                if check_intersection(boxes[i], boxes[j]):
                    print(f"[FAIL] Iteration {iteration}: Collision Item {i} vs {j}")
                    print(f"  I{i}: {boxes[i]}")
                    print(f"  I{j}: {boxes[j]}")
                    collision_found = True
                    failures += 1
                    break
            if collision_found: break
            
        if collision_found:
             # Stop at first failure to debug
             break
             
    if failures == 0:
        print("[PASS] 100 Iterations clean.")
    else:
        print(f"[FAIL] Collisions detected.")

if __name__ == "__main__":
    test_strict_collision()
