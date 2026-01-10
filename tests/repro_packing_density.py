import sys
import os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact

def test_packing_density():
    print("Testing Packing Density...")
    
    # Warehouse: 100x100x100
    # Two layers: 0-50, 50-100
    wh_dims = (100, 100, 100, 0, 0)
    layer_heights = [0.0, 50.0]
    allocation_zones = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100}]
    
    # Generate items that perfectly fill layer 1 if packed correctly.
    # 100 items of 10x10x50. Area = 100. Total Area = 10000. Warehouse Area = 10000.
    # If perfect, all fit in Layer 1.
    # But let's make it a bit realistic/harder. 
    # Mix of large and small.
    # 4 items of 50x50x40. (Area 2500*4 = 10000). Fits perfectly if optimized.
    # But if small items get there first...
    
    items_props = np.zeros((8, 8), dtype=np.float32)
    # 4 Large
    for i in range(4):
        items_props[i] = [50, 50, 40, 0, 1, 0, 100, 0] # 50x50
        
    # 4 Small (distraction) - wait, if I fill perfectly, I can't add more.
    # Let's try 3 Large (7500 area) + Many Small to fill rest.
    
    items_props = np.zeros((10, 8), dtype=np.float32)
    # 3 Large (50x50) = 7500 area.
    for i in range(3):
         items_props[i] = [50, 50, 40, 0, 1, 0, 100, 0]
    
    # Rest 2500 area. 25 items of 10x10.
    # Total items = 3 + 25 = 28.
    items_props = np.zeros((28, 8), dtype=np.float32)
    for i in range(3):
        items_props[i] = [50, 50, 40, 0, 1, 0, 100, 0]
    for i in range(3, 28):
        items_props[i] = [10, 10, 40, 0, 1, 0, 10, 0]
        
    # Initial solution: Randomize order by putting random Z
    # Simulate a "Bad" GA individual that puts small items at Z=0 and Large items at Z=10
    solution = np.zeros((28, 4))
    
    # Put small items (indices 3+) at Low Z
    for i in range(3, 28):
        solution[i] = [0, 0, 0, 0]
    
    # Put large items (indices 0-3) at High Z
    for i in range(3):
        solution[i] = [0, 0, 100, 0]
        
    # Run Repair
    print("Running repair with Bad GA Input (Small First, Large Last)...")
    new_sol = repair_solution_compact(solution.copy(), items_props, wh_dims, allocation_zones, layer_heights)
    
    # Analyze
    layer1_count = 0
    layer2_count = 0
    layer1_area = 0
    
    print("\nResults:")
    for i in range(28):
        z = new_sol[i, 2]
        l = items_props[i, 0]
        w = items_props[i, 1]
        
        if z < 10: # Layer 1
            layer1_count += 1
            layer1_area += l * w
        elif z >= 50: # Layer 2
            layer2_count += 1
            print(f"Item {i} (Dim {l}x{w}) pushed to Layer 2 (Z={z})")
            
    print(f"\nLayer 1 Utilization: {layer1_area}/10000 ({layer1_area/100:.1f}%)")
    print(f"Items in Layer 2: {layer2_count}")
    
    if layer2_count > 0:
        print("[FAIL] Optimization failed to pack all items in Layer 1 despite theoretical fit.")
    else:
        print("[PASS] Perfect Packing achieved.")

if __name__ == "__main__":
    test_packing_density()
