
import numpy as np
import optimizer
from optimizer import repair_solution_compact

def test_last_layer_stacking():
    print("Testing Last Layer Stacking...")
    
    # Warehouse 30x30x100 (Tall)
    warehouse_dims = (30, 30, 100, 0, 0)
    # Layers: 0, 10.
    layer_heights = [0.0, 10.0]
    
    # Items 10x10x10.
    # Floor: 3x3=9 items.
    # Layer 10: 9 items.
    # Layer 20: Not defined.
    
    # We send 20 items.
    # 9 should land on Z=0.
    # 9 should land on Z=10.
    # 2 remaining.
    # If Last Layer (10) has ceiling 100, these 2 will land on Z=20 (stacking on Layer 10 items).
    # If this happens, it confirms "Infinite Stacking" on the last layer.
    
    num_items = 20
    items_props = np.zeros((num_items, 8))
    for i in range(num_items):
        items_props[i] = [10, 10, 10, 0, 1, 0, 0, 0] # 10x10x10
        
    solution = np.zeros((num_items, 4))
    for i in range(num_items):
        solution[i] = [15, 15, 0, 0] 
        
    repaired = repair_solution_compact(
        solution, items_props, warehouse_dims, 
        layer_heights=layer_heights
    )
    
    # Count items by Z
    z_counts = {}
    for i in range(num_items):
        z = round(repaired[i, 2], 1)
        z_counts[z] = z_counts.get(z, 0) + 1
        
    print("Z Counts:", flush=True)
    for z in sorted(z_counts.keys()):
        print(f"Z={z}: {z_counts[z]}", flush=True)
        
    # Check if Z=20 exists
    if 20.0 in z_counts:
        print("CONFIRMED: Logic allows stacking on top of Last Layer (Z=20).", flush=True)
    else:
        print("RESULT: Logic prevented stacking above Last Layer.", flush=True)

if __name__ == "__main__":
    test_last_layer_stacking()
