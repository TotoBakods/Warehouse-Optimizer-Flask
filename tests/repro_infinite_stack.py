
import numpy as np
import optimizer
from optimizer import repair_solution_compact

def test_stacking_with_space():
    print("Testing Stacking Behavior with Available Space...")
    
    # Warehouse 30x30x30
    # Floor: 3x3 = 9 items.
    warehouse_dims = (30, 30, 30, 0, 0)
    layer_heights = [0.0, 10.0, 20.0]
    
    # 5 items. All fit on floor.
    num_items = 5
    items_props = np.zeros((num_items, 8))
    for i in range(num_items):
        items_props[i] = [10, 10, 10, 0, 1, 0, 0, 0] # 10x10x10
        
    solution = np.zeros((num_items, 4))
    # All initialized at same spot
    for i in range(num_items):
        solution[i] = [15, 15, 0, 0] 
        
    repaired = repair_solution_compact(
        solution, items_props, warehouse_dims, 
        layer_heights=layer_heights
    )
    
    print("Repaired Solution:")
    print(repaired[:, :3])
    
    # Check max Z
    max_z = np.max(repaired[:, 2])
    print(f"Max Z: {max_z}")
    
    # If Max Z > 0, it means we stacked even though we had floor space?
    # Actually, the logic *prefers* Z=0.
    # It scans for first valid position.
    
    count_z0 = np.sum(repaired[:, 2] < 1.0)
    print(f"Items at Z=0: {count_z0}")
    
    if count_z0 < 4:
        # We expect at least first 4 items to be on floor (placing around center 15,15)
        # 15,15 -> 10x10 takes space.
        # Neighbors: 5,15; 25,15; 15,5; 15,25.
        print("FAILURE: Items stacked unnecessarily!")
    else:
        print("SUCCESS: Items spread out on floor.")

if __name__ == "__main__":
    test_stacking_with_space()
