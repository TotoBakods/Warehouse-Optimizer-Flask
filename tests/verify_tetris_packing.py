
import numpy as np
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import repair_solution_compact, calculate_z_for_item

def test_tetris_repair():
    print("Testing repair_solution_compact...")
    
    # Define 3 simple items
    # Item 1: 10x10x10
    # Item 2: 10x10x10
    # Item 3: 20x10x10 (can rotate)
    
    num_items = 3
    # items_props cols: len, wid, hgt, can_rot, stackable, ...
    items_props = np.zeros((num_items, 8))
    items_props[0] = [10, 10, 10, 0, 1, 0, 0, 0] # 10x10x10, no rot
    items_props[1] = [10, 10, 10, 0, 1, 0, 0, 0] # 10x10x10, no rot
    items_props[2] = [20, 10, 10, 1, 1, 0, 0, 0] # 20x10x10, CAN rot
    
    # Create a "bad" initial solution (floating, scattered)
    solution = np.zeros((num_items, 4))
    solution[0] = [50, 50, 100, 0] 
    solution[1] = [80, 80, 200, 0]
    solution[2] = [20, 20, 300, 0] # Has wrong rotation initially? 
    
    # Run repair
    new_sol = repair_solution_compact(solution, items_props)
    
    print("\noptimized solution:")
    for i in range(num_items):
        print(f"Item {i}: x={new_sol[i,0]:.2f}, y={new_sol[i,1]:.2f}, z={new_sol[i,2]:.2f}, rot={new_sol[i,3]}")
        
    # Validation logic
    # Item 0 should be at 5, 5, 0 (centered at 5,5 with dims 10,10) -> min 0,0
    # Item 1 should be at 15, 5, 0 (next to Item 0?) or 5, 15, 0?
    # Logic: sorted by dist.
    # Init:
    # 0: 50+50+100k
    # 1: 80+80+200k
    # 2: 20+20+300k
    # Sort order will likely be 0, 1, 2 depending on original Z.
    
    # Expected behavior:
    # Item 0 goes to 0,0,0 (center 5,5,0)
    # Item 1: slides to X=10? Center 15,5,0.
    # Item 2: 20x10x10. If it rotates to 10x20x10?
    # If it stays 20x10, it needs 20 width.
    # Current space after 0 and 1:
    # (0,0) to (10,10) filled.
    # (10,0) to (20,10) filled.
    # Item 2 could go to (0,10)?
    
    # Let's see what happens.
    
    # Validated function behavior. Now test integration.
    pass

def test_hybrid_integration():
    print("\nTesting HybridOptimizer integration...")
    from optimizer import HybridOptimizer
    
    # 5 items
    items = []
    for i in range(5):
        items.append({
            'id': f'item_{i}',
            'length': 10, 'width': 10, 'height': 10,
            'can_rotate': True, 'stackable': True,
            'access_freq': 1, 'weight': 10
        })
        
    warehouse = {'id': 1, 'length': 100, 'width': 100, 'height': 100}
    
    hybrid = HybridOptimizer(ga_generations=2, eo_iterations=2)
    # Use standard optimize (GA->EO)
    sol, fit, t = hybrid.optimize(items, warehouse)
    
    print("Hybrid GA->EO solution:")
    for item in sol:
        print(item)
        
    # Use EO->GA
    sol2, fit2, t2 = hybrid.optimize_eo_ga(items, warehouse)
    print("\nHybrid EO->GA solution:")
    for item in sol2:
        print(item)

if __name__ == "__main__":
    test_tetris_repair()
    test_hybrid_integration()
