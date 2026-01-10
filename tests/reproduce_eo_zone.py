import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer import ExtremalOptimization, GeneticAlgorithm, repair_solution_compact

def test_eo_multiple_zones():
    print("Testing Extremal Optimization with Multiple Allocation Zones...")
    
    # 1. Setup Warehouse and Items
    warehouse = {
        'id': 1, 'length': 100, 'width': 100, 'height': 100,
        'door_x': 0, 'door_y': 0
    }
    
    # Define 2 Zones:
    # Zone 1: Left side (0-40)
    # Zone 2: Right side (60-100)
    zones = [
        {'id': 1, 'name': 'Zone 1', 'x1': 0, 'y1': 0, 'x2': 40, 'y2': 100, 'z1': 0, 'z2': 100, 'zone_type': 'allocation'},
        {'id': 2, 'name': 'Zone 2', 'x1': 60, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100, 'zone_type': 'allocation'}
    ]
    
    # Mock database.get_exclusion_zones since EO calls it
    # We need to monkeypatch it or just rely on the fact that we can't easily mock DB calls here without setup.
    # Actually, EO calls `get_exclusion_zones(warehouse['id'])`. 
    # For this test, I'll modify the EO call or just instantiate EO and bypass the DB lookups if possible?
    # EO.optimize calls `get_exclusion_zones` inside.
    
    # Hack: We can assume validation/reproduction might need the DB. 
    # OR, we can just instantiate the class and test the mutation logic? 
    # Actually, let's run a mocked optimization flow.
    
    # Let's trust that I can just run it if I mock `get_exclusion_zones`.
    import optimizer
    optimizer.get_exclusion_zones = lambda wid: zones
    optimizer.get_valid_z_positions = lambda w: [0.0]
    
    # Create items
    items = []
    for i in range(20):
        items.append({
            'id': f'item_{i}', 'name': 'Box',
            'length': 10, 'width': 10, 'height': 10,
            'can_rotate': 1, 'stackable': 1, 'access_freq': 1, 'weight': 10,
            'category': 'General'
        })
        
    eo = ExtremalOptimization(iterations=50) # Short run
    
    print("Running EO optimize...")
    try:
        solution, fitness, _ = eo.optimize(items, warehouse)
    except Exception as e:
        print(f"EO Crashed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Analyze Results
    in_zone_1 = 0
    in_zone_2 = 0
    outside = 0
    
    for item in solution:
        x = item['x']
        # center x is what we have? solution contains centers? 
        # API returns centers.
        if 0 <= x <= 40: in_zone_1 += 1
        elif 60 <= x <= 100: in_zone_2 += 1
        else: outside += 1
            
    print(f"\nSummary:")
    print(f"Items in Zone 1: {in_zone_1}")
    print(f"Items in Zone 2: {in_zone_2}")
    print(f"Items Outside:   {outside}")
    
    if in_zone_1 > 0 and in_zone_2 > 0:
        print("\nSUCCESS: Items distributed.")
    else:
        print("\nFAILURE: Items clustered in one zone.")

if __name__ == "__main__":
    test_eo_multiple_zones()
