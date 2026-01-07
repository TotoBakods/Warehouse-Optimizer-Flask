
import sys
import os
import sqlite3
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import GeneticAlgorithm
from database import get_warehouse_config

def verify_optimizer():
    print("Verifying optimizer...")
    
    # manually insert dummy data into items if needed, or query existing?
    # Let's create dummy items list for test to be self-contained and fast
    items = []
    for i in range(100):
        items.append({
            'id': f'item_{i}',
            'length': 1.0, 'width': 1.0, 'height': 1.0,
            'can_rotate': 1, 'stackable': 1, 'access_freq': 1,
            'category': 'Test', 'weight': 10
        })
        
    warehouse = {
        'id': 1, 'name': 'Test',
        'length': 10, 'width': 10, 'height': 5,
        'door_x': 0, 'door_y': 0,
        'levels': 1
    }
    
    print("Running GA optimization...")
    ga = GeneticAlgorithm(population_size=10, generations=2)
    
    try:
        solution, fitness, time_to_best = ga.optimize(items, warehouse)
        print(f"Optimization completed. Fitness: {fitness}")
        print(f"Solution size: {len(solution)}")
        
        if len(solution) != len(items):
            print("ERROR: Solution items count mismatch.")
            sys.exit(1)
            
        # Check output structure
        if not isinstance(solution[0], dict):
            print("ERROR: Output solution should be list of dicts (converted back from numpy)")
            print(f"Got: {type(solution[0])}")
            sys.exit(1)
            
        print("Success! Optimizer returned valid structure.")
        
    except Exception as e:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        with open('error.log', 'w') as f:
            f.write(f"Error Type: {exc_type}\n")
            f.write(f"Error Value: {exc_value}\n")
            tb_list = traceback.format_tb(exc_traceback)
            for line in tb_list:
                f.write(line)
        sys.exit(1)

if __name__ == "__main__":
    verify_optimizer()
