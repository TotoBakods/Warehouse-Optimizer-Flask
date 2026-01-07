
import sys
import os
import time
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import GeneticAlgorithm
from app import optimization_state

# Mock data
items = [{
    'id': 'item_1', 'length': 1.0, 'width': 1.0, 'height': 1.0,
    'can_rotate': 1, 'stackable': 1, 'access_freq': 1, 'category': 'Test', 'weight': 10
}]
warehouse = {
    'id': 1, 'name': 'Test', 'length': 10, 'width': 10, 'height': 5,
    'door_x': 0, 'door_y': 0, 'levels': 1
}
weights = {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}

def update_progress(progress, avg_fitness, best_fitness, best_solution, *args):
    if best_solution:
        print(f"Callback received solution with {len(best_solution)} items.")
    pass

def run_test():
    print("Starting optimization test...")
    optimization_state['running'] = True
    
    try:
        optimizer = GeneticAlgorithm(population_size=10, generations=2)
        best_solution, best_fitness, time_to_best = optimizer.optimize(
            items, warehouse, weights, callback=update_progress, optimization_state=optimization_state
        )
        print("Optimization Success!")
    except Exception as e:
        import traceback
        print(f"Optimization FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
