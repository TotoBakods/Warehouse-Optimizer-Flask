
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import time
from optimizer import GeneticAlgorithm, fitness_function_numpy, create_random_solution_array, init_worker

class TestMemoryOptimization(unittest.TestCase):
    def test_optimization_run(self):
        print("Testing Genetic Algorithm with Memory Optimizations...")
        
        # Mock Data
        num_items = 500
        items = [{'id': i, 'length': 10, 'width': 10, 'height': 10, 
                  'weight': 1, 'can_rotate': True, 'stackable': True, 
                  'access_freq': 1, 'category': 'CatA'} for i in range(num_items)]
        
        warehouse = {
            'id': 1, 'length': 100, 'width': 100, 'height': 100,
            'door_x': 0, 'door_y': 0
        }
        
        # Setup GA
        ga = GeneticAlgorithm(population_size=20, generations=5)
        
        start_time = time.time()
        best_sol, best_fit, ttb = ga.optimize(items, warehouse)
        end_time = time.time()
        
        print(f"Optimization finished in {end_time - start_time:.4f}s")
        print(f"Best Fitness: {best_fit}")
        
        self.assertTrue(len(best_sol) == num_items)
        self.assertGreater(best_fit, -100) # Should be valid

    def test_float32_usage(self):
        # Verify that create_random_solution_array returns float32
        num_items = 10
        wh_dims = (100, 100, 100, 0, 0)
        items_props = np.zeros((num_items, 8), dtype=np.float32)
        items_props[:, 0] = 10 # len
        items_props[:, 1] = 10 # wid
        items_props[:, 2] = 10 # hgt
        
        # We need to shim globals or pass args
        # Let's pass args explicitly to test that path
        sol = create_random_solution_array(num_items, wh_dims, items_props, None)
        
        print(f"Solution Dtype: {sol.dtype}")
        self.assertTrue(sol.dtype == np.float32)

if __name__ == '__main__':
    unittest.main()
