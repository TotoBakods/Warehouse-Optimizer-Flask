
import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import optimizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import GeneticAlgorithm, get_valid_z_positions

class TestFitnessCalculation(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm(population_size=10, generations=5)
        
        # Mock Warehouse
        self.warehouse = {
            'id': 1,
            'length': 100,
            'width': 50,
            'height': 20,
            'door_x': 0,
            'door_y': 0,
            'levels': 1
        }
        self.wh_dims = (100, 50, 20, 0, 0)
        self.valid_z = [0.0]
        
        # Mock Items (N=5)
        # Props: len, wid, hgt, can_rot, stackable, access_freq, weight, category_hash
        self.items_props = np.array([
            [10, 10, 10, 1, 1, 10, 100, 1234],
            [10, 10, 10, 1, 1, 5, 50, 1234],
            [5, 5, 5, 0, 0, 2, 10, 5678], # Unstackable
            [5, 5, 5, 0, 1, 8, 20, 5678],
            [20, 20, 5, 1, 1, 1, 500, 9999]
        ])
        
    def test_fitness_function_runs_without_error(self):
        print("\nTesting fitness function stability...")
        
        # Create a mock solution (N, 4) -> x, y, z, rotation
        population = np.zeros((1, 5, 4))
        # Place items distinctly
        population[0, 0] = [5, 5, 0, 0]
        population[0, 1] = [20, 5, 0, 0]
        population[0, 2] = [35, 5, 0, 0]
        population[0, 3] = [45, 5, 0, 0]
        population[0, 4] = [70, 20, 0, 90]
        
        weights = {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
        
        try:
            # Test single solution fitness call
            f, su, acc, sta, grp = self.ga.fitness_function_numpy(
                population[0], self.items_props, self.wh_dims, weights, self.valid_z
            )
            
            print(f"Fitness: {f}")
            print(f"Metrics: Space={su}, Access={acc}, Stab={sta}, Group={grp}")
            
            self.assertIsInstance(f, float)
            self.assertIsInstance(su, float)
            self.assertIsInstance(acc, float)
            self.assertIsInstance(sta, float)
            self.assertIsInstance(grp, float)
            
            # Check for NaN
            self.assertFalse(np.isnan(f), "Fitness is NaN")
            
        except NameError as e:
            self.fail(f"NameError detected: {e}")
        except Exception as e:
            self.fail(f"Fitness function crashed: {e}")

    def test_fitness_with_overlapping_items(self):
        print("\nTesting fitness with overlaps (should not crash)...")
        population = np.zeros((1, 5, 4))
        # All at 0,0,0 - massive overlap
        
        try:
            f, su, acc, sta, grp = self.ga.fitness_function_numpy(
                population[0], self.items_props, self.wh_dims, {'space':1}, self.valid_z
            )
            print(f"Overlap Fitness: {f}")
            self.assertLess(f, 0.001, "Fitness should be very low/zero for overlaps")
            
        except Exception as e:
            self.fail(f"Crash on overlap: {e}")

if __name__ == '__main__':
    unittest.main()
