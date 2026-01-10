
import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import ExtremalOptimization, GeneticAlgorithm, repair_solution_compact

class TestEOStackability(unittest.TestCase):
    def setUp(self):
        self.eo = ExtremalOptimization(iterations=20)
        self.wh_dims = (100, 100, 20, 0, 0)
        self.warehouse = {'id': 1, 'length': 100, 'width': 100, 'height': 20, 'levels': 1}
        self.valid_z = [0.0]
        
        # 1. Base Item (Unstackable) at 50,50
        # 2. Top Item (Stackable) trying to be placed on top
        # Props: len, wid, hgt, can_rot, stackable, access_freq, weight, category_hash
        self.items_props = np.zeros((2, 8))
        self.items_props[0] = [10, 10, 10, 0, 0, 1, 100, 0] # Unstackable
        self.items_props[1] = [10, 10, 10, 0, 1, 1, 100, 0] # Stackable
        
        self.items = [
            {'id': '1', 'length': 10, 'width': 10, 'height': 10, 'can_rotate': 0, 'stackable': 0, 'access_freq': 1, 'category': 'A'},
            {'id': '2', 'length': 10, 'width': 10, 'height': 10, 'can_rotate': 0, 'stackable': 1, 'access_freq': 1, 'category': 'B'}
        ]

    def test_eo_respects_stackability(self):
        print("\nTesting EO Stackability...")
        
        # Manually force a solution where item 2 is on item 1
        solution = np.zeros((2, 4))
        solution[0] = [50, 50, 0, 0]
        solution[1] = [50, 50, 10, 0] # On top
        
        # Run repair
        # The repair should MOVE item 2 off item 1 because item 1 is unstackable
        repaired = repair_solution_compact(solution.copy(), self.items_props, self.wh_dims, None, self.valid_z)
        
        z2 = repaired[1, 2]
        print(f"Item 2 Z after repair: {z2}")
        
        # Expectation: Z should be 0 (moved to floor) or somewhere else, NOT 10 (on top)
        # Or if it fails to find spot (fallback), it returns high Z?
        # With random probe, it should find a spot on the floor.
        
        # If it stayed at 10, it violated stackability.
        # But wait, repair_solution_compact REBUILDS placement order.
        # It places item 0 then item 1.
        # Item 0 goes to 50,50,0.
        # Item 1 tries 50,50. calculate_z says "blocked".
        # Fallback triggers -> finds random spot.
        # So Z should be 0.
        
        self.assertLess(z2, 9.0, "Item 2 should not be on top of unstackable Item 1")
    
    def test_eo_optimization_run(self):
        print("\nRunning full EO loop...")
        # Should finish without error
        try:
            sol, fit, t = self.eo.optimize(self.items, self.warehouse)
            print(f"EO Finishes with Fitness: {fit}")
            self.assertGreater(fit, 0, "Fitness should be positive")
        except Exception as e:
            self.fail(f"EO crashed: {e}")

if __name__ == '__main__':
    unittest.main()
