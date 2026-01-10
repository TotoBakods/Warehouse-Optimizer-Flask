
import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import HybridOptimizer

class TestHybridOptimizers(unittest.TestCase):
    def setUp(self):
        # Very short run for verification
        self.hybrid = HybridOptimizer(ga_generations=2, eo_iterations=10)
        self.warehouse = {'id': 1, 'length': 100, 'width': 50, 'height': 20, 'levels': 1}
        self.items = [
            {'id': '1', 'length': 10, 'width': 10, 'height': 10, 'can_rotate': 1, 'stackable': 1, 'access_freq': 1, 'category': 'A', 'weight': 10},
            {'id': '2', 'length': 20, 'width': 20, 'height': 10, 'can_rotate': 1, 'stackable': 1, 'access_freq': 1, 'category': 'B', 'weight': 50},
            {'id': '3', 'length': 5, 'width': 5, 'height': 5, 'can_rotate': 0, 'stackable': 0, 'access_freq': 1, 'category': 'C', 'weight': 5}
        ]
        self.weights = {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}

    def test_hybrid_ga_eo_runs(self):
        print("\nTesting Hybrid GA->EO...")
        try:
            sol, fit, t = self.hybrid.optimize(self.items, self.warehouse, self.weights)
            print(f"Hybrid GA->EO Finished. Fitness: {fit}, Items placed: {len(sol)}")
            self.assertGreater(fit, 0, "Fitness should be positive")
            self.assertEqual(len(sol), 3, "All items should be returned in solution")
        except Exception as e:
            self.fail(f"Hybrid GA->EO crashed: {e}")

    def test_hybrid_eo_ga_runs(self):
        print("\nTesting Hybrid EO->GA...")
        try:
            sol, fit, t = self.hybrid.optimize_eo_ga(self.items, self.warehouse, self.weights)
            print(f"Hybrid EO->GA Finished. Fitness: {fit}, Items placed: {len(sol)}")
            self.assertGreater(fit, 0, "Fitness should be positive")
            self.assertEqual(len(sol), 3, "All items should be returned in solution")
        except Exception as e:
            self.fail(f"Hybrid EO->GA crashed: {e}")

if __name__ == '__main__':
    unittest.main()
