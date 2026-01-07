
import unittest
import json
import urllib.request
import urllib.parse
import time

BASE_URL = "http://localhost:5000"

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        # Ensure we have a warehouse (ID 1)
        self.warehouse_id = 1
        
    def test_1_delete_all(self):
        try:
            print("\nTesting Delete All Items...")
            # Add a dummy item first
            item_data = {
                "warehouse_id": self.warehouse_id,
                "name": "TestItemDelete",
                "length": 1.0, "width": 1.0, "height": 1.0,
                "weight": 10.0, "category": "General",
                "quantity": 1
            }
            
            req = urllib.request.Request(
                f"{BASE_URL}/api/items",
                data=json.dumps(item_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req) as response:
                self.assertEqual(response.status, 201)
                print("  Created dummy item.")

            # Verify item exists
            with urllib.request.urlopen(f"{BASE_URL}/api/items?warehouse_id={self.warehouse_id}") as response:
                items = json.loads(response.read().decode())
                self.assertTrue(len(items) > 0)
                print(f"  Confirmed {len(items)} items exist.")

            # DELETE ALL
            req = urllib.request.Request(
                f"{BASE_URL}/api/items/delete_all?warehouse_id={self.warehouse_id}",
                method='DELETE'
            )
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                self.assertTrue(data['success'])
                print("  Delete All request successful.")

            # Verify empty
            with urllib.request.urlopen(f"{BASE_URL}/api/items?warehouse_id={self.warehouse_id}") as response:
                items = json.loads(response.read().decode())
                self.assertEqual(len(items), 0)
                print("  Confirmed warehouse is empty.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def test_2_optimization_time_to_best(self):
        try:
            print("\nTesting Optimization Time To Best...")
            # Add a few items to optimize
            for i in range(3):
                item_data = {
                    "warehouse_id": self.warehouse_id,
                    "name": f"OptItem{i}",
                    "length": 1.0, "width": 1.0, "height": 1.0,
                    "weight": 10.0, "category": "General",
                    "quantity": 1
                }
                req = urllib.request.Request(
                    f"{BASE_URL}/api/items",
                    data=json.dumps(item_data).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                urllib.request.urlopen(req)

            # Run GA Optimization (Quickly)
            params = {
                "warehouse_id": self.warehouse_id,
                "population_size": 10,
                "generations": 5, # Small generation count for speed
                "weights": {"space": 0.5, "accessibility": 0.5, "stability": 0}
            }
            
            req = urllib.request.Request(
                f"{BASE_URL}/api/optimize/ga",
                data=json.dumps(params).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                self.assertTrue(data['success'])
                print("  Optimization started.")

            # Wait for completion
            max_retries = 20
            for _ in range(max_retries):
                time.sleep(1)
                with urllib.request.urlopen(f"{BASE_URL}/api/optimize/status") as response:
                    status = json.loads(response.read().decode())
                    if not status.get('running', False):
                        print("  Optimization completed.")
                        break
            
            # Check History for time_to_best
            with urllib.request.urlopen(f"{BASE_URL}/api/metrics/history?warehouse_id={self.warehouse_id}") as response:
                 history = json.loads(response.read().decode())
                 self.assertTrue(len(history) > 0)
                 latest = history[0] # Assuming sorted by time (desc) or check last
                 # Actually API returns sorted DESC usually? Let's check logic. script.js assumes it.
                 # but Database usually returns ordered by timestamp DESC.
                 
                 print(f"  Latest Log: Algo={latest['algorithm']}, Fitness={latest['fitness']}")
                 print(f"  Time To Best: {latest.get('time_to_best')}")
                 
                 self.assertIn('time_to_best', latest)
                 self.assertIsNotNone(latest['time_to_best'])
                 # It might be 0.0 if best found immediately, but should be present.
                 self.assertIsInstance(latest['time_to_best'], (int, float))
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

if __name__ == '__main__':
    unittest.main()
