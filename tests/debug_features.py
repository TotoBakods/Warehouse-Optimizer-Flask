
import json
import urllib.request
import urllib.parse
import time
import traceback

BASE_URL = "http://localhost:5000"
WAREHOUSE_ID = 1

def run_debug():
    print("Starting manual debug...")
    try:
        # Check Server
        print("Checking server status...")
        with urllib.request.urlopen(f"{BASE_URL}/api/warehouses") as response:
            print("Server is UP.")
            
        # Test Delete All
        print("Testing Delete All...")
        req = urllib.request.Request(
            f"{BASE_URL}/api/items/delete_all?warehouse_id={WAREHOUSE_ID}",
            method='DELETE'
        )
        with urllib.request.urlopen(req) as response:
            print(f"Delete All Response: {response.status}")
            print(response.read().decode())
            
        # ADD ITEMS
        print("Adding items for optimization...")
        import uuid
        for i in range(5):
             item = {
                 "id": str(uuid.uuid4()),
                 "warehouse_id": WAREHOUSE_ID, "name": f"Item{i}", "category": "General", 
                 "length": 1, "width": 1, "height": 1, "weight": 10, "quantity": 1
             }
             urllib.request.urlopen(urllib.request.Request(
                 f"{BASE_URL}/api/items", data=json.dumps(item).encode('utf-8'),
                 headers={'Content-Type': 'application/json'}, method='POST'
             ))
        
        # Test Optimization
        print("Testing Optimization...")
        params = {
            "warehouse_id": WAREHOUSE_ID,
            "population_size": 10,
            "generations": 2,
            "weights": {"space": 0.5, "accessibility": 0.5, "stability": 0}
        }
        req = urllib.request.Request(
            f"{BASE_URL}/api/optimize/ga",
            data=json.dumps(params).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req) as response:
            print("Optimization started.")
            
        # Wait for completion with polling
        print("Waiting for optimization to complete...")
        for i in range(15):
             time.sleep(1)
             try:
                 with urllib.request.urlopen(f"{BASE_URL}/api/optimize/status") as response:
                    status = json.loads(response.read().decode())
                    if not status.get('running', False):
                        print(f"Optimization finished after {i+1}s.")
                        break
             except:
                 pass
        
        # Check History
        print("Checking History...")
        with urllib.request.urlopen(f"{BASE_URL}/api/metrics/history?warehouse_id={WAREHOUSE_ID}") as response:
            history = json.loads(response.read().decode())
            if history:
                print("History found.")
                latest = history[0]
                print(f"Latest: Algo={latest.get('algorithm')}, Fitness={latest.get('fitness')}")
                print(f"Time To Best: {latest.get('time_to_best')}")
            else:
                print("No history found.")
                
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
