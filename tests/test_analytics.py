import urllib.request
import json
import time

BASE_URL = "http://localhost:5000"

def test_compare_api():
    print("Testing /api/optimize/compare...")
    payload = {
        "warehouse_id": 1,
        "weights": {"space": 0.5, "accessibility": 0.5, "stability": 0}
    }
    
    try:
        start = time.time()
        req = urllib.request.Request(
            f"{BASE_URL}/api/optimize/compare",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            elapsed = time.time() - start
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if data['success']:
                    print("Success!")
                    results = data['results']
                    for algo, metrics in results.items():
                        print(f"Algorithm: {algo}")
                        print(f"  Query Time: {elapsed:.2f}s")
                        if 'error' in metrics:
                            print(f"  Error: {metrics['error']}")
                        else:
                            print(f"  Execution Time: {metrics.get('time', 'N/A')}")
                            print(f"  Fitness: {metrics.get('fitness', 'N/A')}")
                            print(f"  Accessibility: {metrics.get('accessibility', 'N/A')}")
                else:
                    print("Failed:", data)
            else:
                print(f"Error Status: {response.status}")
            
    except Exception as e:
        print(f"Exception: {e}")
        print("ensure the server is running!")

if __name__ == "__main__":
    test_compare_api()
