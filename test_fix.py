import requests
import time
import sys

def test_optimization():
    print("Testing optimization API...")
    url = "http://127.0.0.1:5000/api/optimize/ga"
    payload = {
        "warehouse_id": 1,
        "population_size": 10,
        "generations": 5
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Optimization started successfully.")
            # Verify logs
            time.sleep(2)
            return True
        else:
            print("Failed to start optimization.")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if test_optimization():
        sys.exit(0)
    else:
        sys.exit(1)
