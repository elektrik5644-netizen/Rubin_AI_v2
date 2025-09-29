import requests

def test_arduino_debug():
    url = "http://localhost:8080/api/chat"
    payload = {"message": "arduino nano пины"}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Category: {data.get('category')}")
            print(f"Server: {data.get('server')}")
            print(f"Response: {data.get('response', '')[:200]}...")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_arduino_debug()





