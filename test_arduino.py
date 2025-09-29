import requests
import json

# Тест встроенной функциональности Arduino Nano
def test_arduino_nano():
    url = "http://localhost:8080/api/chat"
    payload = {"message": "arduino nano пины"}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Module: {data.get('module')}")
            print(f"Response: {data.get('response')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_arduino_nano()





