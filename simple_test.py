import requests

url = "http://localhost:8080/api/chat"
payload = {"message": "arduino nano пины"}

try:
    response = requests.post(url, json=payload, timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")