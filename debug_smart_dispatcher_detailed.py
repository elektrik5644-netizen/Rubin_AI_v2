import requests
import traceback

def debug_smart_dispatcher_detailed():
    try:
        print("🔍 Детальная диагностика Smart Dispatcher...")
        response = requests.post(
            "http://localhost:8080/api/chat", 
            json={"message": "arduino nano pins"}, 
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Category: {data.get('category')}")
            print(f"Server: {data.get('server')}")
            print(f"Response: {data.get('response', '')[:200]}...")
        
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_smart_dispatcher_detailed()





