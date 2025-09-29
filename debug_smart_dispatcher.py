import requests
import traceback

def debug_smart_dispatcher():
    try:
        print("🔍 Тестируем Smart Dispatcher с подробной диагностикой...")
        response = requests.post(
            "http://localhost:8080/api/chat", 
            json={"message": "arduino nano pins"}, 
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_smart_dispatcher()





