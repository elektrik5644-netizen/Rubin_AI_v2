import requests
import json

def test_math_api():
    try:
        print("🧮 Тестируем Mathematics API...")
        
        # Тест health check
        health_response = requests.get("http://localhost:8086/api/health", timeout=5)
        print(f"Health Status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Health Response: {health_response.json()}")
        
        # Тест chat endpoint
        chat_payload = {"message": "2 + 2"}
        chat_response = requests.post(
            "http://localhost:8086/api/chat", 
            json=chat_payload, 
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Chat Status: {chat_response.status_code}")
        print(f"Chat Response: {chat_response.text}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    test_math_api()





