import requests
import traceback

def test_arduino_direct():
    try:
        print("🔍 Тестируем Arduino Nano напрямую через Smart Dispatcher...")
        
        # Тест Arduino Nano запроса
        arduino_response = requests.post(
            "http://localhost:8080/api/chat", 
            json={"message": "arduino nano pins"}, 
            timeout=10
        )
        print(f"Status: {arduino_response.status_code}")
        print(f"Response: {arduino_response.text}")
        
        if arduino_response.status_code == 200:
            print("✅ Arduino Nano работает!")
        else:
            print("❌ Arduino Nano не работает")
        
    except Exception as e:
        print(f"❌ Исключение: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_arduino_direct()





