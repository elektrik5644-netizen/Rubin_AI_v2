import requests
import traceback

def test_smart_dispatcher_with_traceback():
    try:
        print("🔍 Тестируем Smart Dispatcher с подробной диагностикой...")
        
        # Тест обычного запроса
        print("\n📝 Тест обычного запроса:")
        general_response = requests.post(
            "http://localhost:8080/api/chat", 
            json={"message": "hello"}, 
            timeout=5
        )
        print(f"Status: {general_response.status_code}")
        if general_response.status_code == 200:
            print("✅ Обычный запрос работает")
        else:
            print(f"❌ Ошибка: {general_response.text}")
        
        # Тест Arduino Nano запроса
        print("\n🔧 Тест Arduino Nano запроса:")
        arduino_response = requests.post(
            "http://localhost:8080/api/chat", 
            json={"message": "arduino nano pins"}, 
            timeout=5
        )
        print(f"Status: {arduino_response.status_code}")
        print(f"Response: {arduino_response.text}")
        
    except Exception as e:
        print(f"❌ Исключение: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_smart_dispatcher_with_traceback()





