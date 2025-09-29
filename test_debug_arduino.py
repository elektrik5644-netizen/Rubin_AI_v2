import requests

def test_debug_arduino():
    try:
        print("🔍 Тестируем отладочный Arduino сервер...")
        response = requests.post(
            "http://localhost:8084/api/chat", 
            json={"message": "arduino nano pins"}, 
            timeout=5
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ ОТЛАДОЧНЫЙ ARDUINO СЕРВЕР РАБОТАЕТ!")
            print(f"Ответ: {data.get('response', '')}")
        else:
            print(f"❌ Ошибка: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка отладочного сервера: {e}")

if __name__ == "__main__":
    test_debug_arduino()





