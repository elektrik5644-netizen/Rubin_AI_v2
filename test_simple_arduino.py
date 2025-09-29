import requests

def test_simple_arduino():
    try:
        # Тест простого Arduino сервера
        response = requests.post(
            "http://localhost:8081/api/chat", 
            json={"message": "arduino nano pins"}, 
            timeout=5
        )
        print(f"Simple Arduino Test - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ ПРОСТОЙ ARDUINO СЕРВЕР РАБОТАЕТ!")
            print(f"Ответ: {data.get('response', '')}")
        else:
            print(f"❌ Ошибка: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка простого сервера: {e}")

if __name__ == "__main__":
    test_simple_arduino()





