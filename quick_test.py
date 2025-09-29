import requests
import json

def quick_test():
    try:
        response = requests.post(
            "http://localhost:8080/api/chat", 
            json={"message": "arduino nano pins"}, 
            timeout=5
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ УСПЕХ!")
                print(f"Категория: {data.get('category')}")
                print(f"Сервер: {data.get('server')}")
                print(f"Ответ: {data.get('response', '')[:200]}...")
            else:
                print("❌ ОШИБКА В ОТВЕТЕ")
                print(data)
        else:
            print(f"❌ HTTP {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Исключение: {e}")

if __name__ == "__main__":
    quick_test()





