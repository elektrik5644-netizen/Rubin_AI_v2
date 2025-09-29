import requests
import json

def test_arduino_nano_integration():
    """Тест встроенной функциональности Arduino Nano"""
    url = "http://localhost:8080/api/chat"
    
    test_cases = [
        {"message": "arduino nano пины", "expected": "пин"},
        {"message": "arduino функции", "expected": "функция"},
        {"message": "arduino библиотеки", "expected": "библиотека"},
        {"message": "arduino проекты", "expected": "проект"},
        {"message": "arduino ошибка", "expected": "проблема"}
    ]
    
    print("🔧 ТЕСТИРОВАНИЕ ВСТРОЕННОЙ ФУНКЦИОНАЛЬНОСТИ ARDUINO NANO")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Тест {i}: {test_case['message']}")
        
        try:
            response = requests.post(url, json={"message": test_case["message"]}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                server_info = data.get('server', 'неизвестно')
                response_text = data.get('response', '')
                
                print(f"✅ Статус: {response.status_code}")
                print(f"🖥️ Сервер: {server_info}")
                print(f"📝 Ответ: {response_text[:100]}...")
                
                # Проверяем, что ответ содержит ожидаемое содержимое
                if test_case['expected'].lower() in response_text.lower():
                    print(f"✅ Тест пройден: найдено '{test_case['expected']}'")
                else:
                    print(f"❌ Тест не пройден: не найдено '{test_case['expected']}'")
                    
                # Проверяем, что это встроенный модуль
                if "встроенный модуль" in server_info:
                    print("✅ Встроенный модуль работает корректно")
                else:
                    print(f"❌ Ошибка: используется внешний сервер - {server_info}")
                    
            else:
                print(f"❌ Ошибка HTTP: {response.status_code}")
                print(f"Ответ: {response.text}")
                
        except Exception as e:
            print(f"❌ Исключение: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_arduino_nano_integration()





