#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Тест Simple Dispatcher
"""

import requests
import json
import time

def test_simple_dispatcher():
    """Тестирование простого маршрутизатора"""
    print("🧪 Тестирование Simple Dispatcher")
    print("=" * 50)
    
    # Проверка статуса
    try:
        response = requests.get("http://localhost:8080/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Simple Dispatcher работает")
            data = response.json()
            print(f"📊 Статус: {data.get('status', 'unknown')}")
        else:
            print(f"❌ Ошибка статуса: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Не удалось подключиться: {e}")
        return False
    
    # Тест чата
    print("\n💬 Тестирование чата:")
    test_messages = [
        "привет",
        "расскажи про сервопривод",
        "как решить интеграл"
    ]
    
    for message in test_messages:
        try:
            response = requests.post("http://localhost:8080/api/chat", 
                                   json={"message": message}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"📝 '{message}' -> {data.get('routed_to', 'unknown')}")
                if 'error' in data:
                    print(f"   ❌ Ошибка: {data['error']}")
                else:
                    print(f"   ✅ Ответ получен")
            else:
                print(f"❌ Ошибка чата '{message}': {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка чата '{message}': {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Тестирование завершено!")

if __name__ == "__main__":
    test_simple_dispatcher()