#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прямой тест Smart Dispatcher
"""

import requests
import json

def test_smart_dispatcher_direct():
    """Тестирует Smart Dispatcher напрямую"""
    
    print("🧪 Прямой тест Smart Dispatcher")
    print("=" * 60)
    
    # Тест health check
    print("1. Проверка здоровья Smart Dispatcher...")
    try:
        response = requests.get('http://localhost:8080/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Smart Dispatcher работает")
            data = response.json()
            print(f"📊 Статус: {data.get('status')}")
            print(f"🔧 Модулей: {data.get('total_modules')}")
            print(f"✅ Здоровых: {data.get('healthy_modules')}")
        else:
            print(f"❌ Ошибка health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return
    
    # Тест классификации
    print("\n2. Тест классификации...")
    test_questions = [
        "что такое фотон?",
        "расскажи про электрон",
        "что такое атом?",
        "объясни квантовую механику"
    ]
    
    for question in test_questions:
        print(f"\n📝 Вопрос: {question}")
        
        try:
            response = requests.post(
                'http://localhost:8080/api/chat',
                json={'message': question},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Ответ получен")
                print(f"📊 Категория: {data.get('category', 'неизвестно')}")
                
                # Проверяем, был ли использован fallback
                if 'fallback_used' in data:
                    print(f"⚠️ Использован fallback: {data['fallback_used']}")
                
                # Проверяем, есть ли информация о сервере
                if 'server_used' in data:
                    print(f"🖥️ Использован сервер: {data['server_used']}")
                
                print(f"💬 Ответ: {data.get('response', 'нет ответа')[:100]}...")
            else:
                print(f"❌ HTTP ошибка: {response.status_code}")
                print(f"📄 Ответ: {response.text}")
                
        except Exception as e:
            print(f"❌ Ошибка запроса: {e}")
        
        print("-" * 40)

if __name__ == '__main__':
    test_smart_dispatcher_direct()



