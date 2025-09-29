#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прямой тест Physics Server
"""

import requests
import json

def test_physics_server_direct():
    """Тестирует Physics Server напрямую"""
    
    print("🧪 Прямой тест Physics Server")
    print("=" * 60)
    
    # Тест health check
    print("1. Проверка здоровья сервера...")
    try:
        response = requests.get('http://localhost:8110/api/physics/health', timeout=5)
        if response.status_code == 200:
            print("✅ Physics Server работает")
            data = response.json()
            print(f"📊 Статус: {data.get('status')}")
            print(f"📚 Понятий в базе: {data.get('knowledge_base', {}).get('total_concepts', 0)}")
        else:
            print(f"❌ Ошибка health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return
    
    # Тест объяснения фотона
    print("\n2. Тест объяснения фотона...")
    try:
        response = requests.post(
            'http://localhost:8110/api/physics/explain',
            json={'concept': 'фотон'},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Ответ получен")
            print(f"📊 Успех: {data.get('success')}")
            if data.get('success'):
                print(f"💡 Понятие: {data.get('concept')}")
                print(f"📝 Объяснение: {data.get('explanation', '')[:100]}...")
            else:
                print(f"❌ Ошибка: {data.get('error')}")
        else:
            print(f"❌ HTTP ошибка: {response.status_code}")
            print(f"📄 Ответ: {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")
    
    # Тест списка понятий
    print("\n3. Тест списка понятий...")
    try:
        response = requests.get('http://localhost:8110/api/physics/concepts', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Список понятий получен")
            concepts = data.get('concepts', [])
            print(f"📚 Доступные понятия: {', '.join(concepts)}")
        else:
            print(f"❌ Ошибка получения понятий: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")

if __name__ == '__main__':
    test_physics_server_direct()



