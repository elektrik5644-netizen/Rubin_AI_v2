#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование интеграции логических задач в Rubin AI v2
"""

import requests
import time
import json

def test_logic_integration():
    """Тестирование интеграции логических задач."""
    print("🧠 Тестирование интеграции логических задач")
    print("=" * 60)
    
    base_url = "http://localhost:8106"
    
    # Тест 1: Проверка здоровья
    print("\n1️⃣ Проверка здоровья Logic Tasks API...")
    try:
        response = requests.get(f"{base_url}/api/logic/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Статус: {data['status']}")
            print(f"📝 Сообщение: {data['message']}")
        else:
            print(f"❌ Ошибка здоровья: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Не удалось подключиться: {e}")
        return False
    
    # Тест 2: Получение случайной задачи
    print("\n2️⃣ Получение случайной логической задачи...")
    try:
        response = requests.get(f"{base_url}/api/logic/task", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✅ Задача получена успешно")
                print(f"📝 Тип: {data['task_type']}")
                print(f"📄 Задача: {data['task'][:200]}...")
            else:
                print(f"❌ Ошибка получения задачи: {data['error']}")
        else:
            print(f"❌ HTTP ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 3: Получение задачи определенного типа
    print("\n3️⃣ Получение задачи на доказательства...")
    try:
        response = requests.get(f"{base_url}/api/logic/task?type=доказательства", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✅ Задача на доказательства получена")
                print(f"📄 Задача: {data['task'][:200]}...")
            else:
                print(f"❌ Ошибка: {data['error']}")
        else:
            print(f"❌ HTTP ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 4: Чат с логическими задачами
    print("\n4️⃣ Тестирование чата с логическими задачами...")
    test_messages = [
        "дай задачу",
        "задача на правила",
        "медицинская задача",
        "покажи статистику"
    ]
    
    for message in test_messages:
        try:
            print(f"\n📝 Тест: '{message}'")
            response = requests.post(
                f"{base_url}/api/logic/chat",
                json={'message': message},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    print(f"✅ Ответ получен")
                    print(f"📄 Ответ: {data['response'][:150]}...")
                else:
                    print(f"❌ Ошибка: {data['error']}")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        time.sleep(1)
    
    # Тест 5: Статистика
    print("\n5️⃣ Получение статистики...")
    try:
        response = requests.get(f"{base_url}/api/logic/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                stats = data['statistics']
                print("✅ Статистика получена")
                print(f"🎯 Всего попыток: {stats['total_attempts']}")
                print(f"✅ Решено правильно: {stats['solved_tasks']}")
                print(f"❌ Неправильно: {stats['failed_tasks']}")
                print(f"📈 Успешность: {stats['success_rate']}")
                
                print("\n📚 Доступные типы задач:")
                for name, info in stats['dataset_stats'].items():
                    print(f"  🔹 {name}: {info['count']} задач")
            else:
                print(f"❌ Ошибка статистики: {data['error']}")
        else:
            print(f"❌ HTTP ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 6: Тестирование через Simple Dispatcher
    print("\n6️⃣ Тестирование через Simple Dispatcher...")
    dispatcher_url = "http://localhost:8080"
    
    try:
        response = requests.post(
            f"{dispatcher_url}/api/chat",
            json={'message': 'дай логическую задачу'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Диспетчер успешно перенаправил запрос")
            print(f"📊 Категория: {data.get('category', 'unknown')}")
            print(f"📝 Описание сервера: {data.get('server_description', 'unknown')}")
            
            if 'error' in data:
                print(f"⚠️ Ошибка сервера: {data['error']}")
            else:
                print("✅ Задача получена через диспетчер")
        else:
            print(f"❌ HTTP ошибка диспетчера: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Ошибка диспетчера: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Тестирование интеграции логических задач завершено!")
    print("=" * 60)

if __name__ == '__main__':
    test_logic_integration()



