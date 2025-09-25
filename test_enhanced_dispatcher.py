#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Тест нового Enhanced Smart Dispatcher
"""

import requests
import json
import time

def test_enhanced_dispatcher():
    """Тестирование нового маршрутизатора"""
    print("🧪 Тестирование Enhanced Smart Dispatcher")
    print("=" * 50)
    
    # Проверка здоровья
    try:
        response = requests.get("http://localhost:8080/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Enhanced Dispatcher работает")
            data = response.json()
            print(f"📊 Статус: {data.get('status', 'unknown')}")
            print(f"🧠 Нейронный роутер: {data.get('neural_router', 'unknown')}")
            print(f"📊 Трекер ошибок: {data.get('error_tracker', 'unknown')}")
        else:
            print(f"❌ Ошибка здоровья: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Не удалось подключиться: {e}")
        return False
    
    # Тест нейронного анализа
    print("\n🧠 Тестирование нейронного анализа:")
    test_messages = [
        "привет",
        "расскажи про сервопривод",
        "как решить интеграл",
        "напиши код на Python",
        "что такое нейросеть"
    ]
    
    for message in test_messages:
        try:
            response = requests.post("http://localhost:8080/api/neural/analyze", 
                                   json={"message": message}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"📝 '{message}' -> {data.get('category', 'unknown')} (уверенность: {data.get('confidence', 0):.2f})")
                print(f"   🎯 Предложенный сервер: {data.get('suggested_server', 'unknown')}")
            else:
                print(f"❌ Ошибка анализа '{message}': {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка анализа '{message}': {e}")
    
    # Тест чата
    print("\n💬 Тестирование чата:")
    for message in test_messages[:3]:  # Тестируем только первые 3
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
    
    # Проверка ошибок
    print("\n🚨 Проверка ошибок системы:")
    try:
        response = requests.get("http://localhost:8080/api/errors?limit=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            errors = data.get('errors', [])
            if errors:
                print(f"📊 Найдено ошибок: {len(errors)}")
                for error in errors[-3:]:  # Последние 3
                    print(f"   {error.get('type', 'unknown')}: {error.get('message', 'unknown')}")
            else:
                print("✅ Ошибок не найдено")
        else:
            print(f"❌ Ошибка получения ошибок: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка получения ошибок: {e}")
    
    # Проверка здоровья системы
    print("\n💚 Проверка здоровья системы:")
    try:
        response = requests.get("http://localhost:8080/api/system/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            health_percentage = data.get('health_percentage', 0)
            healthy_servers = data.get('healthy_servers', 0)
            total_servers = data.get('total_servers', 0)
            
            health_icon = "🟢" if health_percentage >= 80 else "🟡" if health_percentage >= 50 else "🔴"
            print(f"{health_icon} Здоровье системы: {health_percentage:.1f}%")
            print(f"📊 Серверов: {healthy_servers}/{total_servers}")
        else:
            print(f"❌ Ошибка здоровья системы: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка здоровья системы: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Тестирование завершено!")

if __name__ == "__main__":
    test_enhanced_dispatcher()



