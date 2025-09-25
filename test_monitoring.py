#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки системы мониторинга Rubin AI
"""

import requests
import json
import time

def test_health_endpoint():
    """Тестирует health эндпоинт Smart Dispatcher"""
    try:
        response = requests.get("http://localhost:8080/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Smart Dispatcher Health Check:")
            print(f"   Статус: {data.get('status', 'unknown')}")
            print(f"   Здоровых модулей: {data.get('healthy_modules', 0)}/{data.get('total_modules', 0)}")
            
            print("\n📊 Статус модулей:")
            for module, status in data.get('modules', {}).items():
                status_icon = "✅" if status['status'] == 'healthy' else "❌" if status['status'] == 'unreachable' else "⚠️"
                print(f"   {status_icon} {module} (порт {status['port']}): {status['status']}")
                if 'error' in status:
                    print(f"      Ошибка: {status['error']}")
            
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def test_module_directly(module_name, port):
    """Тестирует модуль напрямую"""
    try:
        response = requests.get(f"http://localhost:{port}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {module_name} (порт {port}): здоров")
            return True
        else:
            print(f"⚠️ {module_name} (порт {port}): HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {module_name} (порт {port}): недоступен - {e}")
        return False

def test_chat_functionality():
    """Тестирует функциональность чата"""
    test_messages = [
        "привет",
        "что такое резистор",
        "помощь по директивам"
    ]
    
    print("\n💬 Тестирование чата:")
    for message in test_messages:
        try:
            payload = {"message": message}
            response = requests.post("http://localhost:8080/api/chat", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ '{message}' → {data.get('success', False)}")
                if 'directives_applied' in data:
                    print(f"   📋 Применены директивы: {len(data['directives_applied'])}")
            else:
                print(f"❌ '{message}' → HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ '{message}' → Ошибка: {e}")

def main():
    print("🧪 Тестирование системы мониторинга Rubin AI")
    print("=" * 50)
    
    # Тест health эндпоинта
    print("1. Проверка Health Check эндпоинта:")
    health_ok = test_health_endpoint()
    
    # Тест модулей напрямую
    print("\n2. Прямая проверка модулей:")
    modules = {
        'general': 8085,
        'mathematics': 8086,
        'electrical': 8087,
        'programming': 8088,
        'controllers': 9000
    }
    
    healthy_count = 0
    for module, port in modules.items():
        if test_module_directly(module, port):
            healthy_count += 1
    
    print(f"\n📊 Результат: {healthy_count}/{len(modules)} модулей здоровы")
    
    # Тест чата
    if health_ok:
        test_chat_functionality()
    
    print("\n" + "=" * 50)
    print("Тестирование завершено")

if __name__ == "__main__":
    main()

