#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование API эндпоинтов Rubin AI
"""

import requests
import json
import time

def test_api_endpoints():
    """Тестирование всех API эндпоинтов"""
    base_url = "http://127.0.0.1:8081"
    
    print("ТЕСТИРОВАНИЕ API ЭНДПОИНТОВ RUBIN AI")
    print("=" * 50)
    
    # 1. Тест здоровья системы
    print("\n1. Тест /api/health")
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    # 2. Тест статуса
    print("\n2. Тест /api/status")
    try:
        response = requests.get(f"{base_url}/api/status")
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    # 3. Тест списка серверов
    print("\n3. Тест /api/servers")
    try:
        response = requests.get(f"{base_url}/api/servers")
        print(f"   Статус: {response.status_code}")
        data = response.json()
        print(f"   Всего серверов: {data.get('total_count', 0)}")
        print(f"   Серверы: {list(data.get('servers', {}).keys())}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    # 4. Тест ошибок
    print("\n4. Тест /api/errors")
    try:
        response = requests.get(f"{base_url}/api/errors")
        print(f"   Статус: {response.status_code}")
        data = response.json()
        print(f"   Всего ошибок: {data.get('total_count', 0)}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    # 5. Тест основного чата
    print("\n5. Тест /api/chat")
    try:
        payload = {"message": "Hello! How are you?"}
        response = requests.post(f"{base_url}/api/chat", 
                               json=payload,
                               headers={"Content-Type": "application/json"})
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Ответ: {data.get('response', 'Нет ответа')[:100]}...")
            print(f"   Категория: {data.get('category', 'Неизвестно')}")
            print(f"   Уверенность: {data.get('confidence', 0):.2f}")
        else:
            print(f"   Ошибка: {response.text}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    # 6. Тест нейронного анализа
    print("\n6. Тест /api/neural/analyze")
    try:
        payload = {"question": "Solve equation 2x + 5 = 11"}
        response = requests.post(f"{base_url}/api/neural/analyze", 
                               json=payload,
                               headers={"Content-Type": "application/json"})
        print(f"   Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Ответ: {data.get('response', 'Нет ответа')[:100]}...")
            print(f"   Категория: {data.get('category', 'Неизвестно')}")
            print(f"   Уверенность: {data.get('confidence', 0):.2f}")
        else:
            print(f"   Ошибка: {response.text}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    # 7. Тест системного здоровья
    print("\n7. Тест /api/system/health")
    try:
        response = requests.get(f"{base_url}/api/system/health")
        print(f"   Статус: {response.status_code}")
        print(f"   Ответ: {response.json()}")
    except Exception as e:
        print(f"   Ошибка: {e}")
    
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    test_api_endpoints()
