#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика порта 8084 - что там работает?
"""

import requests
import json

def test_port_8084():
    """Тестирует что работает на порту 8084"""
    print("🔍 ДИАГНОСТИКА ПОРТА 8084")
    print("=" * 40)
    
    # Тест 1: Health check
    try:
        response = requests.get("http://localhost:8084/api/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Сервер: {data.get('name', 'N/A')}")
            print(f"   Версия: {data.get('version', 'N/A')}")
    except Exception as e:
        print(f"❌ Health check: {e}")
    
    # Тест 2: Простой запрос
    print(f"\n🧪 Тест простого запроса...")
    try:
        response = requests.post(
            "http://localhost:8084/api/chat",
            json={"message": "привет"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ответ получен:")
            print(f"   Провайдер: {data.get('provider', 'N/A')}")
            print(f"   Категория: {data.get('category', 'N/A')}")
            print(f"   Нейронная сеть: {data.get('neural_network', 'N/A')}")
            print(f"   Интеграция: {data.get('enhanced_integration', 'N/A')}")
            
            response_text = data.get('response', '')
            preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
            print(f"   Ответ: {preview}")
            
            # Проверяем, это наш исправленный сервер?
            if "🧠" in response_text or "Neural" in data.get('provider', ''):
                print("✅ ЭТО НАШ ИСПРАВЛЕННЫЙ СЕРВЕР!")
            else:
                print("❌ ЭТО СТАРЫЙ СЕРВЕР!")
                
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")
    
    # Тест 3: Программный запрос
    print(f"\n🧪 Тест программного запроса...")
    try:
        response = requests.post(
            "http://localhost:8084/api/chat",
            json={"message": "Сравни C++ и Python", "category": "programming"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            
            if "Programming Knowledge Handler" in data.get('provider', ''):
                print("✅ Программный обработчик работает!")
            elif "C++ vs Python" in response_text:
                print("✅ Получен правильный программный ответ!")
            elif "математика" in response_text.lower():
                print("❌ Получен математический ответ вместо программного!")
            else:
                print("⚠️ Неопределенный результат")
                
            preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
            print(f"   Ответ: {preview}")
            
    except Exception as e:
        print(f"❌ Ошибка программного запроса: {e}")
    
    print(f"\n" + "=" * 40)

if __name__ == "__main__":
    test_port_8084()