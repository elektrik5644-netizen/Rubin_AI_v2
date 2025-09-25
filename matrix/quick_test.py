#!/usr/bin/env python3
"""
Быстрый тест улучшенного сервера Rubin AI
"""

import requests
import json
import time

def test_enhanced_server():
    """Быстрый тест улучшенного сервера"""
    server_url = "http://localhost:8083"
    
    print("🧪 БЫСТРЫЙ ТЕСТ УЛУЧШЕННОГО СЕРВЕРА RUBIN AI")
    print("="*50)
    
    # Тест 1: Health check
    print("1. Проверка health endpoint...")
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Сервер работает: {data.get('message', 'OK')}")
            if 'database' in data:
                print(f"   📊 База данных: {data['database']['messages']} сообщений, {data['database']['documents']} документов")
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест 2: Chat
    print("\n2. Тест чата...")
    try:
        response = requests.post(f"{server_url}/api/chat", 
                               json={"message": "Привет, Enhanced Rubin!"}, 
                               timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Ответ получен: {data.get('response', '')[:50]}...")
            print(f"   📝 Message ID: {data.get('message_id', 'N/A')}")
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест 3: Анализ кода
    print("\n3. Тест анализа кода...")
    try:
        test_code = "def hello():\n    print('Hello, World!')"
        response = requests.post(f"{server_url}/api/code/analyze", 
                               json={"code": test_code, "language": "python"}, 
                               timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Анализ завершен")
            print(f"   📊 Оценка качества: {data.get('quality_score', 'N/A')}")
            print(f"   🔍 Найдено проблем: {len(data.get('issues', []))}")
            print(f"   💡 Рекомендаций: {len(data.get('recommendations', []))}")
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест 4: Загрузка документа
    print("\n4. Тест загрузки документа...")
    try:
        test_content = "Тестовый документ для проверки загрузки в Enhanced Rubin AI"
        response = requests.post(f"{server_url}/api/documents/upload-content", 
                               json={
                                   "filename": "test_document.txt",
                                   "content": test_content,
                                   "category": "test",
                                   "tags": ["test", "enhanced"]
                               }, 
                               timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Документ загружен: {data.get('message', 'OK')}")
            print(f"   📄 Document ID: {data.get('document_id', 'N/A')}")
            print(f"   📏 Размер: {data.get('size', 'N/A')} байт")
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест 5: Статистика документов
    print("\n5. Тест статистики документов...")
    try:
        response = requests.get(f"{server_url}/api/documents/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            print(f"   ✅ Статистика получена")
            print(f"   📊 Всего документов: {stats.get('total_documents', 'N/A')}")
            print(f"   💾 Общий размер: {stats.get('total_size_mb', 'N/A')} MB")
            print(f"   📁 Категории: {stats.get('categories', {})}")
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    # Тест 6: Системная статистика
    print("\n6. Тест системной статистики...")
    try:
        response = requests.get(f"{server_url}/api/system/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('system_stats', {})
            print(f"   ✅ Системная статистика получена")
            print(f"   💬 Всего сообщений: {stats.get('messages', {}).get('total', 'N/A')}")
            print(f"   ⏰ За последний час: {stats.get('messages', {}).get('last_hour', 'N/A')}")
            print(f"   📄 Документов: {stats.get('documents', {}).get('total', 'N/A')}")
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*50)
    print("🎉 БЫСТРЫЙ ТЕСТ ЗАВЕРШЕН!")

if __name__ == "__main__":
    test_enhanced_server()
