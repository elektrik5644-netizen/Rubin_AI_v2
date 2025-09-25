#!/usr/bin/env python3
"""
Скрипт для запуска сервера и тестирования подключения
"""

import subprocess
import time
import requests
import sys
import os

def start_server():
    """Запуск сервера в фоновом режиме"""
    print("🚀 Запуск Rubin AI сервера...")
    
    try:
        # Запуск сервера в фоновом режиме
        process = subprocess.Popen(
            [sys.executable, 'minimal_rubin_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        
        print("✅ Сервер запущен в фоновом режиме")
        print("⏳ Ожидание запуска сервера...")
        
        # Ждем запуска сервера
        time.sleep(3)
        
        return process
        
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        return None

def test_connection():
    """Тестирование подключения к серверу"""
    print("\n🧪 Тестирование подключения...")
    
    try:
        # Тест health endpoint
        response = requests.get('http://localhost:8083/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Сервер работает!")
            print(f"📊 Статус: {data.get('message', 'OK')}")
            return True
        else:
            print(f"❌ HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def test_chat():
    """Тестирование чата"""
    print("\n💬 Тестирование чата...")
    
    try:
        response = requests.post(
            'http://localhost:8083/api/chat',
            json={'message': 'Привет, Rubin!'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Чат работает!")
            print(f"🤖 Ответ: {data.get('response', 'Нет ответа')[:50]}...")
            return True
        else:
            print(f"❌ HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка чата: {e}")
        return False

def main():
    """Главная функция"""
    print("🔧 ЗАПУСК И ТЕСТИРОВАНИЕ RUBIN AI")
    print("="*50)
    
    # Запуск сервера
    process = start_server()
    
    if process is None:
        print("❌ Не удалось запустить сервер")
        return
    
    # Тестирование подключения
    if test_connection():
        print("\n🎉 Сервер успешно запущен и работает!")
        
        # Тестирование чата
        if test_chat():
            print("\n✅ Все тесты пройдены успешно!")
            print("🌐 Rubin AI готов к использованию!")
            print("📱 Откройте RubinIDE.html в браузере")
        else:
            print("\n⚠️ Сервер работает, но чат не отвечает")
    else:
        print("\n❌ Сервер не отвечает")
        print("💡 Попробуйте запустить сервер вручную:")
        print("   python minimal_rubin_server.py")
    
    print("\n" + "="*50)
    print("🛑 Для остановки сервера закройте окно или нажмите Ctrl+C")

if __name__ == "__main__":
    main()
