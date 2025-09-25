#!/usr/bin/env python3
"""
Запуск простой системы Rubin AI
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def start_simple_rubin():
    """Запуск простой системы Rubin AI"""
    
    print("🚀 ЗАПУСК ПРОСТОЙ СИСТЕМЫ RUBIN AI")
    print("=" * 50)
    
    # Проверяем, что файлы существуют
    required_files = [
        'simple_chat_system.py',
        'simple_rubin_interface.html'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Файл {file} не найден!")
            return False
    
    print("✅ Все файлы найдены")
    
    # Запускаем сервер
    print("🌐 Запуск сервера на порту 8085...")
    try:
        # Запускаем сервер в отдельном процессе
        server_process = subprocess.Popen([
            sys.executable, 'simple_chat_system.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Ждем немного для запуска
        time.sleep(3)
        
        # Проверяем, что сервер запустился
        if server_process.poll() is None:
            print("✅ Сервер запущен успешно")
            
            # Открываем браузер
            def open_browser():
                time.sleep(2)
                webbrowser.open('file://' + os.path.abspath('simple_rubin_interface.html'))
            
            browser_thread = Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            print("🌐 Открываем интерфейс в браузере...")
            print("\n📡 Система готова к работе!")
            print("🔗 API: http://localhost:8085")
            print("🌐 Интерфейс: simple_rubin_interface.html")
            print("\n💡 Для остановки нажмите Ctrl+C")
            
            try:
                # Ждем завершения процесса
                server_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Остановка сервера...")
                server_process.terminate()
                server_process.wait()
                print("✅ Сервер остановлен")
            
            return True
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Ошибка запуска сервера:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_system():
    """Тестирование системы"""
    print("\n🧪 ТЕСТИРОВАНИЕ СИСТЕМЫ")
    print("=" * 30)
    
    try:
        import requests
        
        # Тест здоровья
        print("1. Тест здоровья системы...")
        response = requests.get('http://localhost:8085/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Система здорова")
        else:
            print(f"❌ Ошибка здоровья: {response.status_code}")
            return False
        
        # Тест чата
        print("2. Тест чата...")
        response = requests.post('http://localhost:8085/api/chat', 
                               json={'message': 'привет'}, 
                               timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Чат работает: {data['response'][:50]}...")
        else:
            print(f"❌ Ошибка чата: {response.status_code}")
            return False
        
        # Тест математики
        print("3. Тест математики...")
        response = requests.post('http://localhost:8085/api/chat', 
                               json={'message': '2+3'}, 
                               timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Математика работает: {data['response'][:50]}...")
        else:
            print(f"❌ Ошибка математики: {response.status_code}")
            return False
        
        print("\n🎉 Все тесты пройдены успешно!")
        return True
        
    except ImportError:
        print("❌ Модуль requests не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    print("🤖 RUBIN AI - ПРОСТАЯ СИСТЕМА")
    print("=" * 50)
    
    # Запускаем систему
    if start_simple_rubin():
        print("\n✅ Система запущена успешно!")
    else:
        print("\n❌ Ошибка запуска системы")
        sys.exit(1)












