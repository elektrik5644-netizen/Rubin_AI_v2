#!/usr/bin/env python3
"""
Скрипт для перезапуска сервера Rubin AI
"""

import os
import sys
import time
import subprocess
import signal

def kill_python_processes():
    """Остановка всех процессов Python"""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         capture_output=True, text=True)
            print("🛑 Остановлены процессы Python")
        else:  # Linux/Mac
            subprocess.run(['pkill', '-f', 'python'], 
                         capture_output=True, text=True)
            print("🛑 Остановлены процессы Python")
    except Exception as e:
        print(f"⚠️ Ошибка при остановке процессов: {e}")

def start_enhanced_server():
    """Запуск улучшенного сервера"""
    try:
        print("🚀 Запуск Enhanced Rubin AI Server...")
        
        # Запуск в фоновом режиме
        if os.name == 'nt':  # Windows
            subprocess.Popen([sys.executable, 'enhanced_rubin_server.py'], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Mac
            subprocess.Popen([sys.executable, 'enhanced_rubin_server.py'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        print("✅ Enhanced Rubin AI Server запущен")
        print("🌐 Доступен по адресу: http://localhost:8083")
        print("📊 База данных: rubin_ai.db")
        
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")

def main():
    """Главная функция"""
    print("🔄 ПЕРЕЗАПУСК RUBIN AI SERVER")
    print("="*40)
    
    # Остановка старых процессов
    kill_python_processes()
    
    # Небольшая пауза
    time.sleep(2)
    
    # Запуск нового сервера
    start_enhanced_server()
    
    # Пауза для запуска
    time.sleep(3)
    
    print("\n🧪 Запуск тестирования...")
    
    # Запуск быстрого теста
    try:
        import quick_test
        quick_test.test_enhanced_server()
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

if __name__ == "__main__":
    main()
