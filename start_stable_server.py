#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Стабильный запуск Rubin AI v2.0 сервера
"""

import os
import sys
import subprocess
import time

def start_server():
    """Запуск основного сервера с обработкой ошибок"""
    print("Запуск стабильного Rubin AI v2.0 сервера...")
    
    # Устанавливаем переменную окружения для корректной кодировки
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # Запускаем основной сервер
        process = subprocess.Popen([
            sys.executable, 'api/rubin_ai_v2_server.py'
        ], encoding='utf-8', errors='ignore')
        
        print(f"Сервер запущен с PID: {process.pid}")
        print("Сервер доступен по адресу: http://localhost:8084")
        print("Веб-интерфейс: http://localhost:8084/RubinIDE.html")
        print("Developer интерфейс: http://localhost:8084/RubinDeveloper.html")
        print("\nНажмите Ctrl+C для остановки сервера")
        
        # Ожидаем завершения процесса
        process.wait()
        
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
        process.terminate()
        process.wait()
        print("Сервер остановлен")
    except Exception as e:
        print(f"Ошибка при запуске сервера: {e}")

if __name__ == "__main__":
    start_server()


















