#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый запуск основного сервера Rubin AI v2.0
"""

import subprocess
import sys
import os
import time

def main():
    """Главная функция"""
    print("🚀 БЫСТРЫЙ ЗАПУСК ОСНОВНОГО СЕРВЕРА RUBIN AI v2.0")
    print("=" * 50)
    
    # Проверяем, что файл существует
    server_script = "start_stable_server.py"
    if not os.path.exists(server_script):
        print(f"❌ Файл {server_script} не найден!")
        return
    
    print(f"🎯 Запускаем основной сервер...")
    
    try:
        # Запускаем сервер
        process = subprocess.Popen([
            sys.executable, server_script
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)
        
        print(f"✅ Основной сервер запущен (PID: {process.pid})")
        print(f"🌐 Веб-интерфейс: http://localhost:8084/RubinIDE.html")
        print(f"📊 Статус: http://localhost:8084/health")
        print(f"⏳ Нажмите Ctrl+C для остановки")
        
        # Ждем завершения процесса
        try:
            process.wait()
        except KeyboardInterrupt:
            print(f"\n🛑 Остановка сервера...")
            process.terminate()
            process.wait()
            print(f"✅ Сервер остановлен")
            
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()


















