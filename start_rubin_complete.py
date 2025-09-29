#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный запуск Rubin AI v2.0 с системой документов
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def start_server(script_path, port, name):
    """Запуск сервера в отдельном потоке"""
    try:
        print(f"🚀 Запуск {name} на порту {port}...")
        process = subprocess.Popen([
            sys.executable, script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)
        
        print(f"✅ {name} запущен (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Ошибка запуска {name}: {e}")
        return None

def check_port(port):
    """Проверка доступности порта"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            return result == 0
    except:
        return False

def main():
    """Главная функция"""
    print("🎯 ПОЛНЫЙ ЗАПУСК RUBIN AI v2.0")
    print("=" * 50)
    
    # Список серверов для запуска
    servers = [
        ("start_stable_server.py", 8084, "AI Чат (Основной)"),
        ("api/electrical_api.py", 8087, "Электротехника"),
        ("api/radiomechanics_api.py", 8089, "Радиомеханика"),
        ("api/controllers_api.py", 8090, "Контроллеры"),
        ("api/documents_api.py", 8088, "Документы"),
        ("static_web_server.py", 8085, "Статический Веб-сервер")
    ]
    
    processes = []
    
    # Запускаем все серверы
    for script, port, name in servers:
        if os.path.exists(script):
            if not check_port(port):
                process = start_server(script, port, name)
                if process:
                    processes.append((process, name, port))
                time.sleep(2)  # Небольшая задержка между запусками
            else:
                print(f"⚠️ Порт {port} уже используется, пропускаем {name}")
        else:
            print(f"⚠️ Файл {script} не найден, пропускаем {name}")
    
    print("\n" + "=" * 50)
    print("📊 СТАТУС ЗАПУЩЕННЫХ СЕРВЕРОВ:")
    print("=" * 50)
    
    # Проверяем статус серверов
    for process, name, port in processes:
        if process and process.poll() is None:
            print(f"✅ {name}: ОНЛАЙН (порт {port}, PID: {process.pid})")
        else:
            print(f"❌ {name}: ОФФЛАЙН (порт {port})")
    
    print("\n🌐 ДОСТУПНЫЕ ИНТЕРФЕЙСЫ:")
    print("   🤖 AI Чат: http://localhost:8084/RubinIDE.html")
    print("   ⚙️ Developer: http://localhost:8084/RubinDeveloper.html")
    print("   📊 Статус: http://localhost:8084/status_check.html")
    print("   📚 Документы: http://localhost:8088/DocumentsManager.html")
    print("   🌐 Статический: http://localhost:8085/RubinIDE.html")
    
    print("\n🔍 СПЕЦИАЛИЗИРОВАННЫЕ API:")
    print("   ⚡ Электротехника: http://localhost:8087/api/electrical/status")
    print("   📡 Радиомеханика: http://localhost:8089/api/radiomechanics/status")
    print("   🎛️ Контроллеры: http://localhost:8090/api/controllers/status")
    print("   📚 Документы API: http://localhost:8088/health")
    
    print("\n⏳ Нажмите Ctrl+C для остановки всех серверов")
    
    try:
        # Ждем завершения
        while True:
            time.sleep(1)
            # Проверяем, что процессы еще работают
            active_processes = []
            for process, name, port in processes:
                if process and process.poll() is None:
                    active_processes.append((process, name, port))
            processes = active_processes
            
            if not processes:
                print("\n⚠️ Все серверы остановлены")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Остановка всех серверов...")
        
        # Останавливаем все процессы
        for process, name, port in processes:
            if process and process.poll() is None:
                try:
                    print(f"🛑 Остановка {name}...")
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"✅ {name} остановлен")
                except:
                    print(f"❌ Ошибка остановки {name}")
        
        print("✅ Все серверы остановлены")

if __name__ == "__main__":
    main()























