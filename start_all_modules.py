#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска всех модулей Rubin AI v2
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def start_server(script_path, description):
    """Запускает сервер в фоновом режиме"""
    try:
        print(f"🚀 Запускаю {description}...")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        print(f"✅ {description} запущен (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Ошибка запуска {description}: {e}")
        return None

def main():
    """Основная функция запуска всех серверов"""
    print("=" * 60)
    print("🤖 RUBIN AI v2 - ЗАПУСК ВСЕХ МОДУЛЕЙ")
    print("=" * 60)
    
    # Определяем пути к скриптам
    base_path = Path(__file__).parent
    
    servers = [
        {
            "script": "smart_dispatcher.py",
            "description": "Умный диспетчер (порт 8080)",
            "port": 8080
        },
        {
            "script": "api/electrical_api.py",
            "description": "Сервер электротехники (порт 8087)",
            "port": 8087
        },
        {
            "script": "api/radiomechanics_api.py",
            "description": "Сервер радиомеханики (порт 8089)",
            "port": 8089
        },
        {
            "script": "api/controllers_api.py",
            "description": "Сервер контроллеров (порт 9000)",
            "port": 9000
        },
        {
            "script": "math_server.py",
            "description": "Математический сервер (порт 8086)",
            "port": 8086
        },
        {
            "script": "api/programming_api.py",
            "description": "Сервер программирования (порт 8088)",
            "port": 8088
        }
    ]
    
    processes = []
    
    # Запускаем все серверы
    for server in servers:
        script_path = base_path / server["script"]
        
        if script_path.exists():
            process = start_server(script_path, server["description"])
            if process:
                processes.append({
                    "process": process,
                    "description": server["description"],
                    "port": server["port"]
                })
            time.sleep(2)  # Небольшая задержка между запусками
        else:
            print(f"❌ Файл не найден: {script_path}")
    
    print("\n" + "=" * 60)
    print("📊 СТАТУС ЗАПУЩЕННЫХ СЕРВЕРОВ:")
    print("=" * 60)
    
    for server_info in processes:
        print(f"✅ {server_info['description']} - PID: {server_info['process'].pid}")
    
    print("\n" + "=" * 60)
    print("🌐 ДОСТУПНЫЕ ЭНДПОИНТЫ:")
    print("=" * 60)
    print("🔗 Умный диспетчер: http://localhost:8080/api/chat")
    print("⚡ Электротехника: http://localhost:8087/api/electrical/explain")
    print("📡 Радиомеханика: http://localhost:8089/api/radiomechanics/explain")
    print("🎛️ Контроллеры: http://localhost:9000/api/controllers/topic/general")
    print("🧮 Математика: http://localhost:8086/api/chat")
    print("💻 Программирование: http://localhost:8088/api/programming/explain")
    
    print("\n" + "=" * 60)
    print("🎯 ИСПОЛЬЗОВАНИЕ:")
    print("=" * 60)
    print("• Отправляйте запросы на http://localhost:8080/api/chat")
    print("• Умный диспетчер автоматически направит запрос к нужному серверу")
    print("• Для остановки нажмите Ctrl+C")
    
    print("\n" + "=" * 60)
    print("⏳ ОЖИДАНИЕ ЗАПРОСОВ...")
    print("=" * 60)
    
    try:
        # Ожидаем завершения
        while True:
            time.sleep(1)
            
            # Проверяем, что все процессы еще работают
            active_processes = []
            for server_info in processes:
                if server_info["process"].poll() is None:
                    active_processes.append(server_info)
                else:
                    print(f"⚠️ {server_info['description']} завершился неожиданно")
            
            processes = active_processes
            
            if not processes:
                print("❌ Все серверы завершились")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 ОСТАНОВКА СЕРВЕРОВ...")
        
        for server_info in processes:
            try:
                server_info["process"].terminate()
                print(f"✅ {server_info['description']} остановлен")
            except Exception as e:
                print(f"❌ Ошибка остановки {server_info['description']}: {e}")
        
        print("\n👋 Все серверы остановлены. До свидания!")

if __name__ == "__main__":
    main()