#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска всех API серверов Rubin AI
"""

import subprocess
import time
import sys
import os

def start_server(script_path, port, name):
    """Запуск сервера в отдельном процессе"""
    try:
        print(f"🚀 Запускаю {name} на порту {port}...")
        process = subprocess.Popen([
            sys.executable, script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Даем серверу время запуститься
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ {name} успешно запущен на порту {port}")
            return process
        else:
            print(f"❌ Ошибка запуска {name}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка запуска {name}: {e}")
        return None

def main():
    """Основная функция запуска серверов"""
    print("🎯 Запуск всех API серверов Rubin AI")
    print("=" * 50)
    
    # Список серверов для запуска
    servers = [
        ("api/general_api.py", 8085, "General API"),
        ("api/mathematics_api.py", 8086, "Mathematics API"),
        ("api/electrical_api.py", 8087, "Electrical API"),
        ("api/programming_api.py", 8088, "Programming API"),
        ("api/neuro_repository_api.py", 8090, "Neuro Repository API"),
        ("api/controllers_api.py", 9000, "Controllers API"),
        ("api/plc_analysis_api.py", 8099, "PLC Analysis API"),
        ("api/advanced_math_api.py", 8100, "Advanced Math API"),
        ("api/data_processing_api.py", 8101, "Data Processing API"),
        ("api/search_engine_api.py", 8102, "Search Engine API"),
        ("api/system_utils_api.py", 8103, "System Utils API"),
        ("api/gai_server_api.py", 8104, "GAI Server API"),
        ("api/ethical_core_api.py", 8105, "Ethical Core API"),
    ]
    
    processes = []
    
    # Запускаем все серверы
    for script, port, name in servers:
        if os.path.exists(script):
            process = start_server(script, port, name)
            if process:
                processes.append((process, name, port))
        else:
            print(f"⚠️ Файл {script} не найден, пропускаю")
    
    print("\n" + "=" * 50)
    print(f"✅ Запущено {len(processes)} серверов")
    print("Нажмите Ctrl+C для остановки всех серверов")
    
    try:
        # Ждем завершения
        while True:
            time.sleep(1)
            
            # Проверяем, что все процессы еще работают
            active_processes = []
            for process, name, port in processes:
                if process.poll() is None:
                    active_processes.append((process, name, port))
                else:
                    print(f"⚠️ {name} (порт {port}) завершился")
            
            processes = active_processes
            
            if not processes:
                print("❌ Все серверы завершились")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Остановка всех серверов...")
        
        # Завершаем все процессы
        for process, name, port in processes:
            try:
                process.terminate()
                print(f"🛑 Остановлен {name} (порт {port})")
            except:
                pass
        
        print("✅ Все серверы остановлены")

if __name__ == "__main__":
    main()