#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start All Servers for RubinDeveloper Testing
Быстрый запуск всех серверов для тестирования в RubinDeveloper
"""

import subprocess
import time
import os
import sys
from datetime import datetime

def start_server(server_name, command, port):
    """Запуск сервера в фоновом режиме"""
    try:
        print(f"🚀 Запуск {server_name} на порту {port}...")
        
        # Проверяем существование файла
        if not os.path.exists(command.split()[1]):
            print(f"❌ Файл не найден: {command.split()[1]}")
            return None
        
        # Запускаем сервер
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Даем серверу время запуститься
        time.sleep(1)
        
        print(f"✅ {server_name} запущен (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"❌ Ошибка запуска {server_name}: {e}")
        return None

def main():
    """Основная функция"""
    print("🎛️ RubinDeveloper - Быстрый запуск всех серверов")
    print("=" * 60)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Список серверов для RubinDeveloper
    servers = [
        # Основные серверы
        ("Smart Dispatcher", "python smart_dispatcher.py", 8080),
        ("General API", "python api/general_api.py", 8085),
        ("Mathematics Server", "python math_server.py", 8086),
        ("Electrical API", "python api/electrical_api.py", 8087),
        ("Programming API", "python api/programming_api.py", 8088),
        ("Radiomechanics API", "python api/radiomechanics_api.py", 8089),
        ("Neural Network API", "python neuro_server.py", 8090),
        ("Controllers API", "python api/controllers_api.py", 9000),
        
        # Новые приоритетные серверы
        ("PLC Analysis API", "python plc_analysis_api_server.py", 8099),
        ("Advanced Math API", "python advanced_math_api_server.py", 8100),
        ("Data Processing API", "python data_processing_api_server.py", 8101),
        ("Search Engine API", "python search_engine_api_server.py", 8102),
        ("System Utils API", "python system_utils_api_server.py", 8103),
        ("GAI Server", "python enhanced_gai_server.py", 8104),
        ("Unified Manager", "python unified_system_manager.py", 8084)
    ]
    
    running_processes = []
    
    print(f"Всего серверов для запуска: {len(servers)}")
    print()
    
    # Запускаем серверы
    for i, (name, command, port) in enumerate(servers, 1):
        print(f"[{i}/{len(servers)}] {name}")
        process = start_server(name, command, port)
        if process:
            running_processes.append((name, process, port))
        print()
    
    # Итоговая статистика
    print("=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"✅ Успешно запущено: {len(running_processes)}")
    print(f"❌ Не удалось запустить: {len(servers) - len(running_processes)}")
    
    if running_processes:
        print("\n🟢 Запущенные серверы:")
        for name, process, port in running_processes:
            print(f"  - {name} (порт {port})")
    
    print("\n🌐 Доступные интерфейсы:")
    print("  - RubinDeveloper: http://localhost:8080/matrix/RubinDeveloper.html")
    print("  - Unified System Manager: http://localhost:8084")
    print("  - Smart Dispatcher: http://localhost:8080")
    
    print("\n📡 Тестирование в RubinDeveloper:")
    print("  1. Откройте http://localhost:8080/matrix/RubinDeveloper.html")
    print("  2. Нажмите '🔌 Проверить все API модули'")
    print("  3. Все новые серверы должны отображаться как ОНЛАЙН")
    
    print("\n" + "=" * 60)
    print("🎉 Запуск завершен!")
    print("Теперь можете тестировать все функции в RubinDeveloper!")
    print("Для остановки всех серверов нажмите Ctrl+C")
    print("=" * 60)
    
    # Ожидаем завершения
    try:
        while True:
            time.sleep(1)
            
            # Проверяем, что все процессы еще работают
            active_processes = []
            for name, process, port in running_processes:
                if process.poll() is None:
                    active_processes.append((name, process, port))
                else:
                    print(f"⚠️ Сервер {name} завершился")
            
            running_processes = active_processes
            
            if not running_processes:
                print("❌ Все серверы завершились")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки...")
        
        # Останавливаем все процессы
        for name, process, port in running_processes:
            try:
                process.terminate()
                print(f"🛑 Остановка {name}...")
            except:
                pass
        
        # Ждем завершения процессов
        time.sleep(3)
        
        # Принудительно завершаем процессы
        for name, process, port in running_processes:
            try:
                if process.poll() is None:
                    process.kill()
            except:
                pass
        
        print("✅ Все серверы остановлены")

if __name__ == '__main__':
    main()



