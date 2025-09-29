#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый запуск всех модулей Rubin AI v2.0
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def start_module(name, command, port):
    """Запуск модуля в отдельном потоке"""
    try:
        print(f"🚀 Запуск {name} на порту {port}...")
        
        # Устанавливаем переменную окружения для корректной кодировки
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore',
            env=env
        )
        
        print(f"✅ {name} запущен (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"❌ Ошибка запуска {name}: {e}")
        return None

def main():
    """Главная функция"""
    print("🎯 БЫСТРЫЙ ЗАПУСК RUBIN AI v2.0")
    print("=" * 50)
    
    # Определяем модули для запуска
    modules = [
        {
            'name': 'AI Чат (Основной сервер)',
            'command': [sys.executable, 'api/rubin_ai_v2_server.py'],
            'port': 8084
        },
        {
            'name': 'Электротехника',
            'command': [sys.executable, 'api/electrical_api.py'],
            'port': 8087
        },
        {
            'name': 'Радиомеханика',
            'command': [sys.executable, 'api/radiomechanics_api.py'],
            'port': 8089
        },
        {
            'name': 'Контроллеры',
            'command': [sys.executable, 'api/controllers_api.py'],
            'port': 8090
        }
    ]
    
    processes = []
    
    # Запускаем все модули
    for module in modules:
        process = start_module(module['name'], module['command'], module['port'])
        if process:
            processes.append(process)
        time.sleep(2)  # Небольшая задержка между запусками
    
    print("\n" + "=" * 50)
    print(f"📊 Результат: {len(processes)}/{len(modules)} модулей запущено")
    
    if processes:
        print("\n🌐 Доступные интерфейсы:")
        print("  - AI Чат: http://localhost:8084/RubinIDE.html")
        print("  - Developer: http://localhost:8084/RubinDeveloper.html")
        print("  - Электротехника: http://localhost:8087/api/electrical/status")
        print("  - Радиомеханика: http://localhost:8089/api/radiomechanics/status")
        print("  - Контроллеры: http://localhost:8090/api/controllers/status")
        print("\n⏳ Нажмите Ctrl+C для остановки всех модулей")
        
        try:
            # Ждем завершения всех процессов
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Получен сигнал остановки...")
            for process in processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
            print("✅ Все модули остановлены")
    else:
        print("❌ Не удалось запустить ни одного модуля")

if __name__ == "__main__":
    main()























