#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start All Servers Script
Скрипт для запуска всех серверов Rubin AI v2
"""

import subprocess
import time
import threading
import os
import sys
from datetime import datetime

# Конфигурация серверов для запуска
SERVERS_TO_START = [
    {
        'name': 'Smart Dispatcher',
        'command': 'python smart_dispatcher.py',
        'port': 8080,
        'description': 'Главный диспетчер запросов'
    },
    {
        'name': 'General API',
        'command': 'python api/general_api.py',
        'port': 8085,
        'description': 'Общие вопросы и справка'
    },
    {
        'name': 'Mathematics Server',
        'command': 'python math_server.py',
        'port': 8086,
        'description': 'Математические вычисления'
    },
    {
        'name': 'Electrical API',
        'command': 'python api/electrical_api.py',
        'port': 8087,
        'description': 'Электротехнические расчеты'
    },
    {
        'name': 'Programming API',
        'command': 'python api/programming_api.py',
        'port': 8088,
        'description': 'Программирование и алгоритмы'
    },
    {
        'name': 'Radiomechanics API',
        'command': 'python api/radiomechanics_api.py',
        'port': 8089,
        'description': 'Радиомеханические расчеты'
    },
    {
        'name': 'Neural Network API',
        'command': 'python neuro_server.py',
        'port': 8090,
        'description': 'Нейронные сети и машинное обучение'
    },
    {
        'name': 'Controllers API',
        'command': 'python api/controllers_api.py',
        'port': 9000,
        'description': 'ПЛК, ЧПУ, автоматизация'
    },
    {
        'name': 'PLC Analysis API',
        'command': 'python plc_analysis_api_server.py',
        'port': 8099,
        'description': 'Анализ и диагностика PLC программ'
    },
    {
        'name': 'Advanced Mathematics API',
        'command': 'python advanced_math_api_server.py',
        'port': 8100,
        'description': 'Продвинутые математические вычисления'
    },
    {
        'name': 'Data Processing API',
        'command': 'python data_processing_api_server.py',
        'port': 8101,
        'description': 'Обработка и анализ данных'
    },
    {
        'name': 'Search Engine API',
        'command': 'python search_engine_api_server.py',
        'port': 8102,
        'description': 'Гибридный поиск и индексация'
    },
    {
        'name': 'System Utils API',
        'command': 'python system_utils_api_server.py',
        'port': 8103,
        'description': 'Системные утилиты и диагностика'
    },
    {
        'name': 'Unified System Manager',
        'command': 'python unified_system_manager.py',
        'port': 8084,
        'description': 'Единая система управления'
    }
]

def start_server(server_config):
    """Запуск отдельного сервера"""
    try:
        print(f"🚀 Запуск {server_config['name']}...")
        
        # Проверяем существование файла
        if not os.path.exists(server_config['command'].split()[1]):
            print(f"❌ Файл не найден: {server_config['command'].split()[1]}")
            return None
        
        # Запускаем сервер
        process = subprocess.Popen(
            server_config['command'],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Даем серверу время запуститься
        time.sleep(2)
        
        # Проверяем, что процесс запустился
        if process.poll() is None:
            print(f"✅ {server_config['name']} запущен (PID: {process.pid})")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Ошибка запуска {server_config['name']}: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка запуска {server_config['name']}: {e}")
        return None

def check_port_availability(port):
    """Проверка доступности порта"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0  # True если порт свободен
    except:
        return False

def wait_for_server(port, timeout=30):
    """Ожидание запуска сервера"""
    import socket
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                return True
        except:
            pass
        
        time.sleep(1)
    
    return False

def main():
    """Основная функция"""
    print("🎛️ Rubin AI v2 - Запуск всех серверов")
    print("=" * 50)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Всего серверов: {len(SERVERS_TO_START)}")
    print("=" * 50)
    
    running_processes = []
    failed_servers = []
    
    # Запускаем серверы по очереди
    for i, server_config in enumerate(SERVERS_TO_START, 1):
        print(f"\n[{i}/{len(SERVERS_TO_START)}] {server_config['name']}")
        print(f"Порт: {server_config['port']}")
        print(f"Описание: {server_config['description']}")
        
        # Проверяем доступность порта
        if not check_port_availability(server_config['port']):
            print(f"⚠️ Порт {server_config['port']} уже занят, пропускаем...")
            continue
        
        # Запускаем сервер
        process = start_server(server_config)
        
        if process:
            running_processes.append({
                'process': process,
                'config': server_config
            })
            
            # Ждем запуска сервера
            if wait_for_server(server_config['port'], timeout=10):
                print(f"✅ {server_config['name']} готов к работе")
            else:
                print(f"⚠️ {server_config['name']} запущен, но не отвечает на порту {server_config['port']}")
        else:
            failed_servers.append(server_config['name'])
        
        # Небольшая пауза между запусками
        time.sleep(1)
    
    # Итоговая статистика
    print("\n" + "=" * 50)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 50)
    print(f"✅ Успешно запущено: {len(running_processes)}")
    print(f"❌ Не удалось запустить: {len(failed_servers)}")
    
    if running_processes:
        print("\n🟢 Запущенные серверы:")
        for proc_info in running_processes:
            config = proc_info['config']
            print(f"  - {config['name']} (порт {config['port']})")
    
    if failed_servers:
        print("\n🔴 Неудачные запуски:")
        for server_name in failed_servers:
            print(f"  - {server_name}")
    
    print("\n🌐 Доступные интерфейсы:")
    print("  - Smart Dispatcher: http://localhost:8080")
    print("  - Unified System Manager: http://localhost:8084")
    print("  - RubinIDE: http://localhost:8080/RubinIDE.html")
    print("  - RubinDeveloper: http://localhost:8080/matrix/RubinDeveloper.html")
    
    print("\n📡 API Endpoints:")
    for proc_info in running_processes:
        config = proc_info['config']
        if config['port'] == 8080:
            print(f"  - Главный API: http://localhost:{config['port']}/api/chat")
        elif config['port'] == 8084:
            print(f"  - Системное управление: http://localhost:{config['port']}/api/system/status")
        else:
            print(f"  - {config['name']}: http://localhost:{config['port']}")
    
    print("\n" + "=" * 50)
    print("🎉 Запуск завершен!")
    print("Для остановки всех серверов нажмите Ctrl+C")
    print("=" * 50)
    
    # Ожидаем завершения
    try:
        while True:
            time.sleep(1)
            
            # Проверяем, что все процессы еще работают
            active_processes = []
            for proc_info in running_processes:
                if proc_info['process'].poll() is None:
                    active_processes.append(proc_info)
                else:
                    print(f"⚠️ Сервер {proc_info['config']['name']} завершился")
            
            running_processes = active_processes
            
            if not running_processes:
                print("❌ Все серверы завершились")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки...")
        
        # Останавливаем все процессы
        for proc_info in running_processes:
            try:
                proc_info['process'].terminate()
                print(f"🛑 Остановка {proc_info['config']['name']}...")
            except:
                pass
        
        # Ждем завершения процессов
        time.sleep(3)
        
        # Принудительно завершаем процессы
        for proc_info in running_processes:
            try:
                if proc_info['process'].poll() is None:
                    proc_info['process'].kill()
            except:
                pass
        
        print("✅ Все серверы остановлены")

if __name__ == '__main__':
    main()








