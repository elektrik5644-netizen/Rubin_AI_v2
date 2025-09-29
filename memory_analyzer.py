#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор потребления памяти серверов Rubin AI v2
"""

import psutil
import subprocess
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Список серверов для анализа
SERVERS = {
    'Smart Dispatcher': {
        'script': 'simple_dispatcher.py',
        'port': 8080,
        'description': 'Основной диспетчер запросов'
    },
    'General API': {
        'script': 'api/general_api.py',
        'port': 8085,
        'description': 'Общие запросы и маршрутизация'
    },
    'Mathematics': {
        'script': 'api/mathematics_api.py',
        'port': 8086,
        'description': 'Математические вычисления'
    },
    'Electrical': {
        'script': 'api/electrical_api.py',
        'port': 8087,
        'description': 'Электротехника и схемы'
    },
    'Programming': {
        'script': 'api/programming_api.py',
        'port': 8088,
        'description': 'Программирование и разработка'
    },
    'Neuro': {
        'script': 'simple_neuro_api_server.py',
        'port': 8090,
        'description': 'Нейросетевые алгоритмы'
    },
    'Controllers': {
        'script': 'simple_controllers_api_server.py',
        'port': 9000,
        'description': 'Промышленные контроллеры'
    },
    'PLC Analysis': {
        'script': 'simple_plc_analysis_api_server.py',
        'port': 8099,
        'description': 'Анализ PLC программ'
    },
    'Advanced Math': {
        'script': 'simple_advanced_math_api_server.py',
        'port': 8100,
        'description': 'Продвинутая математика'
    },
    'Data Processing': {
        'script': 'simple_data_processing_api_server.py',
        'port': 8101,
        'description': 'Обработка данных'
    },
    'Search Engine': {
        'script': 'search_engine_api_server.py',
        'port': 8102,
        'description': 'Поисковая система'
    },
    'System Utils': {
        'script': 'system_utils_api_server.py',
        'port': 8103,
        'description': 'Системные утилиты'
    },
    'GAI Server': {
        'script': 'lightweight_gai_server.py',
        'port': 8104,
        'description': 'Генеративный ИИ (облегченный)'
    },
    'Unified Manager': {
        'script': 'unified_system_manager.py',
        'port': 8084,
        'description': 'Управление системой'
    },
    'Ethical Core': {
        'script': 'ethical_core_api_server.py',
        'port': 8105,
        'description': 'Этическое ядро'
    }
}

def get_system_memory():
    """Получить общую информацию о памяти системы"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': round(mem.total / (1024**3), 1),
        'available_gb': round(mem.available / (1024**3), 1),
        'used_gb': round(mem.used / (1024**3), 1),
        'percent': mem.percent
    }

def get_python_processes():
    """Получить все процессы Python с информацией о памяти"""
    python_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'Unknown'
                memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                
                python_processes.append({
                    'pid': proc.info['pid'],
                    'cmdline': cmdline,
                    'memory_mb': round(memory_mb, 1),
                    'create_time': proc.info['create_time']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return python_processes

def identify_server_processes():
    """Определить какие процессы относятся к каким серверам"""
    python_procs = get_python_processes()
    server_processes = {}
    
    for server_name, server_info in SERVERS.items():
        server_processes[server_name] = []
        
        for proc in python_procs:
            cmdline = proc['cmdline'].lower()
            script_name = server_info['script'].lower()
            
            # Проверяем, содержит ли командная строка имя скрипта сервера
            if script_name in cmdline:
                server_processes[server_name].append(proc)
    
    return server_processes

def analyze_memory_usage():
    """Анализ потребления памяти"""
    print("🔍 Анализ потребления памяти серверов Rubin AI v2")
    print("=" * 80)
    
    # Общая информация о системе
    system_mem = get_system_memory()
    print(f"💾 Системная память: {system_mem['used_gb']:.1f}GB / {system_mem['total_gb']:.1f}GB ({system_mem['percent']:.1f}%)")
    print(f"🟢 Доступно: {system_mem['available_gb']:.1f}GB")
    print()
    
    # Анализ процессов серверов
    server_processes = identify_server_processes()
    
    total_server_memory = 0
    active_servers = 0
    
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ СЕРВЕРОВ:")
    print("-" * 80)
    print(f"{'Сервер':<20} {'Статус':<10} {'Память (MB)':<12} {'Процессы':<8} {'Описание'}")
    print("-" * 80)
    
    for server_name, server_info in SERVERS.items():
        processes = server_processes[server_name]
        
        if processes:
            total_memory = sum(proc['memory_mb'] for proc in processes)
            process_count = len(processes)
            status = "🟢 ОНЛАЙН"
            active_servers += 1
        else:
            total_memory = 0
            process_count = 0
            status = "🔴 ОФФЛАЙН"
        
        total_server_memory += total_memory
        
        print(f"{server_name:<20} {status:<10} {total_memory:<12.1f} {process_count:<8} {server_info['description']}")
    
    print("-" * 80)
    print(f"{'ИТОГО':<20} {'':<10} {total_server_memory:<12.1f} {'':<8} {active_servers} активных серверов")
    print()
    
    # Анализ других Python процессов
    all_python_procs = get_python_processes()
    other_python_memory = 0
    other_processes = []
    
    for proc in all_python_procs:
        is_server_process = False
        for server_name, processes in server_processes.items():
            if any(p['pid'] == proc['pid'] for p in processes):
                is_server_process = True
                break
        
        if not is_server_process:
            other_python_memory += proc['memory_mb']
            other_processes.append(proc)
    
    if other_processes:
        print("🐍 ДРУГИЕ PYTHON ПРОЦЕССЫ:")
        print("-" * 80)
        for proc in other_processes:
            print(f"PID {proc['pid']:<8} {proc['memory_mb']:<8.1f}MB - {proc['cmdline'][:60]}...")
        print(f"Общая память других Python процессов: {other_python_memory:.1f}MB")
        print()
    
    # Рекомендации
    print("💡 РЕКОМЕНДАЦИИ:")
    print("-" * 80)
    
    if system_mem['percent'] > 85:
        print("🚨 КРИТИЧНО: Использование памяти превышает 85%!")
        print("   - Перезагрузите систему")
        print("   - Закройте ненужные программы")
        print("   - Рассмотрите увеличение RAM")
    elif system_mem['percent'] > 70:
        print("⚠️  ВНИМАНИЕ: Высокое использование памяти")
        print("   - Остановите неиспользуемые серверы")
        print("   - Используйте lightweight_gai_server.py вместо полного GAI")
    else:
        print("✅ Использование памяти в норме")
    
    if total_server_memory > 1000:  # Больше 1GB
        print(f"📊 Серверы Rubin AI используют {total_server_memory:.1f}MB памяти")
        print("   - Рассмотрите запуск только критических серверов")
        print("   - Используйте optimized_dispatcher.py для экономии памяти")
    
    print()
    print("🎯 ОПТИМИЗАЦИЯ:")
    print("-" * 80)
    print("• Используйте lightweight_gai_server.py (экономия ~500MB)")
    print("• Запускайте только необходимые серверы")
    print("• Используйте optimized_dispatcher.py вместо simple_dispatcher.py")
    print("• Регулярно перезапускайте систему")

if __name__ == '__main__':
    analyze_memory_usage()








