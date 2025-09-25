#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Стабильный лаунчер для Rubin AI v2
Поэтапный запуск с проверкой памяти
"""

import subprocess
import time
import psutil
import gc
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Критически важные серверы (минимум для работы)
CRITICAL_SERVERS = [
    "optimized_dispatcher.py",  # Новый оптимизированный диспетчер
    "api/general_api.py",
    "api/mathematics_api.py", 
    "api/electrical_api.py",
    "api/programming_api.py"
]

# Дополнительные серверы (запускаются после основных)
ADDITIONAL_SERVERS = [
    "simple_neuro_api_server.py",
    "simple_controllers_api_server.py",
    "lightweight_gai_server.py",
    "unified_system_manager.py",
    "ethical_core_api_server.py"
]

def get_memory_usage():
    """Получить текущее использование памяти"""
    mem = psutil.virtual_memory()
    return {
        'total': round(mem.total / (1024**3), 1),
        'used': round(mem.used / (1024**3), 1),
        'available': round(mem.available / (1024**3), 1),
        'percent': mem.percent
    }

def cleanup_memory():
    """Очистка памяти"""
    logger.info("🧹 Очистка памяти...")
    gc.collect()
    
    # Останавливаем тяжелые процессы Python
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if proc.info['name'] == 'python.exe':
                mem_usage_mb = proc.info['memory_info'].rss / (1024 * 1024)
                if mem_usage_mb > 200:  # Процессы больше 200MB
                    logger.warning(f"Останавливаю тяжелый процесс Python (PID: {proc.info['pid']}, Память: {mem_usage_mb:.1f}MB)")
                    proc.terminate()
                    time.sleep(0.5)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def launch_server(script_name):
    """Запуск сервера"""
    logger.info(f"🚀 Запускаю {script_name}...")
    try:
        if script_name == "optimized_dispatcher.py":
            # Диспетчер запускаем в отдельном окне
            subprocess.Popen(['python', script_name], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
        else:
            # Остальные серверы в фоне
            subprocess.Popen(['python', script_name], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        logger.info(f"✅ {script_name} запущен")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка запуска {script_name}: {e}")
        return False

def check_memory_before_launch():
    """Проверка памяти перед запуском"""
    mem = get_memory_usage()
    logger.info(f"📊 Память: {mem['used']:.1f}GB / {mem['total']:.1f}GB ({mem['percent']:.1f}%)")
    
    if mem['percent'] > 80:
        logger.warning("⚠️ Высокое использование памяти! Выполняю очистку...")
        cleanup_memory()
        time.sleep(2)
        
        # Проверяем еще раз
        mem = get_memory_usage()
        logger.info(f"📊 После очистки: {mem['used']:.1f}GB / {mem['total']:.1f}GB ({mem['percent']:.1f}%)")
        
        if mem['percent'] > 85:
            logger.error("❌ Критически мало памяти! Запуск отменен.")
            return False
    
    return True

def main():
    """Основная функция"""
    import os
    logger.info("🚀 Стабильный лаунчер Rubin AI v2")
    logger.info("=" * 50)
    
    # 1. Проверка памяти
    if not check_memory_before_launch():
        return
    
    # 2. Запуск критических серверов
    logger.info("📋 Этап 1: Запуск критических серверов...")
    critical_success = 0
    for server in CRITICAL_SERVERS:
        if launch_server(server):
            critical_success += 1
        time.sleep(1)  # Пауза между запусками
    
    logger.info(f"✅ Критические серверы: {critical_success}/{len(CRITICAL_SERVERS)}")
    
    # 3. Проверка памяти перед дополнительными серверами
    time.sleep(3)  # Даем время на запуск
    if not check_memory_before_launch():
        logger.warning("⚠️ Память критична, пропускаю дополнительные серверы")
        return
    
    # 4. Запуск дополнительных серверов
    logger.info("📋 Этап 2: Запуск дополнительных серверов...")
    additional_success = 0
    for server in ADDITIONAL_SERVERS:
        if launch_server(server):
            additional_success += 1
        time.sleep(1)
    
    logger.info(f"✅ Дополнительные серверы: {additional_success}/{len(ADDITIONAL_SERVERS)}")
    
    # 5. Финальная проверка
    time.sleep(2)
    mem = get_memory_usage()
    total_success = critical_success + additional_success
    
    logger.info("=" * 50)
    logger.info(f"🎉 Запуск завершен!")
    logger.info(f"📊 Серверов запущено: {total_success}/{len(CRITICAL_SERVERS) + len(ADDITIONAL_SERVERS)}")
    logger.info(f"💾 Память: {mem['used']:.1f}GB / {mem['total']:.1f}GB ({mem['percent']:.1f}%)")
    logger.info(f"🌐 Smart Dispatcher: http://localhost:8080")
    logger.info(f"📱 RubinDeveloper: http://localhost:8080/matrix/RubinDeveloper.html")

if __name__ == '__main__':
    main()
