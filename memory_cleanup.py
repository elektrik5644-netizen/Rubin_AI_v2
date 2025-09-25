#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 Memory Cleanup - Очистка памяти и остановка тяжелых процессов
"""

import psutil
import os
import gc
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_memory():
    """Очистка памяти Python"""
    logger.info("🧹 Очистка памяти Python...")
    
    # Принудительная сборка мусора
    collected = gc.collect()
    logger.info(f"✅ Собрано объектов: {collected}")
    
    # Очистка кэша модулей
    import sys
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('torch') or module_name.startswith('transformers'):
            del sys.modules[module_name]
    
    logger.info("✅ Очищен кэш модулей")

def kill_heavy_processes():
    """Остановка тяжелых процессов"""
    logger.info("🛑 Поиск тяжелых процессов...")
    
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            memory_mb = proc.info['memory_info'].rss // (1024**2)
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            # Ищем тяжелые Python процессы
            if (proc.info['name'] and 'python' in proc.info['name'].lower() and 
                memory_mb > 500 and  # Больше 500MB
                any(keyword in cmdline.lower() for keyword in ['torch', 'transformers', 'localai', 'docker'])):
                
                logger.warning(f"🔪 Останавливаем тяжелый процесс: {proc.info['name']} (PID: {proc.info['pid']}, {memory_mb}MB)")
                proc.terminate()
                killed_count += 1
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    logger.info(f"✅ Остановлено процессов: {killed_count}")

def optimize_system():
    """Оптимизация системы"""
    logger.info("⚙️  Оптимизация системы...")
    
    # Очистка временных файлов
    temp_dirs = ['/tmp', 'C:\\temp', 'C:\\Windows\\Temp']
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(('.log', '.tmp', '.cache')):
                            file_path = os.path.join(root, file)
                            try:
                                os.remove(file_path)
                            except:
                                pass
            except:
                pass
    
    logger.info("✅ Очищены временные файлы")

def show_memory_status():
    """Показать статус памяти"""
    memory = psutil.virtual_memory()
    
    logger.info("📊 Статус памяти:")
    logger.info(f"  💾 Всего: {memory.total // (1024**3):.1f}GB")
    logger.info(f"  🔴 Использовано: {memory.used // (1024**3):.1f}GB ({memory.percent:.1f}%)")
    logger.info(f"  🟢 Доступно: {memory.available // (1024**3):.1f}GB")
    
    if memory.percent > 90:
        logger.warning("⚠️  КРИТИЧНО: Использование памяти превышает 90%!")
    elif memory.percent > 80:
        logger.warning("⚠️  ВНИМАНИЕ: Использование памяти превышает 80%")
    else:
        logger.info("✅ Память в норме")

def main():
    logger.info("🧹 Memory Cleanup запущен")
    logger.info("=" * 40)
    
    # Показываем текущий статус
    show_memory_status()
    
    # Очистка памяти
    cleanup_memory()
    
    # Остановка тяжелых процессов
    kill_heavy_processes()
    
    # Оптимизация системы
    optimize_system()
    
    # Показываем финальный статус
    logger.info("\n📊 Финальный статус:")
    show_memory_status()
    
    logger.info("✅ Очистка завершена")

if __name__ == '__main__':
    main()



