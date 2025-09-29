#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизатор памяти для Rubin AI v2.0
Мониторинг и автоматическая оптимизация потребления памяти
"""

import psutil
import gc
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import threading

class MemoryOptimizer:
    """Оптимизатор памяти для Rubin AI"""
    
    def __init__(self, threshold_mb: int = 500, cleanup_interval: int = 300):
        self.threshold_mb = threshold_mb
        self.cleanup_interval = cleanup_interval
        self.logger = logging.getLogger("rubin_ai.memory_optimizer")
        self.is_monitoring = False
        self.monitor_thread = None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Получить текущее использование памяти"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Физическая память
            'vms_mb': memory_info.vms / 1024 / 1024,  # Виртуальная память
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def cleanup_memory(self) -> Dict[str, float]:
        """Очистка памяти"""
        before = self.get_memory_usage()
        
        # Принудительная сборка мусора
        collected = gc.collect()
        
        # Очистка кэшей (если есть)
        try:
            import sys
            # Очистка модулей, которые больше не используются
            modules_to_remove = []
            for module_name, module in sys.modules.items():
                if hasattr(module, '__file__') and module.__file__:
                    if 'cache' in module_name.lower() or 'temp' in module_name.lower():
                        modules_to_remove.append(module_name)
            
            for module_name in modules_to_remove:
                if module_name in sys.modules:
                    del sys.modules[module_name]
        except Exception as e:
            self.logger.warning(f"Ошибка при очистке модулей: {e}")
        
        after = self.get_memory_usage()
        
        return {
            'before_mb': before['rss_mb'],
            'after_mb': after['rss_mb'],
            'freed_mb': before['rss_mb'] - after['rss_mb'],
            'collected_objects': collected
        }
    
    def should_optimize(self) -> bool:
        """Проверить, нужно ли оптимизировать память"""
        memory_usage = self.get_memory_usage()
        return memory_usage['rss_mb'] > self.threshold_mb
    
    def start_monitoring(self):
        """Запустить мониторинг памяти"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Мониторинг памяти запущен")
    
    def stop_monitoring(self):
        """Остановить мониторинг памяти"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Мониторинг памяти остановлен")
    
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.is_monitoring:
            try:
                if self.should_optimize():
                    self.logger.warning("Превышен порог памяти, запуск оптимизации...")
                    result = self.cleanup_memory()
                    self.logger.info(f"Оптимизация завершена: освобождено {result['freed_mb']:.2f} МБ")
                
                time.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Ошибка в мониторинге памяти: {e}")
                time.sleep(60)  # Пауза при ошибке
    
    def get_system_stats(self) -> Dict[str, any]:
        """Получить статистику системы"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'percent': memory.percent
            },
            'cpu_percent': cpu,
            'process_memory': self.get_memory_usage()
        }

def main():
    """Тестирование оптимизатора памяти"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = MemoryOptimizer(threshold_mb=200, cleanup_interval=30)
    
    print("🧠 Rubin AI Memory Optimizer")
    print("=" * 40)
    
    # Показать текущую статистику
    stats = optimizer.get_system_stats()
    print(f"📊 Текущая статистика:")
    print(f"  Память системы: {stats['memory']['used_mb']:.1f} МБ / {stats['memory']['total_mb']:.1f} МБ ({stats['memory']['percent']:.1f}%)")
    print(f"  Память процесса: {stats['process_memory']['rss_mb']:.1f} МБ")
    print(f"  CPU: {stats['cpu_percent']:.1f}%")
    
    # Запустить мониторинг
    optimizer.start_monitoring()
    
    try:
        print("\n⏳ Мониторинг запущен. Нажмите Ctrl+C для остановки...")
        while True:
            time.sleep(10)
            stats = optimizer.get_system_stats()
            print(f"📈 Память процесса: {stats['process_memory']['rss_mb']:.1f} МБ")
    except KeyboardInterrupt:
        print("\n🛑 Остановка мониторинга...")
        optimizer.stop_monitoring()
        print("✅ Мониторинг остановлен")

if __name__ == "__main__":
    main()







