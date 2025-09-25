#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Memory Optimized Launcher - Запуск серверов с оптимизацией памяти
"""

import subprocess
import time
import psutil
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizedLauncher:
    def __init__(self):
        self.servers = {
            'smart_dispatcher': {
                'command': 'python simple_dispatcher.py',
                'port': 8080,
                'memory_limit': 100,  # MB
                'priority': 1
            },
            'general_api': {
                'command': 'python api/general_api.py',
                'port': 8085,
                'memory_limit': 150,
                'priority': 1
            },
            'mathematics': {
                'command': 'python api/mathematics_api.py',
                'port': 8086,
                'memory_limit': 200,
                'priority': 2
            },
            'electrical': {
                'command': 'python api/electrical_api.py',
                'port': 8087,
                'memory_limit': 150,
                'priority': 2
            },
            'programming': {
                'command': 'python api/programming_api.py',
                'port': 8088,
                'memory_limit': 200,
                'priority': 2
            },
            'neuro': {
                'command': 'python simple_neuro_api_server.py',
                'port': 8090,
                'memory_limit': 100,
                'priority': 3
            },
            'controllers': {
                'command': 'python simple_controllers_api_server.py',
                'port': 9000,
                'memory_limit': 100,
                'priority': 3
            },
            'plc_analysis': {
                'command': 'python simple_plc_analysis_api_server.py',
                'port': 8099,
                'memory_limit': 100,
                'priority': 3
            },
            'advanced_math': {
                'command': 'python simple_advanced_math_api_server.py',
                'port': 8100,
                'memory_limit': 150,
                'priority': 3
            },
            'data_processing': {
                'command': 'python simple_data_processing_api_server.py',
                'port': 8101,
                'memory_limit': 150,
                'priority': 3
            },
            'search_engine': {
                'command': 'python search_engine_api_server.py',
                'port': 8102,
                'memory_limit': 200,
                'priority': 3
            },
            'system_utils': {
                'command': 'python system_utils_api_server.py',
                'port': 8103,
                'memory_limit': 100,
                'priority': 3
            },
            'gai_server': {
                'command': 'python lightweight_gai_server.py',  # Используем оптимизированную версию
                'port': 8104,
                'memory_limit': 100,
                'priority': 3
            },
            'unified_manager': {
                'command': 'python unified_system_manager.py',
                'port': 8084,
                'memory_limit': 150,
                'priority': 2
            },
            'ethical_core': {
                'command': 'python ethical_core_api_server.py',
                'port': 8105,
                'memory_limit': 100,
                'priority': 3
            }
        }
        
        self.running_processes = {}
    
    def check_memory_available(self, required_mb):
        """Проверить доступность памяти"""
        memory = psutil.virtual_memory()
        available_mb = memory.available // (1024**2)
        return available_mb >= required_mb
    
    def get_total_memory_usage(self):
        """Получить общее потребление памяти Python процессами"""
        total_memory = 0
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    total_memory += proc.info['memory_info'].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_memory // (1024**2)  # MB
    
    def start_server(self, server_name):
        """Запустить сервер с проверкой памяти"""
        if server_name not in self.servers:
            logger.error(f"❌ Неизвестный сервер: {server_name}")
            return False
        
        server_config = self.servers[server_name]
        
        # Проверка доступности памяти
        if not self.check_memory_available(server_config['memory_limit']):
            logger.warning(f"⚠️  Недостаточно памяти для {server_name} (требуется {server_config['memory_limit']}MB)")
            return False
        
        try:
            # Запуск сервера
            process = subprocess.Popen(
                server_config['command'],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.running_processes[server_name] = process
            logger.info(f"✅ {server_name} запущен (PID: {process.pid})")
            
            # Небольшая задержка для инициализации
            time.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска {server_name}: {e}")
            return False
    
    def start_priority_servers(self):
        """Запустить серверы по приоритету"""
        # Сортируем серверы по приоритету
        sorted_servers = sorted(
            self.servers.items(),
            key=lambda x: x[1]['priority']
        )
        
        started_count = 0
        total_memory_used = 0
        
        for server_name, config in sorted_servers:
            # Проверяем общее потребление памяти
            current_memory = self.get_total_memory_usage()
            if current_memory > 4000:  # Лимит 4GB
                logger.warning(f"⚠️  Достигнут лимит памяти ({current_memory}MB), останавливаем запуск")
                break
            
            if self.start_server(server_name):
                started_count += 1
                total_memory_used += config['memory_limit']
                logger.info(f"📊 Запущено серверов: {started_count}, Память: {total_memory_used}MB")
        
        return started_count
    
    def monitor_servers(self):
        """Мониторинг запущенных серверов"""
        logger.info("🔍 Начинаем мониторинг серверов...")
        
        while True:
            try:
                # Проверяем статус процессов
                for server_name, process in list(self.running_processes.items()):
                    if process.poll() is not None:
                        logger.warning(f"⚠️  {server_name} завершился неожиданно")
                        del self.running_processes[server_name]
                
                # Показываем статистику
                memory_usage = self.get_total_memory_usage()
                logger.info(f"📊 Активных серверов: {len(self.running_processes)}, Память: {memory_usage}MB")
                
                time.sleep(30)  # Проверка каждые 30 секунд
                
            except KeyboardInterrupt:
                logger.info("🛑 Остановка мониторинга...")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка мониторинга: {e}")
                time.sleep(10)
    
    def stop_all_servers(self):
        """Остановить все серверы"""
        logger.info("🛑 Останавливаем все серверы...")
        
        for server_name, process in self.running_processes.items():
            try:
                process.terminate()
                logger.info(f"✅ {server_name} остановлен")
            except Exception as e:
                logger.error(f"❌ Ошибка остановки {server_name}: {e}")
        
        self.running_processes.clear()

def main():
    launcher = MemoryOptimizedLauncher()
    
    try:
        logger.info("🚀 Memory Optimized Launcher запущен")
        logger.info("=" * 50)
        
        # Показываем доступную память
        memory = psutil.virtual_memory()
        available_gb = memory.available // (1024**3)
        logger.info(f"💾 Доступно памяти: {available_gb}GB")
        
        if available_gb < 2:
            logger.warning("⚠️  Мало доступной памяти! Рекомендуется освободить RAM")
        
        # Запускаем серверы
        started = launcher.start_priority_servers()
        logger.info(f"✅ Запущено {started} серверов")
        
        # Мониторинг
        launcher.monitor_servers()
        
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал остановки")
    finally:
        launcher.stop_all_servers()
        logger.info("🏁 Launcher завершен")

if __name__ == '__main__':
    main()



