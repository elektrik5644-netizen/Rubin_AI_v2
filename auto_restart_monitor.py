#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система автоматического мониторинга и перезапуска модулей Rubin AI v2
"""

import subprocess
import time
import requests
import logging
import psutil
import signal
import os
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_restart.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModuleMonitor:
    def __init__(self):
        self.modules = {
            'smart_dispatcher': {
                'port': 8080,
                'script': 'smart_dispatcher.py',
                'health_url': 'http://localhost:8080/api/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            },
            'electrical': {
                'port': 8087,
                'script': 'api/electrical_api.py',
                'health_url': 'http://localhost:8087/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            },
            'radiomechanics': {
                'port': 8089,
                'script': 'api/radiomechanics_api.py',
                'health_url': 'http://localhost:8089/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            },
            'controllers': {
                'port': 9000,
                'script': 'api/controllers_api.py',
                'health_url': 'http://localhost:9000/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            },
            'mathematics': {
                'port': 8086,
                'script': 'api/mathematics_api.py',
                'health_url': 'http://localhost:8086/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            },
            'programming': {
                'port': 8088,
                'script': 'api/programming_api.py',
                'health_url': 'http://localhost:8088/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            },
            'general': {
                'port': 8085,
                'script': 'api/general_api.py',
                'health_url': 'http://localhost:8085/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            },
            'localai': {
                'port': 11434,
                'script': 'simple_localai_server.py',
                'health_url': 'http://localhost:11434/health',
                'process': None,
                'pid': None,
                'restart_count': 0,
                'max_restarts': 5,
                'last_restart': None
            }
        }
        
        self.running = True
        self.check_interval = 30  # Проверка каждые 30 секунд
        self.startup_delay = 10   # Задержка перед проверкой после запуска

    def start_module(self, module_name):
        """Запуск модуля"""
        module = self.modules[module_name]
        
        try:
            logger.info(f"🚀 Запускаю {module_name} (порт {module['port']})...")
            
            # Запускаем модуль в фоновом режиме
            process = subprocess.Popen(
                ['python', module['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            module['process'] = process
            module['pid'] = process.pid
            module['last_restart'] = datetime.now()
            
            logger.info(f"✅ {module_name} запущен (PID: {process.pid})")
            
            # Ждем немного перед проверкой здоровья
            time.sleep(self.startup_delay)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска {module_name}: {str(e)}")
            return False

    def stop_module(self, module_name):
        """Остановка модуля"""
        module = self.modules[module_name]
        
        if module['process'] and module['process'].poll() is None:
            try:
                logger.info(f"🛑 Останавливаю {module_name} (PID: {module['pid']})...")
                
                # Мягкая остановка
                if os.name == 'nt':
                    # Windows
                    subprocess.run(['taskkill', '/F', '/PID', str(module['pid'])], 
                                 capture_output=True)
                else:
                    # Linux/Mac
                    os.kill(module['pid'], signal.SIGTERM)
                
                # Ждем завершения
                try:
                    module['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Принудительная остановка
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/F', '/PID', str(module['pid'])], 
                                     capture_output=True)
                    else:
                        os.kill(module['pid'], signal.SIGKILL)
                
                logger.info(f"✅ {module_name} остановлен")
                
            except Exception as e:
                logger.error(f"❌ Ошибка остановки {module_name}: {str(e)}")
        
        module['process'] = None
        module['pid'] = None

    def check_module_health(self, module_name):
        """Проверка здоровья модуля"""
        module = self.modules[module_name]
        
        try:
            response = requests.get(module['health_url'], timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Модуль {module_name} недоступен: {str(e)}")
            return False

    def is_module_running(self, module_name):
        """Проверка, запущен ли модуль"""
        module = self.modules[module_name]
        
        if module['process'] is None:
            return False
        
        # Проверяем, что процесс еще работает
        if module['process'].poll() is not None:
            return False
        
        # Проверяем, что процесс действительно существует
        try:
            if module['pid']:
                process = psutil.Process(module['pid'])
                return process.is_running()
        except psutil.NoSuchProcess:
            pass
        
        return False

    def restart_module(self, module_name):
        """Перезапуск модуля"""
        module = self.modules[module_name]
        
        # Проверяем лимит перезапусков
        if module['restart_count'] >= module['max_restarts']:
            logger.error(f"❌ {module_name} превысил лимит перезапусков ({module['max_restarts']})")
            return False
        
        logger.warning(f"🔄 Перезапускаю {module_name} (попытка {module['restart_count'] + 1}/{module['max_restarts']})")
        
        # Останавливаем модуль
        self.stop_module(module_name)
        
        # Ждем немного
        time.sleep(5)
        
        # Запускаем модуль
        if self.start_module(module_name):
            module['restart_count'] += 1
            logger.info(f"✅ {module_name} успешно перезапущен")
            return True
        else:
            logger.error(f"❌ Не удалось перезапустить {module_name}")
            return False

    def monitor_modules(self):
        """Основной цикл мониторинга"""
        logger.info("🔍 Начинаю мониторинг модулей Rubin AI v2...")
        
        while self.running:
            try:
                for module_name in self.modules.keys():
                    # Проверяем, запущен ли модуль
                    if not self.is_module_running(module_name):
                        logger.warning(f"⚠️ Модуль {module_name} не запущен")
                        self.restart_module(module_name)
                        continue
                    
                    # Проверяем здоровье модуля
                    if not self.check_module_health(module_name):
                        logger.warning(f"⚠️ Модуль {module_name} не отвечает на health check")
                        self.restart_module(module_name)
                        continue
                    
                    # Сбрасываем счетчик перезапусков при успешной работе
                    if self.modules[module_name]['restart_count'] > 0:
                        logger.info(f"✅ {module_name} работает стабильно, сбрасываю счетчик перезапусков")
                        self.modules[module_name]['restart_count'] = 0
                
                # Ждем перед следующей проверкой
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле мониторинга: {str(e)}")
                time.sleep(10)

    def start_all_modules(self):
        """Запуск всех модулей"""
        logger.info("🚀 Запускаю все модули Rubin AI v2...")
        
        for module_name in self.modules.keys():
            self.start_module(module_name)
            time.sleep(2)  # Небольшая задержка между запусками
        
        logger.info("✅ Все модули запущены")

    def stop_all_modules(self):
        """Остановка всех модулей"""
        logger.info("🛑 Останавливаю все модули...")
        
        for module_name in self.modules.keys():
            self.stop_module(module_name)
        
        logger.info("✅ Все модули остановлены")

    def get_status(self):
        """Получение статуса всех модулей"""
        status = {}
        
        for module_name, module in self.modules.items():
            is_running = self.is_module_running(module_name)
            is_healthy = self.check_module_health(module_name) if is_running else False
            
            status[module_name] = {
                'running': is_running,
                'healthy': is_healthy,
                'port': module['port'],
                'pid': module['pid'],
                'restart_count': module['restart_count'],
                'last_restart': module['last_restart']
            }
        
        return status

def main():
    """Главная функция"""
    monitor = ModuleMonitor()
    
    try:
        # Запускаем все модули
        monitor.start_all_modules()
        
        # Начинаем мониторинг
        monitor.monitor_modules()
        
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал остановки...")
    finally:
        # Останавливаем все модули
        monitor.stop_all_modules()
        logger.info("👋 Мониторинг завершен")

if __name__ == "__main__":
    main()











