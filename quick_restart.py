#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый запуск Rubin AI v2 с автоперезапуском
"""

import subprocess
import time
import requests
import logging
import os
import signal
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickRestart:
    def __init__(self):
        self.modules = {
            'smart_dispatcher': {'port': 8080, 'script': 'smart_dispatcher.py'},
            'electrical': {'port': 8087, 'script': 'api/electrical_api.py'},
            'radiomechanics': {'port': 8089, 'script': 'api/radiomechanics_api.py'},
            'controllers': {'port': 9000, 'script': 'api/controllers_api.py'},
            'mathematics': {'port': 8086, 'script': 'api/mathematics_api.py'},
            'programming': {'port': 8088, 'script': 'api/programming_api.py'},
            'general': {'port': 8085, 'script': 'api/general_api.py'},
            'localai': {'port': 11434, 'script': 'simple_localai_server.py'}
        }
        
        self.processes = {}
        self.running = True

    def start_module(self, name):
        """Запуск модуля"""
        module = self.modules[name]
        try:
            logger.info(f"🚀 Запускаю {name} (порт {module['port']})...")
            
            process = subprocess.Popen(
                ['python', module['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.processes[name] = process
            logger.info(f"✅ {name} запущен (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска {name}: {str(e)}")
            return False

    def stop_module(self, name):
        """Остановка модуля"""
        if name in self.processes:
            process = self.processes[name]
            try:
                logger.info(f"🛑 Останавливаю {name}...")
                
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/PID', str(process.pid)], 
                                 capture_output=True)
                else:
                    os.kill(process.pid, signal.SIGTERM)
                
                del self.processes[name]
                logger.info(f"✅ {name} остановлен")
                
            except Exception as e:
                logger.error(f"❌ Ошибка остановки {name}: {str(e)}")

    def check_health(self, name):
        """Проверка здоровья модуля"""
        module = self.modules[name]
        try:
            response = requests.get(f"http://localhost:{module['port']}{module['health_endpoint']}", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {name}: {e}")
            return False

    def is_running(self, name):
        """Проверка, запущен ли модуль"""
        if name not in self.processes:
            return False
        
        process = self.processes[name]
        return process.poll() is None

    def restart_module(self, name):
        """Перезапуск модуля"""
        logger.warning(f"🔄 Перезапускаю {name}...")
        self.stop_module(name)
        time.sleep(3)
        self.start_module(name)

    def start_all(self):
        """Запуск всех модулей"""
        logger.info("🚀 Запускаю все модули Rubin AI v2...")
        
        for name in self.modules.keys():
            self.start_module(name)
            time.sleep(2)
        
        logger.info("✅ Все модули запущены")

    def monitor(self):
        """Мониторинг и автоперезапуск"""
        logger.info("🔍 Начинаю мониторинг...")
        
        # Счетчики неудачных попыток
        failure_count = {name: 0 for name in self.modules.keys()}
        max_failures = 3  # Максимум 3 неудачные попытки подряд
        
        while self.running:
            try:
                for name in self.modules.keys():
                    # Проверяем, запущен ли модуль
                    if not self.is_running(name):
                        logger.warning(f"⚠️ {name} не запущен, перезапускаю...")
                        self.restart_module(name)
                        failure_count[name] += 1
                        continue
                    
                    # Проверяем здоровье
                    if not self.check_health(name):
                        failure_count[name] += 1
                        if failure_count[name] >= max_failures:
                            logger.warning(f"⚠️ {name} не отвечает {failure_count[name]} раз подряд, перезапускаю...")
                            self.restart_module(name)
                            failure_count[name] = 0  # Сбрасываем счетчик после перезапуска
                        else:
                            logger.debug(f"⚠️ {name} не отвечает ({failure_count[name]}/{max_failures})")
                        continue
                    else:
                        # Если модуль здоров, сбрасываем счетчик
                        failure_count[name] = 0
                
                # Ждем 60 секунд (увеличили интервал)
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Ошибка мониторинга: {str(e)}")
                time.sleep(10)

    def stop_all(self):
        """Остановка всех модулей"""
        logger.info("🛑 Останавливаю все модули...")
        
        for name in self.processes.keys():
            self.stop_module(name)
        
        logger.info("✅ Все модули остановлены")

def main():
    """Главная функция"""
    restart = QuickRestart()
    
    try:
        # Запускаем все модули
        restart.start_all()
        
        # Ждем немного
        time.sleep(10)
        
        # Начинаем мониторинг
        restart.monitor()
        
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал остановки...")
    finally:
        restart.stop_all()
        logger.info("👋 Завершено")

if __name__ == "__main__":
    main()
