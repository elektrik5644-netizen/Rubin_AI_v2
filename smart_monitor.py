#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Умная система мониторинга и автоперезапуска для Rubin AI v2
"""

import subprocess
import time
import requests
import logging
import psutil
import os
import signal
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartMonitor:
    def __init__(self):
        self.running = True
        self.processes = {}
        
        # Конфигурация модулей
        self.modules = {
            "smart_dispatcher": {
                "path": "smart_dispatcher.py",
                "port": 8080,
                "health_endpoint": "/api/health",
                "startup_delay": 5
            },
            "electrical": {
                "path": "api/electrical_api.py",
                "port": 8087,
                "health_endpoint": "/health",
                "startup_delay": 3
            },
            "radiomechanics": {
                "path": "api/radiomechanics_api.py",
                "port": 8089,
                "health_endpoint": "/health",
                "startup_delay": 3
            },
            "controllers": {
                "path": "api/controllers_api.py",
                "port": 9000,
                "health_endpoint": "/health",
                "startup_delay": 3
            },
            "mathematics": {
                "path": "api/mathematics_api.py",
                "port": 8086,
                "health_endpoint": "/health",
                "startup_delay": 3
            },
            "programming": {
                "path": "api/programming_api.py",
                "port": 8088,
                "health_endpoint": "/health",
                "startup_delay": 3
            },
            "general": {
                "path": "api/general_api.py",
                "port": 8085,
                "health_endpoint": "/health",
                "startup_delay": 3
            },
            "localai": {
                "path": "simple_localai_server.py",
                "port": 11434,
                "health_endpoint": "/health",
                "startup_delay": 3
            },
            "neuro": {
                "path": "api/neuro_repository_api.py",
                "port": 8090,
                "health_endpoint": "/health",
                "startup_delay": 5
            }
        }
        
        # Счетчики неудачных попыток
        self.failure_count = {name: 0 for name in self.modules.keys()}
        self.max_failures = 5  # Увеличиваем до 5 попыток
        self.restart_count = {name: 0 for name in self.modules.keys()}
        self.max_restarts = 5
        
        # Настройка обработчика сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        logger.info("🛑 Получен сигнал остановки...")
        self.running = False
        self.stop_all()

    def kill_port_process(self, port):
        """Завершение процесса на указанном порту"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    connections = proc.net_connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == port:
                            logger.info(f"🛑 Завершаю процесс {proc.pid} на порту {port}")
                            proc.terminate()
                            time.sleep(1)
                            if proc.is_running():
                                proc.kill()
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
                    pass
        except Exception as e:
            logger.error(f"Ошибка при завершении процесса на порту {port}: {e}")
        return False

    def start_module(self, name):
        """Запуск модуля"""
        if name in self.processes and self.processes[name].poll() is None:
            logger.info(f"✅ {name} уже запущен")
            return True
            
        module = self.modules[name]
        script_path = module["path"]
        port = module["port"]
        
        # Завершаем процесс на порту, если он занят
        self.kill_port_process(port)
        time.sleep(1)
        
        logger.info(f"🚀 Запускаю {name} (порт {port})...")
        try:
            process = subprocess.Popen(
                f"python {script_path}",
                shell=True,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes[name] = process
            logger.info(f"✅ {name} запущен (PID: {process.pid})")
            
            # Ждем запуска
            time.sleep(module["startup_delay"])
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска {name}: {e}")
            return False

    def stop_module(self, name):
        """Остановка модуля"""
        if name not in self.processes:
            return
            
        process = self.processes[name]
        if process.poll() is None:
            logger.info(f"🛑 Останавливаю {name}...")
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                logger.info(f"✅ {name} остановлен")
            except Exception as e:
                logger.error(f"❌ Ошибка остановки {name}: {e}")
        
        del self.processes[name]

    def check_health(self, name):
        """Проверка здоровья модуля"""
        module = self.modules[name]
        try:
            url = f"http://localhost:{module['port']}{module['health_endpoint']}"
            response = requests.get(url, timeout=10)
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
        if self.restart_count[name] >= self.max_restarts:
            logger.error(f"❌ {name} превысил лимит перезапусков ({self.max_restarts})")
            return False
            
        self.restart_count[name] += 1
        logger.warning(f"🔄 Перезапускаю {name} (попытка {self.restart_count[name]}/{self.max_restarts})...")
        
        self.stop_module(name)
        time.sleep(3)
        success = self.start_module(name)
        
        if success:
            self.failure_count[name] = 0  # Сбрасываем счетчик неудач
            
        return success

    def start_all(self):
        """Запуск всех модулей"""
        logger.info("🚀 Запускаю все модули Rubin AI v2...")
        
        for name in self.modules.keys():
            self.start_module(name)
            time.sleep(2)
        
        logger.info("✅ Все модули запущены")

    def monitor(self):
        """Мониторинг и автоперезапуск"""
        logger.info("Начинаю умный мониторинг...")
        
        while self.running:
            try:
                for name in self.modules.keys():
                    if not self.running:
                        break
                        
                    # Проверяем, запущен ли модуль
                    if not self.is_running(name):
                        logger.warning(f"⚠️ {name} не запущен, перезапускаю...")
                        self.restart_module(name)
                        continue
                    
                    # Проверяем здоровье
                    if not self.check_health(name):
                        self.failure_count[name] += 1
                        if self.failure_count[name] >= self.max_failures:
                            logger.warning(f"⚠️ {name} не отвечает {self.failure_count[name]} раз подряд, перезапускаю...")
                            self.restart_module(name)
                        else:
                            logger.debug(f"⚠️ {name} не отвечает ({self.failure_count[name]}/{self.max_failures})")
                        continue
                    else:
                        # Если модуль здоров, сбрасываем счетчик
                        self.failure_count[name] = 0
                
                # Ждем 60 секунд
                time.sleep(120)  # Увеличиваем интервал до 2 минут
                
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в мониторинге: {e}")
                time.sleep(10)

    def stop_all(self):
        """Остановка всех модулей"""
        logger.info("🛑 Останавливаю все модули...")
        for name in list(self.processes.keys()):
            self.stop_module(name)
        logger.info("✅ Все модули остановлены")

    def status_report(self):
        """Отчет о статусе системы"""
        logger.info("Отчет о статусе системы:")
        for name in self.modules.keys():
            status = "ОНЛАЙН" if self.is_running(name) and self.check_health(name) else "ОФФЛАЙН"
            restarts = self.restart_count[name]
            failures = self.failure_count[name]
            logger.info(f"  {name}: {status} (перезапусков: {restarts}, неудач: {failures})")

def main():
    monitor = SmartMonitor()
    
    try:
        # Запускаем все модули
        monitor.start_all()
        
        # Ждем немного для стабилизации
        time.sleep(10)
        
        # Показываем отчет о статусе
        monitor.status_report()
        
        # Начинаем мониторинг
        monitor.monitor()
        
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал остановки...")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    finally:
        monitor.stop_all()
        logger.info("👋 Мониторинг завершен")

if __name__ == "__main__":
    main()
