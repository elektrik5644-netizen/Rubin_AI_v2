#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматический перезапуск Simple Dispatcher
Обеспечивает максимальную надежность работы
"""

import subprocess
import time
import logging
import requests
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DispatcherManager:
    def __init__(self):
        self.process = None
        self.restart_count = 0
        self.max_restarts = 10  # Максимум перезапусков за сессию
        self.restart_delay = 5   # Задержка между перезапусками
        self.health_check_url = "http://localhost:8080/api/health"
        
    def start_dispatcher(self):
        """Запуск диспетчера."""
        try:
            logger.info("🚀 Запуск Simple Dispatcher...")
            self.process = subprocess.Popen(
                ['python', 'simple_dispatcher.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"✅ Dispatcher запущен с PID: {self.process.pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка запуска: {e}")
            return False
    
    def stop_dispatcher(self):
        """Остановка диспетчера."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("🛑 Dispatcher остановлен")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning("⚠️ Принудительная остановка Dispatcher")
            except Exception as e:
                logger.error(f"❌ Ошибка остановки: {e}")
    
    def check_health(self):
        """Проверка здоровья диспетчера."""
        try:
            response = requests.get(self.health_check_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def restart_dispatcher(self):
        """Перезапуск диспетчера."""
        if self.restart_count >= self.max_restarts:
            logger.error(f"❌ Достигнут лимит перезапусков ({self.max_restarts})")
            return False
            
        self.restart_count += 1
        logger.warning(f"🔄 Перезапуск #{self.restart_count}")
        
        self.stop_dispatcher()
        time.sleep(self.restart_delay)
        
        if self.start_dispatcher():
            logger.info("✅ Перезапуск успешен")
            return True
        else:
            logger.error("❌ Ошибка перезапуска")
            return False
    
    def run(self):
        """Основной цикл работы."""
        logger.info("🎯 Запуск Dispatcher Manager")
        logger.info(f"📊 Максимум перезапусков: {self.max_restarts}")
        logger.info(f"⏰ Интервал проверки: 30 секунд")
        
        # Первоначальный запуск
        if not self.start_dispatcher():
            logger.error("❌ Не удалось запустить Dispatcher")
            return
        
        # Ожидание запуска
        time.sleep(10)
        
        # Основной цикл мониторинга
        while True:
            try:
                if not self.check_health():
                    logger.warning("⚠️ Dispatcher не отвечает")
                    if not self.restart_dispatcher():
                        logger.error("❌ Критическая ошибка - остановка мониторинга")
                        break
                else:
                    logger.info("✅ Dispatcher работает нормально")
                
                # Проверка каждые 30 секунд
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки")
                break
            except Exception as e:
                logger.error(f"❌ Неожиданная ошибка: {e}")
                time.sleep(10)
        
        # Остановка при выходе
        self.stop_dispatcher()
        logger.info("👋 Dispatcher Manager завершен")

if __name__ == '__main__':
    manager = DispatcherManager()
    manager.run()



