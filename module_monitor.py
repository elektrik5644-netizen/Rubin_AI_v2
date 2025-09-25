#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система мониторинга и автоматического перезапуска модулей Rubin AI
"""

import time
import subprocess
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rubin_monitor")

class ModuleMonitor:
    """Мониторинг и управление модулями Rubin AI"""
    
    def __init__(self):
        self.modules = {
            'general': {'port': 8085, 'script': 'api/general_api.py', 'process': None},
            'mathematics': {'port': 8086, 'script': 'api/mathematics_api.py', 'process': None},
            'electrical': {'port': 8087, 'script': 'api/electrical_api.py', 'process': None},
            'programming': {'port': 8088, 'script': 'api/programming_api.py', 'process': None},
            'controllers': {'port': 9000, 'script': 'controllers_server.py', 'process': None}
        }
        self.restart_attempts = {name: 0 for name in self.modules.keys()}
        self.max_restart_attempts = 3
        self.check_interval = 30  # секунд
        
    def check_module_health(self, module_name: str) -> bool:
        """Проверяет здоровье модуля"""
        try:
            port = self.modules[module_name]['port']
            response = requests.get(f"http://localhost:{port}/api/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"❌ Модуль {module_name} недоступен: {e}")
            return False
    
    def start_module(self, module_name: str) -> bool:
        """Запускает модуль"""
        try:
            script_path = self.modules[module_name]['script']
            logger.info(f"🚀 Запускаю модуль {module_name} ({script_path})")
            
            # Запускаем модуль в фоновом режиме
            process = subprocess.Popen(
                ['python', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            self.modules[module_name]['process'] = process
            
            # Ждем немного для инициализации
            time.sleep(5)
            
            # Проверяем, что модуль запустился
            if self.check_module_health(module_name):
                logger.info(f"✅ Модуль {module_name} успешно запущен")
                self.restart_attempts[module_name] = 0
                return True
            else:
                logger.error(f"❌ Модуль {module_name} не отвечает после запуска")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска модуля {module_name}: {e}")
            return False
    
    def restart_module(self, module_name: str) -> bool:
        """Перезапускает модуль"""
        logger.info(f"🔄 Перезапускаю модуль {module_name}")
        
        # Останавливаем текущий процесс
        if self.modules[module_name]['process']:
            try:
                self.modules[module_name]['process'].terminate()
                time.sleep(2)
            except:
                pass
        
        # Увеличиваем счетчик попыток
        self.restart_attempts[module_name] += 1
        
        if self.restart_attempts[module_name] > self.max_restart_attempts:
            logger.error(f"❌ Превышено максимальное количество попыток перезапуска для {module_name}")
            return False
        
        # Запускаем модуль
        return self.start_module(module_name)
    
    def check_all_modules(self) -> Dict[str, Any]:
        """Проверяет все модули и перезапускает недоступные"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'modules': {},
            'actions_taken': []
        }
        
        for module_name in self.modules.keys():
            is_healthy = self.check_module_health(module_name)
            
            results['modules'][module_name] = {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'port': self.modules[module_name]['port'],
                'restart_attempts': self.restart_attempts[module_name]
            }
            
            if not is_healthy:
                logger.warning(f"⚠️ Модуль {module_name} недоступен, пытаюсь перезапустить")
                if self.restart_module(module_name):
                    results['actions_taken'].append(f"Перезапущен модуль {module_name}")
                else:
                    results['actions_taken'].append(f"Не удалось перезапустить модуль {module_name}")
        
        return results
    
    def start_all_modules(self):
        """Запускает все модули"""
        logger.info("🚀 Запускаю все модули Rubin AI")
        
        for module_name in self.modules.keys():
            if not self.check_module_health(module_name):
                self.start_module(module_name)
            else:
                logger.info(f"✅ Модуль {module_name} уже запущен")
    
    def monitor_loop(self):
        """Основной цикл мониторинга"""
        logger.info("🔍 Запуск мониторинга модулей Rubin AI")
        logger.info(f"Интервал проверки: {self.check_interval} секунд")
        
        while True:
            try:
                results = self.check_all_modules()
                
                # Логируем результаты
                healthy_count = sum(1 for m in results['modules'].values() if m['status'] == 'healthy')
                total_count = len(results['modules'])
                
                logger.info(f"📊 Статус модулей: {healthy_count}/{total_count} здоровы")
                
                if results['actions_taken']:
                    for action in results['actions_taken']:
                        logger.info(f"🔧 {action}")
                
                # Сохраняем результаты в файл
                with open('monitor_results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Остановка мониторинга по запросу пользователя")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле мониторинга: {e}")
                time.sleep(10)

def main():
    monitor = ModuleMonitor()
    
    # Запускаем все модули
    monitor.start_all_modules()
    
    # Запускаем мониторинг
    monitor.monitor_loop()

if __name__ == "__main__":
    main()

