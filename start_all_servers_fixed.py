#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматический запуск всех серверов Rubin AI v2
Исправляет проблемы с падением серверов
"""

import subprocess
import time
import requests
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerManager:
    def __init__(self):
        self.servers = {
            'smart_dispatcher': {
                'script': 'simple_dispatcher.py',
                'port': 8080,
                'health_url': 'http://localhost:8080/api/health',
                'description': 'Smart Dispatcher'
            },
            'general_api': {
                'script': 'api/general_api.py',
                'port': 8085,
                'health_url': 'http://localhost:8085/api/health',
                'description': 'General API'
            },
            'mathematics_api': {
                'script': 'api/mathematics_api.py',
                'port': 8086,
                'health_url': 'http://localhost:8086/health',
                'description': 'Mathematics API'
            },
            'electrical_api': {
                'script': 'api/electrical_api.py',
                'port': 8087,
                'health_url': 'http://localhost:8087/api/electrical/status',
                'description': 'Electrical API'
            },
            'programming_api': {
                'script': 'api/programming_api.py',
                'port': 8088,
                'health_url': 'http://localhost:8088/health',
                'description': 'Programming API'
            },
            'radiomechanics_api': {
                'script': 'api/radiomechanics_api.py',
                'port': 8089,
                'health_url': 'http://localhost:8089/api/radiomechanics/status',
                'description': 'Radiomechanics API'
            },
            'neuro_api': {
                'script': 'api/neuro_repository_api.py',
                'port': 8090,
                'health_url': 'http://localhost:8090/api/health',
                'description': 'Neuro API'
            },
            'plc_analysis_api': {
                'script': 'plc_analysis_api_server.py',
                'port': 8099,
                'health_url': 'http://localhost:8099/api/plc/health',
                'description': 'PLC Analysis API'
            },
            'advanced_math_api': {
                'script': 'advanced_math_api_server.py',
                'port': 8100,
                'health_url': 'http://localhost:8100/api/math/health',
                'description': 'Advanced Math API'
            },
            'data_processing_api': {
                'script': 'data_processing_api_server.py',
                'port': 8101,
                'health_url': 'http://localhost:8101/api/data/health',
                'description': 'Data Processing API'
            },
            'search_engine_api': {
                'script': 'search_engine_api_server.py',
                'port': 8102,
                'health_url': 'http://localhost:8102/api/search/health',
                'description': 'Search Engine API'
            },
            'system_utils_api': {
                'script': 'system_utils_api_server.py',
                'port': 8103,
                'health_url': 'http://localhost:8103/api/system/health',
                'description': 'System Utils API'
            },
            'gai_api': {
                'script': 'enhanced_gai_server.py',
                'port': 8104,
                'health_url': 'http://localhost:8104/api/gai/health',
                'description': 'GAI Server'
            },
            'unified_manager': {
                'script': 'unified_system_manager.py',
                'port': 8084,
                'health_url': 'http://localhost:8084/api/system/health',
                'description': 'Unified Manager'
            },
            'ethical_core_api': {
                'script': 'ethical_core_api_server.py',
                'port': 8105,
                'health_url': 'http://localhost:8105/api/ethical/health',
                'description': 'Ethical Core API'
            },
            'logic_tasks_api': {
                'script': 'logic_tasks_api_server.py',
                'port': 8106,
                'health_url': 'http://localhost:8106/api/logic/health',
                'description': 'Logic Tasks API'
            }
        }
        self.processes = {}
    
    def start_server(self, server_name):
        """Запускает сервер."""
        if server_name not in self.servers:
            logger.error(f"❌ Неизвестный сервер: {server_name}")
            return False
        
        server_config = self.servers[server_name]
        
        try:
            logger.info(f"🚀 Запуск {server_config['description']}...")
            process = subprocess.Popen(
                ['python', server_config['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[server_name] = process
            logger.info(f"✅ {server_config['description']} запущен с PID: {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска {server_config['description']}: {e}")
            return False
    
    def check_server_health(self, server_name):
        """Проверяет здоровье сервера."""
        if server_name not in self.servers:
            return False
        
        server_config = self.servers[server_name]
        
        try:
            response = requests.get(server_config['health_url'], timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_all_servers(self):
        """Запускает все серверы."""
        logger.info("🚀 Запуск всех серверов Rubin AI v2")
        logger.info("=" * 60)
        
        started_count = 0
        
        for server_name in self.servers.keys():
            if self.start_server(server_name):
                started_count += 1
            time.sleep(2)  # Пауза между запусками
        
        logger.info(f"✅ Запущено серверов: {started_count}/{len(self.servers)}")
        
        # Ожидание запуска
        logger.info("⏳ Ожидание запуска серверов...")
        time.sleep(10)
        
        # Проверка здоровья
        logger.info("🔍 Проверка здоровья серверов...")
        healthy_count = 0
        
        for server_name in self.servers.keys():
            if self.check_server_health(server_name):
                logger.info(f"✅ {self.servers[server_name]['description']} - здоров")
                healthy_count += 1
            else:
                logger.warning(f"⚠️ {self.servers[server_name]['description']} - не отвечает")
        
        logger.info("=" * 60)
        logger.info(f"📊 Статистика: {healthy_count}/{len(self.servers)} серверов здоровы")
        
        if healthy_count >= len(self.servers) * 0.8:
            logger.info("🎉 Система запущена успешно!")
        elif healthy_count >= len(self.servers) * 0.6:
            logger.info("👍 Система работает с незначительными проблемами")
        else:
            logger.warning("⚠️ Система требует внимания")
        
        return healthy_count
    
    def stop_all_servers(self):
        """Останавливает все серверы."""
        logger.info("🛑 Остановка всех серверов...")
        
        for server_name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ {self.servers[server_name]['description']} остановлен")
            except:
                try:
                    process.kill()
                    logger.warning(f"⚠️ {self.servers[server_name]['description']} принудительно остановлен")
                except:
                    logger.error(f"❌ Не удалось остановить {self.servers[server_name]['description']}")
        
        self.processes.clear()

def main():
    """Основная функция."""
    manager = ServerManager()
    
    try:
        healthy_count = manager.start_all_servers()
        
        if healthy_count > 0:
            logger.info("🎯 Система готова к работе!")
            logger.info("🌐 Smart Dispatcher: http://localhost:8080")
            logger.info("📱 RubinDeveloper: http://localhost:8080/matrix/RubinDeveloper.html")
            logger.info("🧠 Logic Tasks: http://localhost:8106")
            
            # Держим систему работающей
            logger.info("⏳ Система работает... Нажмите Ctrl+C для остановки")
            try:
                while True:
                    time.sleep(30)
                    # Периодическая проверка здоровья
                    healthy_count = sum(1 for server_name in manager.servers.keys() 
                                     if manager.check_server_health(server_name))
                    logger.info(f"📊 Статус: {healthy_count}/{len(manager.servers)} серверов онлайн")
            except KeyboardInterrupt:
                logger.info("🛑 Получен сигнал остановки")
        else:
            logger.error("❌ Не удалось запустить ни одного сервера")
    
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    
    finally:
        manager.stop_all_servers()
        logger.info("👋 Все серверы остановлены")

if __name__ == '__main__':
    main()



