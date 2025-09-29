#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт запуска системы Rubin AI с интеграцией MCSetup
"""

import subprocess
import time
import requests
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_service(port, service_name, timeout=10):
    """Проверяет доступность сервиса"""
    try:
        response = requests.get(f"http://localhost:{port}/api/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def start_service(script_name, service_name, port=None):
    """Запускает сервис"""
    try:
        logger.info(f"🚀 Запуск {service_name}...")
        process = subprocess.Popen(
            ['python', script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Ждем запуска сервиса
        if port:
            for i in range(30):  # Ждем до 30 секунд
                if check_service(port, service_name, timeout=2):
                    logger.info(f"✅ {service_name} запущен на порту {port}")
                    return process
                time.sleep(1)
            
            logger.warning(f"⚠️ {service_name} не отвечает на порту {port}")
        
        return process
        
    except Exception as e:
        logger.error(f"❌ Ошибка запуска {service_name}: {e}")
        return None

def main():
    """Основная функция запуска системы"""
    logger.info("🚀 Запуск системы Rubin AI с интеграцией MCSetup")
    logger.info("=" * 60)
    
    # Список сервисов для запуска
    services = [
        {
            'script': 'smart_dispatcher.py',
            'name': 'Smart Dispatcher',
            'port': 8080,
            'required': True
        },
        {
            'script': 'mcsetup_bridge_server.py',
            'name': 'MCSetup Bridge',
            'port': 8096,
            'required': True
        },
        {
            'script': 'graph_analyzer_server.py',
            'name': 'Graph Analyzer',
            'port': 8097,
            'required': True
        },
        {
            'script': 'general_server.py',
            'name': 'General Server',
            'port': 8085,
            'required': False
        },
        {
            'script': 'math_server.py',
            'name': 'Math Server',
            'port': 8086,
            'required': False
        },
        {
            'script': 'electrical_server.py',
            'name': 'Electrical Server',
            'port': 8087,
            'required': False
        },
        {
            'script': 'programming_server.py',
            'name': 'Programming Server',
            'port': 8088,
            'required': False
        },
        {
            'script': 'controllers_server.py',
            'name': 'Controllers Server',
            'port': 9000,
            'required': False
        },
        {
            'script': 'gai_server.py',
            'name': 'GAI Server',
            'port': 8104,
            'required': False
        }
    ]
    
    processes = []
    
    # Запускаем обязательные сервисы
    for service in services:
        if service['required']:
            process = start_service(service['script'], service['name'], service['port'])
            if process:
                processes.append((process, service['name']))
            else:
                logger.error(f"❌ Не удалось запустить обязательный сервис {service['name']}")
                return False
    
    # Ждем запуска обязательных сервисов
    logger.info("⏳ Ожидание запуска обязательных сервисов...")
    time.sleep(5)
    
    # Проверяем статус обязательных сервисов
    for service in services:
        if service['required']:
            if check_service(service['port'], service['name']):
                logger.info(f"✅ {service['name']} работает")
            else:
                logger.error(f"❌ {service['name']} не отвечает")
    
    # Запускаем дополнительные сервисы
    logger.info("🔄 Запуск дополнительных сервисов...")
    for service in services:
        if not service['required']:
            process = start_service(service['script'], service['name'], service['port'])
            if process:
                processes.append((process, service['name']))
                time.sleep(2)  # Небольшая задержка между запусками
    
    # Финальная проверка
    logger.info("🔍 Финальная проверка сервисов...")
    time.sleep(3)
    
    for service in services:
        if check_service(service['port'], service['name']):
            logger.info(f"✅ {service['name']} ({service['port']}) - работает")
        else:
            logger.warning(f"⚠️ {service['name']} ({service['port']}) - не отвечает")
    
    # Тестируем интеграцию MCSetup
    logger.info("🧪 Тестирование интеграции MCSetup...")
    try:
        # Тест MCSetup Bridge
        response = requests.post(
            "http://localhost:8096/api/mcsetup/status",
            timeout=5
        )
        if response.status_code == 200:
            logger.info("✅ MCSetup Bridge - интеграция работает")
        else:
            logger.warning("⚠️ MCSetup Bridge - проблемы с интеграцией")
        
        # Тест Graph Analyzer
        response = requests.post(
            "http://localhost:8097/api/graph/health",
            timeout=5
        )
        if response.status_code == 200:
            logger.info("✅ Graph Analyzer - интеграция работает")
        else:
            logger.warning("⚠️ Graph Analyzer - проблемы с интеграцией")
        
        # Тест интеграции с Rubin AI
        response = requests.post(
            "http://localhost:8080/api/chat",
            json={'message': 'Проанализируй производительность моторов MCSetup'},
            timeout=10
        )
        if response.status_code == 200:
            logger.info("✅ Интеграция с Rubin AI - работает")
        else:
            logger.warning("⚠️ Интеграция с Rubin AI - проблемы")
            
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования интеграции: {e}")
    
    logger.info("=" * 60)
    logger.info("🎉 Система Rubin AI с интеграцией MCSetup запущена!")
    logger.info("📊 Доступные сервисы:")
    logger.info("  - Smart Dispatcher: http://localhost:8080")
    logger.info("  - MCSetup Bridge: http://localhost:8096")
    logger.info("  - Graph Analyzer: http://localhost:8097")
    logger.info("  - General Server: http://localhost:8085")
    logger.info("  - Math Server: http://localhost:8086")
    logger.info("  - Electrical Server: http://localhost:8087")
    logger.info("  - Programming Server: http://localhost:8088")
    logger.info("  - Controllers Server: http://localhost:9000")
    logger.info("  - GAI Server: http://localhost:8104")
    logger.info("")
    logger.info("🔧 Команды для тестирования:")
    logger.info("  - Анализ моторов: curl -X POST http://localhost:8096/api/mcsetup/analyze/motors")
    logger.info("  - Анализ графиков: curl -X POST http://localhost:8097/api/graph/analyze/motors")
    logger.info("  - Чат с Rubin: curl -X POST http://localhost:8080/api/chat -H 'Content-Type: application/json' -d '{\"message\": \"Проанализируй моторы MCSetup\"}'")
    logger.info("")
    logger.info("⏹️ Для остановки нажмите Ctrl+C")
    
    try:
        # Ожидаем завершения
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Остановка системы...")
        for process, name in processes:
            try:
                process.terminate()
                logger.info(f"🛑 {name} остановлен")
            except:
                pass
        logger.info("✅ Система остановлена")

if __name__ == "__main__":
    main()



