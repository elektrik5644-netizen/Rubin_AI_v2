#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт запуска автоматического анализатора MCSetup
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Проверяет зависимости"""
    try:
        import watchdog
        logger.info("✅ watchdog установлен")
        return True
    except ImportError:
        logger.error("❌ watchdog не установлен")
        logger.info("Установите: pip install watchdog")
        return False

def check_mcsetup_path():
    """Проверяет путь к MCSetup"""
    mcsetup_path = r"C:\Users\elekt\OneDrive\Desktop\MCSetup_V1_9_0\MCSetup_V1_9_0"
    
    if not Path(mcsetup_path).exists():
        logger.error(f"❌ Путь MCSetup не найден: {mcsetup_path}")
        return False
    
    plot_dir = Path(mcsetup_path) / "Plot, Watch windows" / "Plot"
    if not plot_dir.exists():
        logger.error(f"❌ Директория графиков не найдена: {plot_dir}")
        return False
    
    logger.info(f"✅ MCSetup найден: {mcsetup_path}")
    return True

def check_servers():
    """Проверяет доступность серверов"""
    import requests
    
    servers = [
        ("MCSetup Bridge", "http://localhost:8096/api/mcsetup/health"),
        ("Smart Dispatcher", "http://localhost:8080/api/health")
    ]
    
    all_ok = True
    for name, url in servers:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {name} доступен")
            else:
                logger.warning(f"⚠️ {name} отвечает с кодом {response.status_code}")
                all_ok = False
        except Exception as e:
            logger.error(f"❌ {name} недоступен: {e}")
            all_ok = False
    
    return all_ok

def check_telegram_config():
    """Проверяет конфигурацию Telegram"""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN не установлен")
        return False
    
    if not chat_id:
        logger.warning("⚠️ TELEGRAM_CHAT_ID не установлен")
        return False
    
    logger.info("✅ Telegram настроен")
    return True

def start_auto_analyzer():
    """Запускает автоматический анализатор"""
    try:
        logger.info("🚀 Запуск автоматического анализатора MCSetup...")
        
        # Запускаем анализатор
        subprocess.run([sys.executable, "mcsetup_auto_analyzer.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка запуска анализатора: {e}")
    except KeyboardInterrupt:
        logger.info("⏹️ Остановка по запросу пользователя")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")

def main():
    """Главная функция"""
    print("🤖 Автоматический анализатор MCSetup")
    print("=" * 50)
    
    # Проверки
    logger.info("🔍 Проверка зависимостей...")
    if not check_dependencies():
        return
    
    logger.info("🔍 Проверка MCSetup...")
    if not check_mcsetup_path():
        return
    
    logger.info("🔍 Проверка серверов...")
    if not check_servers():
        logger.warning("⚠️ Некоторые серверы недоступны, но продолжаем...")
    
    logger.info("🔍 Проверка Telegram...")
    telegram_ok = check_telegram_config()
    if not telegram_ok:
        logger.warning("⚠️ Telegram не настроен, рекомендации не будут отправляться")
    
    print("\n" + "=" * 50)
    logger.info("✅ Все проверки пройдены")
    logger.info("📊 Автоматический анализатор готов к работе")
    logger.info("💡 Теперь при нажатии кнопки 'Отрисовать' в MCSetup будет автоматически запускаться анализ")
    print("=" * 50)
    
    # Запуск
    start_auto_analyzer()

if __name__ == '__main__':
    main()



