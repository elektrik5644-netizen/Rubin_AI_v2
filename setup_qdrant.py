#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Установка и настройка Qdrant для Rubin AI v2.0
"""

import subprocess
import sys
import os
import requests
import json
from pathlib import Path

def install_qdrant_client():
    """Установка Python клиента для Qdrant"""
    try:
        print("📦 Установка Qdrant Python клиента...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-client"])
        print("✅ Qdrant клиент установлен успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки Qdrant клиента: {e}")
        return False

def check_docker():
    """Проверка наличия Docker"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker найден: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker не найден")
            return False
    except FileNotFoundError:
        print("❌ Docker не установлен")
        return False

def start_qdrant_docker():
    """Запуск Qdrant в Docker контейнере"""
    try:
        print("🐳 Запуск Qdrant в Docker...")
        
        # Проверяем, не запущен ли уже Qdrant
        result = subprocess.run(["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"], 
                              capture_output=True, text=True)
        
        if "qdrant" in result.stdout:
            print("✅ Qdrant уже запущен")
            return True
        
        # Запускаем Qdrant
        cmd = [
            "docker", "run", "-p", "6333:6333", "-p", "6334:6334",
            "-v", f"{os.getcwd()}/qdrant_storage:/qdrant/storage:z",
            "--name", "qdrant", "-d",
            "qdrant/qdrant"
        ]
        
        subprocess.check_call(cmd)
        print("✅ Qdrant запущен в Docker контейнере")
        print("🌐 Qdrant доступен по адресу: http://localhost:6333")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка запуска Qdrant: {e}")
        return False

def test_qdrant_connection():
    """Тестирование подключения к Qdrant"""
    try:
        print("🔍 Тестирование подключения к Qdrant...")
        
        # Ждем запуска сервиса
        import time
        time.sleep(5)
        
        response = requests.get("http://localhost:6333/collections", timeout=10)
        if response.status_code == 200:
            print("✅ Подключение к Qdrant успешно")
            return True
        else:
            print(f"❌ Ошибка подключения: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка подключения к Qdrant: {e}")
        return False

def create_qdrant_config():
    """Создание конфигурационного файла для Qdrant"""
    config = {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "timeout": 30,
            "collections": {
                "rubin_knowledge": {
                    "vector_size": 384,
                    "distance": "Cosine",
                    "description": "База знаний Rubin AI"
                },
                "rubin_documents": {
                    "vector_size": 768,
                    "distance": "Cosine", 
                    "description": "Документы и материалы"
                },
                "rubin_conversations": {
                    "vector_size": 384,
                    "distance": "Cosine",
                    "description": "История разговоров"
                }
            }
        }
    }
    
    config_path = Path("config/qdrant_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Конфигурация Qdrant создана: {config_path}")
    return config_path

def main():
    """Главная функция установки"""
    print("🚀 УСТАНОВКА QDRANT ДЛЯ RUBIN AI v2.0")
    print("=" * 50)
    
    # Создаем директорию для хранения данных Qdrant
    os.makedirs("qdrant_storage", exist_ok=True)
    
    # Устанавливаем Python клиент
    if not install_qdrant_client():
        return False
    
    # Проверяем Docker
    if not check_docker():
        print("💡 Для запуска Qdrant требуется Docker")
        print("   Скачайте Docker с https://www.docker.com/get-started")
        return False
    
    # Запускаем Qdrant
    if not start_qdrant_docker():
        return False
    
    # Тестируем подключение
    if not test_qdrant_connection():
        return False
    
    # Создаем конфигурацию
    create_qdrant_config()
    
    print("\n" + "=" * 50)
    print("🎉 QDRANT УСПЕШНО УСТАНОВЛЕН И НАСТРОЕН!")
    print("=" * 50)
    print("🌐 Веб-интерфейс: http://localhost:6333/dashboard")
    print("📡 API: http://localhost:6333")
    print("🔧 gRPC: localhost:6334")
    print("📁 Данные: ./qdrant_storage/")
    print("\n💡 Следующие шаги:")
    print("   1. Запустить интеграцию с Rubin AI")
    print("   2. Настроить векторные коллекции")
    print("   3. Индексировать существующие данные")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Установка Qdrant завершилась с ошибками")
        sys.exit(1)
    else:
        print("\n✅ Установка Qdrant завершена успешно")







