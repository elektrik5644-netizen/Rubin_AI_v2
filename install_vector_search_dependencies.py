#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для установки зависимостей векторного поиска
"""

import subprocess
import sys
import os

def install_package(package):
    """Установка пакета через pip"""
    try:
        print(f"📦 Установка {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} установлен успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки {package}: {e}")
        return False

def check_package(package):
    """Проверка установки пакета"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Основная функция установки зависимостей"""
    print("🎯 УСТАНОВКА ЗАВИСИМОСТЕЙ ВЕКТОРНОГО ПОИСКА")
    print("=" * 50)
    
    # Список необходимых пакетов
    packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("numpy", "numpy"),
        ("flask", "flask"),
        ("flask-cors", "flask_cors")
    ]
    
    print("🔍 Проверка установленных пакетов...")
    
    missing_packages = []
    for package_name, import_name in packages:
        if check_package(import_name):
            print(f"✅ {package_name} уже установлен")
        else:
            print(f"❌ {package_name} не найден")
            missing_packages.append(package_name)
    
    if not missing_packages:
        print("\n🎉 Все зависимости уже установлены!")
        return True
    
    print(f"\n📦 Необходимо установить {len(missing_packages)} пакетов:")
    for package in missing_packages:
        print(f"   - {package}")
    
    # Подтверждение установки
    response = input("\n❓ Продолжить установку? (y/n): ").lower().strip()
    if response not in ['y', 'yes', 'да', 'д']:
        print("❌ Установка отменена")
        return False
    
    # Установка пакетов
    print("\n🚀 Начало установки...")
    success_count = 0
    
    for package in missing_packages:
        if install_package(package):
            success_count += 1
        print()  # Пустая строка для читаемости
    
    # Результат
    print("📊 РЕЗУЛЬТАТ УСТАНОВКИ:")
    print(f"   Успешно установлено: {success_count}/{len(missing_packages)}")
    
    if success_count == len(missing_packages):
        print("🎉 Все зависимости установлены успешно!")
        print("\n🚀 Теперь можно запустить векторный поиск:")
        print("   python index_documents_for_vector_search.py")
        return True
    else:
        print("⚠️ Некоторые пакеты не удалось установить")
        print("💡 Попробуйте установить их вручную:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False

if __name__ == "__main__":
    main()






















