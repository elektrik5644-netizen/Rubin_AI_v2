#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Установка зависимостей для системы автоперезапуска
"""

import subprocess
import sys

def install_package(package):
    """Установка пакета"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} установлен")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка установки {package}")
        return False

def main():
    """Установка всех зависимостей"""
    print("🔧 Устанавливаю зависимости для системы автоперезапуска...")
    
    packages = [
        "requests",
        "psutil",
        "flask",
        "flask-cors"
    ]
    
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Результат: {success_count}/{len(packages)} пакетов установлено")
    
    if success_count == len(packages):
        print("✅ Все зависимости установлены успешно!")
        print("\n🚀 Теперь можно запустить:")
        print("   python quick_restart.py")
        print("   или")
        print("   python auto_restart_monitor.py")
    else:
        print("❌ Некоторые пакеты не удалось установить")

if __name__ == "__main__":
    main()











