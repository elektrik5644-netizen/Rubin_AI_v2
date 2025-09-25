#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Установка зависимостей для нейронной сети Rubin AI
"""

import subprocess
import sys

def install_package(package):
    """Устанавливает пакет через pip"""
    try:
        print(f"📦 Устанавливаю {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} установлен")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки {package}: {e}")
        return False

def main():
    """Устанавливает все необходимые библиотеки"""
    print("🧠 УСТАНОВКА НЕЙРОННОЙ СЕТИ ДЛЯ RUBIN AI")
    print("=" * 50)
    
    # Основные ML библиотеки
    packages = [
        "torch",                    # PyTorch для нейронных сетей
        "transformers",             # Hugging Face трансформеры
        "sentence-transformers",    # Для эмбеддингов
        "scikit-learn",            # Классические ML алгоритмы
        "numpy",                   # Численные вычисления
        "pandas",                  # Работа с данными
        "nltk",                    # Обработка естественного языка
        "spacy",                   # Продвинутый NLP
        "datasets",                # Наборы данных
        "accelerate",              # Ускорение обучения
    ]
    
    successful = 0
    failed = 0
    
    for package in packages:
        if install_package(package):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТ:")
    print(f"✅ Успешно установлено: {successful}")
    print(f"❌ Ошибок: {failed}")
    
    if failed == 0:
        print("🎉 Все библиотеки установлены! Можно внедрять нейронную сеть!")
    else:
        print("⚠️ Некоторые библиотеки не установились. Проверьте ошибки выше.")

if __name__ == "__main__":
    main()