#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки knowledge_api_server.py
"""

import sys
import traceback

def test_imports():
    """Тестирует импорты"""
    try:
        print("🔄 Тестируем импорты...")
        
        from intelligent_knowledge_chat import get_rubin_chat
        print("✅ intelligent_knowledge_chat импортирован")
        
        from central_knowledge_base import get_knowledge_base
        print("✅ central_knowledge_base импортирован")
        
        from flask import Flask
        print("✅ Flask импортирован")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        traceback.print_exc()
        return False

def test_initialization():
    """Тестирует инициализацию компонентов"""
    try:
        print("\n🔄 Тестируем инициализацию...")
        
        from intelligent_knowledge_chat import get_rubin_chat
        from central_knowledge_base import get_knowledge_base
        
        chat_system = get_rubin_chat()
        print("✅ Chat system инициализирован")
        
        knowledge_base = get_knowledge_base()
        print("✅ Knowledge base инициализирован")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Тестирует создание Flask приложения"""
    try:
        print("\n🔄 Тестируем Flask приложение...")
        
        from flask import Flask
        app = Flask(__name__)
        print("✅ Flask приложение создано")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка Flask: {e}")
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование knowledge_api_server.py")
    print("=" * 50)
    
    # Тест 1: Импорты
    if not test_imports():
        print("\n❌ Тест импортов провален")
        return False
    
    # Тест 2: Инициализация
    if not test_initialization():
        print("\n❌ Тест инициализации провален")
        return False
    
    # Тест 3: Flask
    if not test_flask_app():
        print("\n❌ Тест Flask провален")
        return False
    
    print("\n✅ Все тесты пройдены успешно!")
    print("🚀 Сервер должен работать корректно")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



