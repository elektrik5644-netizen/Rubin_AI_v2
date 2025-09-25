#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРИНУДИТЕЛЬНОЕ ИСПРАВЛЕНИЕ - Запуск правильного сервера
"""

import subprocess
import sys
import time
import os
import signal

def kill_all_python():
    """Убивает все Python процессы"""
    try:
        print("🔪 Убиваю ВСЕ Python процессы...")
        if os.name == 'nt':  # Windows
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
            subprocess.run(["taskkill", "/f", "/im", "pythonw.exe"], capture_output=True)
        else:  # Linux/Mac
            subprocess.run(["pkill", "-f", "python"], capture_output=True)
        time.sleep(3)
        print("✅ Все Python процессы остановлены")
    except Exception as e:
        print(f"⚠️ Ошибка остановки процессов: {e}")

def test_components():
    """Тестирует наличие исправленных компонентов"""
    print("🔍 Проверяю исправленные компоненты...")
    
    components = [
        ("Нейронная сеть", "neural_rubin", "get_neural_rubin"),
        ("Улучшенный категоризатор", "enhanced_request_categorizer", "get_enhanced_categorizer"),
        ("Программный обработчик", "programming_knowledge_handler", "get_programming_handler"),
        ("Электротехнический обработчик", "electrical_knowledge_handler", "get_electrical_handler"),
        ("Интеллектуальный диспетчер", "intelligent_dispatcher", "get_intelligent_dispatcher")
    ]
    
    all_ok = True
    for name, module, function in components:
        try:
            mod = __import__(module)
            func = getattr(mod, function)
            instance = func()
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Основная функция принудительного исправления"""
    print("🚨 ПРИНУДИТЕЛЬНОЕ ИСПРАВЛЕНИЕ RUBIN AI")
    print("=" * 50)
    
    # Шаг 1: Убиваем все Python процессы
    kill_all_python()
    
    # Шаг 2: Проверяем компоненты
    if not test_components():
        print("❌ Не все компоненты доступны!")
        print("Возможно, нужно перезапустить из правильной директории")
        return
    
    # Шаг 3: Тестируем интеграцию
    print("\n🧪 Быстрый тест интеграции...")
    try:
        from neural_rubin import get_neural_rubin
        neural_ai = get_neural_rubin()
        
        # Тест программного запроса
        test_response = neural_ai.generate_response("Сравни C++ и Python")
        if "Programming Knowledge Handler" in test_response.get('provider', ''):
            print("✅ Интеграция работает!")
        else:
            print("⚠️ Интеграция может работать неправильно")
            print(f"Провайдер: {test_response.get('provider', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
    
    # Шаг 4: Запускаем исправленный сервер
    print(f"\n🚀 ЗАПУСК ИСПРАВЛЕННОГО СЕРВЕРА НА ПОРТУ 8084")
    print("📍 RubinDeveloper: file:///C:/Users/elekt/OneDrive/Desktop/Rubin_AI_v2/matrix/RubinDeveloper.html")
    print("\n🧪 ТЕСТОВЫЕ ЗАПРОСЫ:")
    print("• привет")
    print("• Как работает ПИД-регулятор? Объясни простыми словами.")
    print("• Проектирование и Архитектура ПЛК?")
    print("• что такое конденсатор?")
    print("• Сравни C++ и Python")
    print("\n" + "=" * 50)
    print("🔄 ОБНОВИТЕ СТРАНИЦУ RUBINDEVELOPER (F5)!")
    print("=" * 50)
    
    # Запускаем сервер
    try:
        # Импортируем исправленный сервер
        from rubin_server import app
        
        # Запускаем на порту 8084
        print("🌐 Сервер запущен на http://localhost:8084")
        app.run(host='0.0.0.0', port=8084, debug=False, threaded=True)
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Сервер остановлен")
    except Exception as e:
        print(f"\n❌ Фатальная ошибка: {e}")
        import traceback
        traceback.print_exc()