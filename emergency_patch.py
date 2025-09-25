#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЭКСТРЕННЫЙ ПАТЧ - Прямая замена функций в работающем сервере
"""

import requests
import time
import subprocess
import sys
import os

def emergency_server_replacement():
    """Экстренная замена сервера"""
    print("🚨 ЭКСТРЕННЫЙ ПАТЧ RUBIN AI")
    print("=" * 40)
    
    # Убиваем все Python процессы
    print("1. 🔪 Останавливаю все Python процессы...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
            subprocess.run(["taskkill", "/f", "/im", "pythonw.exe"], capture_output=True)
        time.sleep(3)
        print("   ✅ Процессы остановлены")
    except Exception as e:
        print(f"   ⚠️ Ошибка: {e}")
    
    # Проверяем компоненты
    print("2. 🔍 Проверяю исправленные компоненты...")
    try:
        # Тестируем нейронную сеть
        from neural_rubin import get_neural_rubin
        neural_ai = get_neural_rubin()
        
        # Тестируем интеграцию
        test_response = neural_ai.generate_response("что такое резистор?")
        
        if "Electrical Knowledge Handler" in test_response.get('provider', ''):
            print("   ✅ Нейронная сеть + электротехника работает!")
        else:
            print("   ⚠️ Интеграция может работать неправильно")
            
        # Тестируем программирование
        prog_response = neural_ai.generate_response("Сравни C++ и Python")
        if "Programming Knowledge Handler" in prog_response.get('provider', ''):
            print("   ✅ Нейронная сеть + программирование работает!")
        else:
            print("   ⚠️ Программная интеграция может работать неправильно")
            
    except Exception as e:
        print(f"   ❌ Ошибка компонентов: {e}")
        return False
    
    # Запускаем исправленный сервер
    print("3. 🚀 Запускаю ИСПРАВЛЕННЫЙ сервер на порту 8084...")
    print("   📍 http://localhost:8084")
    print("   🌐 RubinDeveloper готов!")
    print("\n🧪 После запуска тестируйте:")
    print("   • что такое резистор? → должна быть информация о резисторах")
    print("   • Сравни C++ и Python → должно быть сравнение языков")
    print("   • Как работает ПИД-регулятор? → должна быть информация о ПИД")
    print("\n" + "=" * 40)
    
    try:
        # Импортируем и запускаем исправленный сервер
        from rubin_server import app
        
        # Убеждаемся, что используем правильный порт
        print("🌐 ИСПРАВЛЕННЫЙ СЕРВЕР ЗАПУЩЕН!")
        print("🔄 ОБНОВИТЕ RUBINDEVELOPER (F5)!")
        
        app.run(host='0.0.0.0', port=8084, debug=False, threaded=True)
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    try:
        emergency_server_replacement()
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
    except Exception as e:
        print(f"\n💥 ФАТАЛЬНАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()