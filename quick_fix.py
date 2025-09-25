#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрое исправление - запуск правильного сервера на порту 8084
"""

import subprocess
import sys
import time

def main():
    print("🚀 БЫСТРОЕ ИСПРАВЛЕНИЕ RUBIN AI")
    print("=" * 40)
    
    print("1. 🔪 Останавливаю все Python процессы...")
    try:
        # Останавливаем все Python процессы (осторожно!)
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                      capture_output=True, shell=True)
        time.sleep(2)
    except:
        pass
    
    print("2. 🚀 Запускаю исправленный сервер...")
    print("📍 Сервер будет на: http://localhost:8084")
    print("🌐 RubinDeveloper готов к тестированию!")
    print("\n🔄 ОБЯЗАТЕЛЬНО ОЧИСТИТЕ КЭШ БРАУЗЕРА (Ctrl+Shift+R)!")
    print("=" * 40)
    
    # Запускаем исправленный сервер
    try:
        from rubin_server import app
        app.run(host='0.0.0.0', port=8084, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()