#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Перезапуск исправленной системы Rubin AI на правильном порту
"""

import subprocess
import time
import sys
import os
import requests
import psutil

def kill_processes_on_port(port):
    """Убивает все процессы на указанном порту"""
    try:
        print(f"🔍 Ищу процессы на порту {port}...")
        
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                if connections:
                    for conn in connections:
                        if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == port:
                            print(f"🔪 Убиваю процесс {proc.info['name']} (PID: {proc.info['pid']}) на порту {port}")
                            proc.kill()
                            time.sleep(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
                
    except Exception as e:
        print(f"⚠️ Ошибка при поиске процессов: {e}")

def check_port(port):
    """Проверяет доступность порта"""
    try:
        response = requests.get(f"http://localhost:{port}/api/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def main():
    """Основная функция перезапуска"""
    print("🔄 ПЕРЕЗАПУСК ИСПРАВЛЕННОЙ СИСТЕМЫ RUBIN AI")
    print("=" * 50)
    
    # Останавливаем все процессы на портах
    ports_to_clear = [5000, 8080, 8084, 8087, 8088, 8089, 8090]
    
    for port in ports_to_clear:
        kill_processes_on_port(port)
    
    print("\n⏳ Ждем 3 секунды...")
    time.sleep(3)
    
    # Проверяем компоненты
    print("\n🔍 Проверка компонентов...")
    
    components = [
        ("Улучшенный категоризатор", "enhanced_request_categorizer"),
        ("Обработчик программирования", "programming_knowledge_handler"),
        ("Обработчик электротехники", "electrical_knowledge_handler"),
        ("Интеллектуальный диспетчер", "intelligent_dispatcher"),
        ("Нейронная сеть", "neural_rubin")
    ]
    
    all_ok = True
    for name, module in components:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_ok = False
    
    if not all_ok:
        print("\n❌ Не все компоненты доступны!")
        return False
    
    print("\n🚀 Запуск исправленного сервера на порту 8084...")
    print("📍 RubinDeveloper: file:///C:/Users/elekt/OneDrive/Desktop/Rubin_AI_v2/matrix/RubinDeveloper.html")
    print("\n🧪 ТЕСТОВЫЕ ЗАПРОСЫ:")
    print("• Сравни C++ и Python для задач промышленной автоматизации")
    print("• Как защитить электрические цепи от короткого замыкания?")
    print("• Напиши простой PLC-код для управления конвейером")
    print("• что такое контроллер и как он работает?")
    print("\n" + "=" * 50)
    print("🔄 ОЧИСТИТЕ КЭШ БРАУЗЕРА (Ctrl+Shift+R) перед тестированием!")
    print("=" * 50)
    
    # Запускаем исправленный сервер
    try:
        from rubin_server import app
        app.run(host='0.0.0.0', port=8084, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Сервер остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()