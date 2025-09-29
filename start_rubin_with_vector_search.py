#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск Rubin AI v2.0 с поддержкой векторного поиска
"""

import os
import sys
import time
import subprocess
import signal
import psutil
from pathlib import Path

def kill_python_processes():
    """Завершение всех процессов Python"""
    print("🔧 Очистка процессов...")
    
    try:
        # Поиск процессов Python
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    # Проверяем, что это наш процесс
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if any(keyword in cmdline for keyword in ['rubin', 'api', 'server']):
                        print(f"Завершение процесса Python: {proc.info['name']} (PID: {proc.info['pid']})")
                        proc.terminate()
                        proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
                
        time.sleep(2)
        print("✅ Процессы очищены")
        
    except Exception as e:
        print(f"⚠️ Ошибка очистки процессов: {e}")

def check_dependencies():
    """Проверка зависимостей для векторного поиска"""
    print("🔍 Проверка зависимостей...")
    
    required_packages = [
        'sentence_transformers',
        'faiss',
        'numpy',
        'flask',
        'flask_cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("💡 Установите их командой: python install_vector_search_dependencies.py")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def start_server(script_path, name, port=None):
    """Запуск сервера"""
    try:
        print(f"🚀 Запуск {name}...")
        
        # Запуск в фоновом режиме
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Небольшая задержка для запуска
        time.sleep(2)
        
        # Проверка, что процесс запустился
        if process.poll() is None:
            print(f"✅ {name} запущен (PID: {process.pid})")
            if port:
                print(f"   Порт: {port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Ошибка запуска {name}")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка запуска {name}: {e}")
        return None

def check_server_health(port, name):
    """Проверка состояния сервера"""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {name} - ОНЛАЙН")
            return True
        else:
            print(f"⚠️ {name} - ПРОБЛЕМЫ (код: {response.status_code})")
            return False
    except:
        print(f"❌ {name} - ОФФЛАЙН")
        return False

def main():
    """Основная функция"""
    print("🎯 ЗАПУСК RUBIN AI v2.0 С ВЕКТОРНЫМ ПОИСКОМ")
    print("=" * 60)
    
    # Очистка процессов
    kill_python_processes()
    
    # Проверка зависимостей
    if not check_dependencies():
        print("\n❌ Не все зависимости установлены")
        print("💡 Запустите: python install_vector_search_dependencies.py")
        return False
    
    # Список серверов для запуска
    servers = [
        {
            'script': 'api/rubin_ai_v2_simple.py',
            'name': 'AI Чат (8084)',
            'port': 8084
        },
        {
            'script': 'api/electrical_api.py',
            'name': 'Электротехника (8087)',
            'port': 8087
        },
        {
            'script': 'api/radiomechanics_api.py',
            'name': 'Радиомеханика (8089)',
            'port': 8089
        },
        {
            'script': 'api/controllers_api.py',
            'name': 'Контроллеры (8090)',
            'port': 8090
        },
        {
            'script': 'api/documents_api.py',
            'name': 'Документы (8088)',
            'port': 8088
        },
        {
            'script': 'api/vector_search_api.py',
            'name': 'Векторный поиск (8091)',
            'port': 8091
        },
        {
            'script': 'static_web_server.py',
            'name': 'Веб-интерфейс (8085)',
            'port': 8085
        }
    ]
    
    # Запуск серверов
    processes = []
    print("\n🚀 Запуск серверов...")
    
    for server in servers:
        process = start_server(server['script'], server['name'], server['port'])
        if process:
            processes.append((process, server['name']))
        time.sleep(1)  # Небольшая задержка между запусками
    
    # Ожидание запуска
    print("\n⏳ Ожидание запуска серверов...")
    time.sleep(5)
    
    # Проверка состояния
    print("\n📊 Проверка состояния серверов:")
    online_count = 0
    
    for server in servers:
        if check_server_health(server['port'], server['name']):
            online_count += 1
    
    # Результат
    print(f"\n📈 Статус: {online_count}/{len(servers)} серверов онлайн")
    
    if online_count == len(servers):
        print("🎉 Все серверы запущены успешно!")
        print("\n🌐 Доступные интерфейсы:")
        print("   - Веб-интерфейс: http://localhost:8085/RubinIDE.html")
        print("   - AI Чат API: http://localhost:8084/api/chat")
        print("   - Векторный поиск: http://localhost:8091/api/hybrid/search")
        print("   - Документы API: http://localhost:8088/health")
        
        print("\n💡 Для индексации документов запустите:")
        print("   python index_documents_for_vector_search.py")
        
        # Ожидание завершения
        try:
            print("\n⏹️ Нажмите Ctrl+C для остановки всех серверов...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Остановка серверов...")
            
            # Завершение процессов
            for process, name in processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"✅ {name} остановлен")
                except:
                    try:
                        process.kill()
                        print(f"🔨 {name} принудительно завершен")
                    except:
                        print(f"❌ Не удалось остановить {name}")
            
            print("👋 Все серверы остановлены")
            return True
    else:
        print("⚠️ Некоторые серверы не запустились")
        print("💡 Проверьте логи для получения подробной информации")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Завершение работы...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")






















