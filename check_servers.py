#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка статуса всех серверов Rubin AI v2.0
"""

import requests
import socket
import time

def check_port(host, port, timeout=3):
    """Проверка доступности порта"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            return result == 0
    except:
        return False

def check_http_endpoint(url, timeout=5):
    """Проверка HTTP endpoint"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            # Для статических файлов не пытаемся парсить JSON
            if url.endswith('.html'):
                return True, None
            else:
                try:
                    return True, response.json()
                except:
                    return True, None
        return False, None
    except:
        return False, None

def main():
    """Главная функция"""
    print("🔍 ПРОВЕРКА СТАТУСА СЕРВЕРОВ RUBIN AI v2.0")
    print("=" * 50)
    
    # Список серверов для проверки
    servers = [
        ("AI Чат (Основной)", "localhost", 8084, "/health"),
        ("Электротехника", "localhost", 8087, "/api/electrical/status"),
        ("Радиомеханика", "localhost", 8089, "/api/radiomechanics/status"),
        ("Контроллеры", "localhost", 8090, "/api/controllers/status"),
        ("Документы", "localhost", 8088, "/health"),
        ("Статический Веб", "localhost", 8085, "/RubinIDE.html")
    ]
    
    print("📊 СТАТУС СЕРВЕРОВ:")
    print("-" * 50)
    
    online_count = 0
    total_count = len(servers)
    
    for name, host, port, endpoint in servers:
        # Проверяем порт
        port_available = check_port(host, port)
        
        if port_available:
            # Проверяем HTTP endpoint
            url = f"http://{host}:{port}{endpoint}"
            http_ok, data = check_http_endpoint(url)
            
            if http_ok:
                print(f"✅ {name}: ОНЛАЙН (порт {port})")
                if data:
                    print(f"   📄 Ответ: {data}")
                online_count += 1
            else:
                print(f"⚠️ {name}: ПОРТ ОТКРЫТ, НО HTTP НЕ ОТВЕЧАЕТ (порт {port})")
        else:
            print(f"❌ {name}: ОФФЛАЙН (порт {port})")
    
    print("-" * 50)
    print(f"📊 ИТОГО: {online_count}/{total_count} серверов онлайн")
    
    if online_count > 0:
        print(f"\n🌐 ДОСТУПНЫЕ ИНТЕРФЕЙСЫ:")
        
        if check_port("localhost", 8084):
            print(f"   🤖 AI Чат: http://localhost:8084/RubinIDE.html")
            print(f"   ⚙️ Developer: http://localhost:8084/RubinDeveloper.html")
            print(f"   📊 Статус: http://localhost:8084/status_check.html")
        
        if check_port("localhost", 8085):
            print(f"   🌐 Статический: http://localhost:8085/RubinIDE.html")
        
        if check_port("localhost", 8088):
            print(f"   📚 Документы: http://localhost:8088/DocumentsManager.html")
        
        print(f"\n🔧 СПЕЦИАЛИЗИРОВАННЫЕ API:")
        
        if check_port("localhost", 8087):
            print(f"   ⚡ Электротехника: http://localhost:8087/api/electrical/status")
        
        if check_port("localhost", 8089):
            print(f"   📡 Радиомеханика: http://localhost:8089/api/radiomechanics/status")
        
        if check_port("localhost", 8090):
            print(f"   🎛️ Контроллеры: http://localhost:8090/api/controllers/status")
    
    else:
        print(f"\n❌ НИ ОДИН СЕРВЕР НЕ ЗАПУЩЕН!")
        print(f"🚀 Запустите серверы:")
        print(f"   python start_stable_server.py")
        print(f"   python start_rubin_complete.py")

if __name__ == "__main__":
    main()

