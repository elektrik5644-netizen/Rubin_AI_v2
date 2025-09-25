#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная проверка статуса серверов Rubin AI v2.0
"""

import requests
import socket
import time
from datetime import datetime

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
    """Основная функция проверки"""
    print("🔍 ПРОВЕРКА СТАТУСА СЕРВЕРОВ RUBIN AI v2.0")
    print("=" * 60)
    print(f"Время проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Список серверов для проверки
    servers = [
        ("AI Чат (Основной)", "localhost", 8084, "/health"),
        ("Электротехника", "localhost", 8087, "/api/electrical/status"),
        ("Радиомеханика", "localhost", 8089, "/api/radiomechanics/status"),
        ("Контроллеры", "localhost", 8090, "/api/controllers/status"),
        ("Документы", "localhost", 8088, "/health"),
        ("Статический Веб", "localhost", 8085, "/RubinIDE.html")
    ]
    
    online_count = 0
    total_count = len(servers)
    
    for name, host, port, endpoint in servers:
        print(f"🔍 Проверка {name}...")
        
        # Проверка порта
        port_open = check_port(host, port)
        
        if port_open:
            # Проверка HTTP endpoint
            url = f"http://{host}:{port}{endpoint}"
            http_ok, data = check_http_endpoint(url)
            
            if http_ok:
                print(f"✅ {name}: ОНЛАЙН (порт {port})")
                if data:
                    print(f"   📊 Данные: {data.get('status', 'OK')}")
                online_count += 1
            else:
                print(f"⚠️ {name}: Порт открыт, но HTTP не отвечает (порт {port})")
        else:
            print(f"❌ {name}: ОФФЛАЙН (порт {port})")
        
        time.sleep(0.5)  # Небольшая пауза между проверками
    
    print("\n" + "=" * 60)
    print(f"📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   ✅ Онлайн: {online_count}/{total_count}")
    print(f"   ❌ Офлайн: {total_count - online_count}/{total_count}")
    
    if online_count == total_count:
        print("🎉 Все серверы работают нормально!")
        return True
    elif online_count > 0:
        print("⚠️ Некоторые серверы недоступны")
        return False
    else:
        print("❌ Все серверы недоступны!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

















