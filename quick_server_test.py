#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый тест основных серверов Rubin AI
"""

import requests
import json
import time

def quick_server_test():
    """Быстрый тест основных серверов"""
    
    print("⚡ БЫСТРЫЙ ТЕСТ ОСНОВНЫХ СЕРВЕРОВ RUBIN AI")
    print("=" * 50)
    
    servers = [
        {"name": "Smart Dispatcher", "port": 8080, "endpoint": "/api/health"},
        {"name": "General API", "port": 8085, "endpoint": "/api/health"},
        {"name": "Math Server", "port": 8086, "endpoint": "/health"},
        {"name": "Electrical Server", "port": 8088, "endpoint": "/api/electrical/status"},
        {"name": "Programming Server", "port": 8089, "endpoint": "/health"},
        {"name": "Radiomechanics Server", "port": 8090, "endpoint": "/api/radiomechanics/status"},
        {"name": "Controllers Server", "port": 9000, "endpoint": "/api/controllers/topic/general"}
    ]
    
    online_count = 0
    total_count = len(servers)
    
    for server in servers:
        try:
            url = f"http://localhost:{server['port']}{server['endpoint']}"
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                print(f"✅ {server['name']} (порт {server['port']}) - ОНЛАЙН")
                online_count += 1
            else:
                print(f"❌ {server['name']} (порт {server['port']}) - ОШИБКА HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {server['name']} (порт {server['port']}) - НЕДОСТУПЕН")
        except requests.exceptions.Timeout:
            print(f"⏰ {server['name']} (порт {server['port']}) - ТАЙМАУТ")
        except Exception as e:
            print(f"💥 {server['name']} (порт {server['port']}) - ОШИБКА: {e}")
    
    print(f"\n📊 ИТОГИ:")
    print(f"• Всего серверов: {total_count}")
    print(f"• Онлайн серверов: {online_count}")
    print(f"• Доступность: {(online_count/total_count*100):.1f}%")
    
    if online_count >= total_count * 0.7:
        print("🎉 СИСТЕМА РАБОТАЕТ ОТЛИЧНО!")
    elif online_count >= total_count * 0.5:
        print("✅ СИСТЕМА РАБОТАЕТ ХОРОШО")
    else:
        print("⚠️ СИСТЕМА ТРЕБУЕТ ВНИМАНИЯ")

if __name__ == "__main__":
    quick_server_test()
