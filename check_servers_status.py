#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time

def check_server_status():
    """Проверяет статус всех серверов Rubin AI v2"""
    
    servers = {
        'neural-dispatcher': {'port': 8080, 'endpoint': '/api/health'},
        'electrical': {'port': 8087, 'endpoint': '/health'},
        'radiomechanics': {'port': 8089, 'endpoint': '/health'},
        'controllers': {'port': 9000, 'endpoint': '/health'},
        'mathematics': {'port': 8086, 'endpoint': '/health'},
        'programming': {'port': 8088, 'endpoint': '/api/programming/explain'},  # Используем основной endpoint
        'general': {'port': 8085, 'endpoint': '/health'}
    }
    
    print("🔍 ПРОВЕРКА СТАТУСА СЕРВЕРОВ RUBIN AI V2")
    print("=" * 60)
    
    online_count = 0
    total_count = len(servers)
    
    for name, config in servers.items():
        port = config['port']
        endpoint = config['endpoint']
        url = f"http://localhost:{port}{endpoint}"
        
        try:
            if name == 'programming':
                # Для programming сервера используем POST запрос
                response = requests.post(url, json={'concept': 'test'}, timeout=10)
            else:
                response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                status = "✅ ОНЛАЙН"
                online_count += 1
            else:
                status = f"❌ ОФФЛАЙН (HTTP {response.status_code})"
        except requests.exceptions.RequestException as e:
            status = f"❌ ОФФЛАЙН ({str(e)[:30]}...)"
        
        print(f"{name:20} | Порт {port:4} | {status}")
    
    print("=" * 60)
    print(f"📊 СТАТИСТИКА: {online_count}/{total_count} серверов онлайн")
    
    if online_count == total_count:
        print("🎉 ВСЕ СЕРВЕРЫ РАБОТАЮТ!")
    elif online_count > 0:
        print("⚠️  НЕКОТОРЫЕ СЕРВЕРЫ ОФФЛАЙН")
    else:
        print("🚨 ВСЕ СЕРВЕРЫ ОФФЛАЙН!")

if __name__ == "__main__":
    check_server_status()
