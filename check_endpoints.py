#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка endpoints на серверах Rubin AI
"""

import requests
import json

def check_endpoints():
    """Проверка endpoints на серверах"""
    
    print("🔍 ПРОВЕРКА ENDPOINTS НА СЕРВЕРАХ RUBIN AI")
    print("=" * 60)
    
    servers = [
        {"name": "Electrical Server", "port": 8088, "endpoints": ["/api/electrical/status", "/api/chat", "/health"]},
        {"name": "Programming Server", "port": 8089, "endpoints": ["/api/health", "/api/chat", "/health"]},
        {"name": "Radiomechanics Server", "port": 8090, "endpoints": ["/api/health", "/api/chat", "/api/radiomechanics/status"]},
        {"name": "Controllers Server", "port": 9000, "endpoints": ["/api/health", "/api/chat", "/api/controllers/topic/general"]}
    ]
    
    for server in servers:
        print(f"\n🔍 {server['name']} (порт {server['port']}):")
        
        for endpoint in server['endpoints']:
            try:
                url = f"http://localhost:{server['port']}{endpoint}"
                response = requests.get(url, timeout=2)
                
                if response.status_code == 200:
                    print(f"  ✅ {endpoint} - HTTP 200")
                else:
                    print(f"  ❌ {endpoint} - HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"  ❌ {endpoint} - НЕДОСТУПЕН")
            except Exception as e:
                print(f"  💥 {endpoint} - ОШИБКА: {e}")

if __name__ == "__main__":
    check_endpoints()





