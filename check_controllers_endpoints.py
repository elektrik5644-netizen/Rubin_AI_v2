#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка endpoints Controllers Server
"""

import requests

def check_controllers_endpoints():
    """Проверяет доступные endpoints Controllers Server"""
    
    base_url = "http://localhost:9000"
    test_message = "тест"
    
    endpoints_to_check = [
        "/api/chat",
        "/api/query", 
        "/api/process",
        "/api/analyze",
        "/api/controller",
        "/api/plc",
        "/api/automation"
    ]
    
    print("🔍 ПРОВЕРКА ENDPOINTS CONTROLLERS SERVER")
    print("=" * 50)
    
    for endpoint in endpoints_to_check:
        try:
            response = requests.post(
                f"{base_url}{endpoint}",
                json={"message": test_message},
                timeout=3
            )
            print(f"✅ {endpoint}: {response.status_code}")
            if response.status_code == 200:
                print(f"   📝 Response: {response.text[:100]}...")
        except requests.exceptions.Timeout:
            print(f"⏰ {endpoint}: Timeout")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

if __name__ == "__main__":
    check_controllers_endpoints()





