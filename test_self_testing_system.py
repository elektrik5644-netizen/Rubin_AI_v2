#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест системы самотестирования Rubin AI
"""

import requests
import json
import time

def test_self_testing_system():
    """Тестирование системы самотестирования"""
    
    base_url = "http://localhost:8102"
    
    print("🧪 ТЕСТИРОВАНИЕ СИСТЕМЫ САМОТЕСТИРОВАНИЯ RUBIN AI")
    print("=" * 60)
    
    # Тест 1: Health check
    print("\n🔍 Тест 1: Health check")
    try:
        response = requests.get(f"{base_url}/api/self_testing/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data['status']}")
            print(f"   Всего серверов: {data['total_servers']}")
            print(f"   Всего возможностей: {data['total_capabilities']}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Тест 2: Server status
    print("\n🔍 Тест 2: Server status")
    try:
        response = requests.get(f"{base_url}/api/self_testing/server_status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server status: {data['online_servers']}/{data['total_servers']} онлайн")
            for server_id, status in data['server_status'].items():
                if status['status'] == 'online':
                    print(f"   ✅ {status['name']} (порт {status['port']}) - ОНЛАЙН")
                else:
                    print(f"   ❌ {status['name']} (порт {status['port']}) - {status['status']}")
        else:
            print(f"❌ Server status failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Server status error: {e}")
    
    # Тест 3: Self description
    print("\n🔍 Тест 3: Self description")
    try:
        response = requests.get(f"{base_url}/api/self_testing/self_description", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Self description generated")
            print(f"   Описание: {data['description'][:200]}...")
        else:
            print(f"❌ Self description failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Self description error: {e}")
    
    # Тест 4: Chat integration
    print("\n🔍 Тест 4: Chat integration")
    try:
        response = requests.post(
            f"{base_url}/api/self_testing/chat_integration",
            json={"message": "что умеешь?"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat integration: {data['type']}")
            if data['type'] == 'self_description':
                print(f"   Описание: {data['description'][:200]}...")
            elif data['type'] == 'server_status':
                print(f"   Статус: {data['status_message'][:200]}...")
        else:
            print(f"❌ Chat integration failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Chat integration error: {e}")
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    test_self_testing_system()










