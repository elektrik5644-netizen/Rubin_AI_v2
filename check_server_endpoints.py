#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка endpoints серверов Rubin AI
"""

import requests
import json

def check_server_endpoints():
    """Проверяет endpoints всех серверов"""
    
    servers = [
        ("Electrical Server", "http://localhost:8087"),
        ("Math Server", "http://localhost:8086"),
        ("Programming Server", "http://localhost:8088"),
        ("Controllers Server", "http://localhost:9000")
    ]
    
    test_message = "Что такое закон Ома?"
    
    for server_name, base_url in servers:
        print(f"\n{'='*60}")
        print(f"🔍 Проверка {server_name}")
        print(f"{'='*60}")
        
        # Проверяем health
        try:
            health_response = requests.get(f"{base_url}/api/health", timeout=5)
            print(f"✅ Health: {health_response.status_code}")
            if health_response.status_code == 200:
                print(f"📊 Health data: {health_response.json()}")
        except Exception as e:
            print(f"❌ Health error: {e}")
        
        # Проверяем chat endpoint
        try:
            chat_response = requests.post(
                f"{base_url}/api/chat",
                json={"message": test_message},
                timeout=10
            )
            print(f"📝 Chat status: {chat_response.status_code}")
            
            if chat_response.status_code == 200:
                try:
                    data = chat_response.json()
                    print(f"📦 Chat response: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}...")
                except:
                    print(f"📝 Raw response: {chat_response.text[:200]}...")
            else:
                print(f"❌ Chat error: {chat_response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Chat error: {e}")
        
        # Проверяем другие возможные endpoints
        possible_endpoints = ["/api/query", "/api/process", "/api/analyze"]
        for endpoint in possible_endpoints:
            try:
                response = requests.post(
                    f"{base_url}{endpoint}",
                    json={"message": test_message},
                    timeout=5
                )
                if response.status_code != 404:
                    print(f"✅ {endpoint}: {response.status_code}")
            except:
                pass

if __name__ == "__main__":
    check_server_endpoints()










