#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест всех endpoints Electrical Server
"""

import requests
import json

def test_all_endpoints():
    """Тестирование всех endpoints Electrical Server"""
    
    base_url = "http://localhost:8087"
    
    endpoints = [
        ("/health", "GET"),
        ("/api/electrical/status", "GET"),
        ("/api/electrical/topics", "GET"),
        ("/api/electrical/explain", "POST"),
        ("/api/electrical/calculate", "POST"),
        ("/api/chat", "POST")
    ]
    
    for endpoint, method in endpoints:
        print(f"\n🔍 Тестирование {method} {endpoint}...")
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            else:
                response = requests.post(
                    f"{base_url}{endpoint}",
                    json={"message": "закон Кирхгофа"},
                    timeout=5
                )
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response: {json.dumps(data, ensure_ascii=False, indent=2)[:200]}...")
                except:
                    print(f"Response: {response.text[:200]}...")
            else:
                print(f"Error: {response.text[:200]}...")
                
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_all_endpoints()










