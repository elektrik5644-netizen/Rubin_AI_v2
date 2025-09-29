#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест Electrical Server
"""

import requests
import json

def test_electrical_server():
    """Тестирование Electrical Server"""
    
    # Тест health endpoint
    print("🔍 Тестирование health endpoint...")
    try:
        health_response = requests.get("http://localhost:8087/api/health", timeout=5)
        print(f"Health status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Health response: {health_response.json()}")
        else:
            print(f"Health error: {health_response.text}")
    except Exception as e:
        print(f"Health error: {e}")
    
    print("\n🔍 Тестирование chat endpoint...")
    try:
        chat_response = requests.post(
            "http://localhost:8087/api/chat",
            json={"message": "закон Кирхгофа"},
            timeout=10
        )
        print(f"Chat status: {chat_response.status_code}")
        if chat_response.status_code == 200:
            print(f"Chat response: {chat_response.json()}")
        else:
            print(f"Chat error: {chat_response.text}")
    except Exception as e:
        print(f"Chat error: {e}")

if __name__ == "__main__":
    test_electrical_server()










