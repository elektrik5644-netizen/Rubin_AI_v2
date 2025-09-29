#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый тест маршрутизации логических задач
"""

import requests

def test_logic_routing():
    """Тестирует маршрутизацию логических задач через диспетчер."""
    print("🧠 Тест маршрутизации логических задач")
    print("=" * 50)
    
    dispatcher_url = "http://localhost:8080"
    
    test_messages = [
        "логическая задача",
        "дай логическую задачу", 
        "задача на доказательства",
        "логическое рассуждение",
        "решить логическую задачу"
    ]
    
    for message in test_messages:
        try:
            print(f"\n📝 Тест: '{message}'")
            response = requests.post(
                f"{dispatcher_url}/api/chat",
                json={'message': message},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                category = data.get('category', 'unknown')
                description = data.get('server_description', 'unknown')
                
                print(f"📊 Категория: {category}")
                print(f"📝 Описание: {description}")
                
                if category == 'logic_tasks':
                    print("✅ Правильно направлено к логическим задачам!")
                elif 'error' in data:
                    print(f"⚠️ Ошибка сервера: {data['error']}")
                else:
                    print(f"⚠️ Направлено к: {category}")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == '__main__':
    test_logic_routing()








