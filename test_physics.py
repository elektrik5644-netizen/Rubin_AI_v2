#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест маршрутизации физических вопросов
"""

import requests
import json

def test_physics_routing():
    """Тестирует маршрутизацию физических вопросов"""
    
    print("🧪 Тестирование маршрутизации физических вопросов")
    print("=" * 60)
    
    # Тестовые вопросы
    test_questions = [
        "что такое фотон?",
        "расскажи про электрон",
        "что такое атом?",
        "объясни квантовую механику"
    ]
    
    for question in test_questions:
        print(f"\n📝 Вопрос: {question}")
        
        try:
            response = requests.post(
                'http://localhost:8080/api/chat',
                json={'message': question},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Ответ получен")
                print(f"📊 Категория: {result.get('category', 'неизвестно')}")
                print(f"💬 Ответ: {result.get('response', 'нет ответа')[:100]}...")
            else:
                print(f"❌ Ошибка HTTP {response.status_code}")
                print(f"📄 Ответ: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка запроса: {e}")
        
        print("-" * 40)

if __name__ == '__main__':
    test_physics_routing()



