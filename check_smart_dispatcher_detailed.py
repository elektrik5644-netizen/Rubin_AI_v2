#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальная проверка ответов Smart Dispatcher
Показывает полные ответы и анализирует проблемы
"""

import requests
import json

def check_smart_dispatcher_responses():
    """Проверяет детальные ответы Smart Dispatcher"""
    
    print("🔍 ДЕТАЛЬНАЯ ПРОВЕРКА ОТВЕТОВ SMART DISPATCHER")
    print("=" * 80)
    
    smart_dispatcher_url = "http://localhost:8080/api/chat"
    
    test_questions = [
        "Что такое закон Ома?",
        "Реши уравнение x^2 + 5x + 6 = 0",
        "Как работает транзистор?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"❓ Вопрос {i}: {question}")
        print(f"{'='*60}")
        
        try:
            response = requests.post(
                smart_dispatcher_url,
                json={"message": question},
                timeout=10
            )
            
            print(f"📊 Статус HTTP: {response.status_code}")
            print(f"📋 Заголовки: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"📦 Полный JSON ответ:")
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                    
                    # Анализируем структуру ответа
                    print(f"\n🔍 Анализ структуры ответа:")
                    for key, value in data.items():
                        print(f"   • {key}: {type(value).__name__} = {str(value)[:100]}...")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Ошибка парсинга JSON: {e}")
                    print(f"📝 Сырой ответ: {response.text}")
            else:
                print(f"❌ Ошибка HTTP: {response.status_code}")
                print(f"📝 Ответ: {response.text}")
                
        except Exception as e:
            print(f"❌ Ошибка запроса: {e}")

if __name__ == "__main__":
    check_smart_dispatcher_responses()





