#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальный тест всех модулей Rubin AI v2
"""

import requests
import json
import time

def test_all_modules():
    """Тестирует все модули с реальными вопросами"""
    print("🎯 ФИНАЛЬНЫЙ ТЕСТ ВСЕХ МОДУЛЕЙ RUBIN AI v2")
    print("=" * 60)
    
    tests = [
        {
            "question": "привет",
            "expected_category": "general",
            "description": "Общий привет"
        },
        {
            "question": "объясни закон Кирхгофа",
            "expected_category": "electrical", 
            "description": "Электротехника"
        },
        {
            "question": "как работает антенна",
            "expected_category": "radiomechanics",
            "description": "Радиомеханика"
        },
        {
            "question": "что такое ПЛК контроллер",
            "expected_category": "controllers",
            "description": "Контроллеры"
        }
    ]
    
    success_count = 0
    
    for test in tests:
        print(f"\n🔍 Тест: {test['description']}")
        print(f"❓ Вопрос: {test['question']}")
        
        try:
            response = requests.post(
                "http://localhost:8080/api/chat",
                json={"message": test['question']},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                category = result.get('category', 'unknown')
                response_text = result.get('response', 'Нет ответа')
                
                print(f"✅ Категория: {category}")
                print(f"📝 Ответ: {response_text[:100] if isinstance(response_text, str) else str(response_text)[:100]}...")
                
                if category == test['expected_category']:
                    print("🎯 Категория определена правильно!")
                    success_count += 1
                else:
                    print(f"⚠️ Ожидалась категория: {test['expected_category']}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТ: {success_count}/{len(tests)} тестов прошли успешно")
    print("=" * 60)
    
    if success_count == len(tests):
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("\n🚀 СИСТЕМА RUBIN AI v2 ПОЛНОСТЬЮ ГОТОВА К РАБОТЕ!")
        print("\n🌐 Веб-интерфейс: http://localhost:8080")
        print("📡 API эндпоинт: http://localhost:8080/api/chat")
    else:
        print("⚠️ Некоторые тесты не прошли. Проверьте логи.")

if __name__ == "__main__":
    test_all_modules()