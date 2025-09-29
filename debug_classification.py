#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отладка классификации Smart Dispatcher
"""

import requests
import json

def debug_classification():
    """Отлаживает классификацию сообщений"""
    
    print("🔍 Отладка классификации Smart Dispatcher")
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
            # Отправляем запрос с дополнительными заголовками для отладки
            response = requests.post(
                'http://localhost:8080/api/chat',
                json={'message': question},
                headers={'X-Debug': 'true'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Ответ получен")
                print(f"📊 Категория: {result.get('category', 'неизвестно')}")
                
                # Проверяем, есть ли информация об отладке
                if 'debug_info' in result:
                    debug_info = result['debug_info']
                    print(f"🔍 Отладочная информация:")
                    print(f"   - Найденные ключевые слова: {debug_info.get('keywords_found', [])}")
                    print(f"   - Счетчики категорий: {debug_info.get('category_scores', {})}")
                    print(f"   - Технические счетчики: {debug_info.get('technical_scores', {})}")
                
                # Проверяем, был ли использован fallback
                if 'fallback_used' in result:
                    print(f"⚠️ Использован fallback: {result['fallback_used']}")
                
                print(f"💬 Ответ: {result.get('response', 'нет ответа')[:100]}...")
            else:
                print(f"❌ Ошибка HTTP {response.status_code}")
                print(f"📄 Ответ: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка запроса: {e}")
        
        print("-" * 40)

if __name__ == '__main__':
    debug_classification()



