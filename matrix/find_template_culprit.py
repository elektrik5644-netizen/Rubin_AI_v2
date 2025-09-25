#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Поиск виновника шаблонных ответов
"""

import requests
import json

def find_template_culprit():
    """Найти, кто генерирует шаблонные ответы"""
    print("🔍 ПОИСК ВИНОВНИКА ШАБЛОННЫХ ОТВЕТОВ")
    print("=" * 50)
    
    server_url = "http://localhost:8083"
    
    # Тестовые вопросы
    test_questions = [
        "Законы Кирхгофа применимы только к цепям постоянного тока",
        "Все датчики выдают аналоговый сигнал", 
        "G-коды используются только в фрезерных станках",
        "Случайный вопрос про автоматизацию",
        "Что-то про программирование",
        "Вопрос про датчики"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Тест {i}: {question}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{server_url}/api/chat",
                json={"message": question},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get("response", "Нет ответа")
                
                # Анализируем ответ
                print(f"📊 Анализ ответа:")
                
                # Проверяем на конкретные ответы
                if "Техническая неточность" in ai_response:
                    print("   ✅ КОНКРЕТНЫЙ ОТВЕТ (исправление неточности)")
                elif "Неточность в понимании" in ai_response:
                    print("   ✅ КОНКРЕТНЫЙ ОТВЕТ (исправление неточности)")
                elif "Неточность в понимании G-кодов" in ai_response:
                    print("   ✅ КОНКРЕТНЫЙ ОТВЕТ (исправление неточности)")
                
                # Проверяем на шаблонные ответы
                elif "Анализ вашего вопроса:" in ai_response:
                    print("   ❌ ШАБЛОННЫЙ ОТВЕТ (generate_contextual_response)")
                    print("   🔍 ВИНОВНИК: Функция generate_contextual_response()")
                    print("   📍 Строка: if detected_topics:")
                elif "Понял ваш запрос:" in ai_response:
                    print("   ❌ ШАБЛОННЫЙ ОТВЕТ (generate_contextual_response)")
                    print("   🔍 ВИНОВНИК: Функция generate_contextual_response()")
                    print("   📍 Строка: return f\"\"\"🤖 **Понял ваш запрос:**")
                else:
                    print("   ❓ НЕИЗВЕСТНЫЙ ТИП ОТВЕТА")
                
                # Показываем первые 100 символов
                preview = ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
                print(f"   📄 Превью: {preview}")
                    
            else:
                print(f"❌ Ошибка API: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка подключения: {e}")
    
    print(f"\n🎯 ВЫВОДЫ:")
    print(f"   🔍 ВИНОВНИК: Функция generate_contextual_response()")
    print(f"   📍 Место: smart_rubin_server.py, строки 509-570")
    print(f"   🚨 Проблема: Срабатывает для вопросов, не попавших в конкретные условия")
    print(f"   ✅ Решение: Добавить больше конкретных обработчиков ПЕРЕД вызовом этой функции")

if __name__ == "__main__":
    find_template_culprit()
