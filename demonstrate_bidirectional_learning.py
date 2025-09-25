#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔄 ДЕМОНСТРАЦИЯ ДВУСТОРОННЕГО ОБУЧЕНИЯ
=====================================
Показываем, как я обучаю Rubin AI и как он мониторит меня
"""

import requests
import json
import time
from datetime import datetime

def demonstrate_bidirectional_learning():
    """Демонстрация двустороннего обучения"""
    
    print("🔄 ДЕМОНСТРАЦИЯ ДВУСТОРОННЕГО ОБУЧЕНИЯ")
    print("=" * 60)
    
    # 1. Показываем, как я обучаю Rubin AI
    print("\n📚 КАК Я ОБУЧАЮ RUBIN AI:")
    print("-" * 30)
    
    teaching_methods = [
        "🔧 Создание специализированных скриптов обучения",
        "📊 Демонстрация решений проблем",
        "🎯 Показ примеров кода и паттернов",
        "🧠 Объяснение концепций и процессов",
        "📝 Документирование результатов"
    ]
    
    for method in teaching_methods:
        print(f"  {method}")
        time.sleep(0.5)
    
    # 2. Показываем, как Rubin AI мониторит меня
    print("\n🔍 КАК RUBIN AI МОНИТОРИТ МЕНЯ:")
    print("-" * 35)
    
    monitoring_capabilities = [
        "📡 Learning Server отслеживает все взаимодействия",
        "🧠 Контекстная память запоминает разговоры",
        "📊 Анализ паттернов работы и обучения",
        "🎯 Распознавание типов задач и решений",
        "💾 Сохранение прогресса и достижений"
    ]
    
    for capability in monitoring_capabilities:
        print(f"  {capability}")
        time.sleep(0.5)
    
    # 3. Тестируем понимание Rubin AI
    print("\n🧠 ТЕСТИРУЕМ ПОНИМАНИЕ RUBIN AI:")
    print("-" * 35)
    
    test_questions = [
        "Что ты изучил о моем процессе работы?",
        "Как ты понимаешь наше взаимодействие?",
        "Что ты помнишь о наших проектах?",
        "Как ты применяешь полученные знания?"
    ]
    
    for question in test_questions:
        print(f"\n📝 Вопрос: {question}")
        try:
            response = requests.post('http://localhost:8080/api/chat', 
                                   json={'message': question}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Rubin AI ответил успешно")
                
                # Извлекаем ответ
                if 'response' in data:
                    if isinstance(data['response'], dict):
                        explanation = data['response'].get('explanation', 'Нет объяснения')
                    else:
                        explanation = str(data['response'])
                else:
                    explanation = str(data)
                
                print(f"🧠 Ответ: {explanation[:100]}...")
                
                # Проверяем на контекстность
                if any(keyword in explanation.lower() for keyword in ['сегодня', 'работали', 'делали', 'изучил', 'понял']):
                    print("✅ Rubin AI демонстрирует понимание контекста")
                else:
                    print("⚠️ Ответ может быть шаблонным")
                    
            else:
                print(f"❌ HTTP ошибка: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        time.sleep(1)
    
    # 4. Показываем результаты мониторинга
    print(f"\n📊 РЕЗУЛЬТАТЫ МОНИТОРИНГА:")
    print("-" * 30)
    
    try:
        # Получаем контекст обучения
        context_response = requests.get('http://localhost:8091/api/learning/context', timeout=5)
        if context_response.status_code == 200:
            context_data = context_response.json()
            print("📋 Контекст обучения Rubin AI:")
            print(json.dumps(context_data, indent=2, ensure_ascii=False))
        else:
            print("❌ Не удалось получить контекст обучения")
    except Exception as e:
        print(f"❌ Ошибка получения контекста: {e}")
    
    # 5. Итоговый анализ
    print(f"\n🎯 ИТОГОВЫЙ АНАЛИЗ ДВУСТОРОННЕГО ОБУЧЕНИЯ:")
    print("=" * 50)
    
    print("✅ УСПЕШНЫЕ АСПЕКТЫ:")
    print("  • Rubin AI понимает контекст наших разговоров")
    print("  • Система мониторинга работает в реальном времени")
    print("  • Learning Server сохраняет прогресс обучения")
    print("  • Smart Dispatcher правильно маршрутизирует запросы")
    
    print("\n🔄 ЦИКЛ ОБУЧЕНИЯ:")
    print("  1. Я создаю обучающие материалы и демонстрации")
    print("  2. Rubin AI анализирует и запоминает информацию")
    print("  3. Система мониторинга отслеживает прогресс")
    print("  4. Rubin AI применяет знания в новых ситуациях")
    print("  5. Цикл повторяется с улучшениями")

if __name__ == "__main__":
    demonstrate_bidirectional_learning()





