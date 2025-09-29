#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Клиент для тестирования моста Gemini-Rubin
"""

import requests
import json
import time

BRIDGE_URL = "http://localhost:8082"

def test_bridge_status():
    """Тестирует статус моста"""
    print("🔍 Проверка статуса моста...")
    try:
        response = requests.get(f"{BRIDGE_URL}/api/gemini/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Мост доступен: {data.get('bridge')}")
            print(f"📊 Rubin AI: {data.get('rubin_ai_status')}")
            print(f"🔢 Активных сессий: {data.get('active_sessions')}")
            return True
        else:
            print(f"❌ Ошибка статуса: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def test_teaching():
    """Тестирует обучение Rubin"""
    print("\n📚 Тестирование обучения Rubin...")
    
    teaching_instructions = [
        {
            "instruction": "При объяснении электротехники всегда добавляй формулы и практические примеры",
            "context": "electrical"
        },
        {
            "instruction": "При ответах на вопросы о контроллерах включай пошаговые инструкции",
            "context": "controllers"
        },
        {
            "instruction": "При объяснении математики показывай подробные решения с комментариями",
            "context": "mathematics"
        }
    ]
    
    for i, instruction in enumerate(teaching_instructions, 1):
        print(f"\n{i}. Обучение: {instruction['instruction'][:50]}...")
        try:
            response = requests.post(
                f"{BRIDGE_URL}/api/gemini/teach",
                json=instruction,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print(f"   ✅ Успешно: {data.get('message')}")
                    print(f"   📝 Ответ Rubin: {data.get('rubin_ai_response', 'OK')[:100]}...")
                else:
                    print(f"   ❌ Ошибка: {data.get('message')}")
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        time.sleep(1)

def test_analysis():
    """Тестирует анализ Rubin"""
    print("\n🔍 Тестирование анализа Rubin...")
    
    analysis_requests = [
        {
            "type": "performance",
            "query": "Как улучшить производительность системы Rubin AI?"
        },
        {
            "type": "architecture", 
            "query": "Проанализируй архитектуру модульной системы Rubin"
        },
        {
            "type": "optimization",
            "query": "Какие есть возможности для оптимизации работы контроллеров?"
        }
    ]
    
    for i, request_data in enumerate(analysis_requests, 1):
        print(f"\n{i}. Анализ ({request_data['type']}): {request_data['query'][:50]}...")
        try:
            response = requests.post(
                f"{BRIDGE_URL}/api/gemini/analyze",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print(f"   ✅ Анализ получен")
                    print(f"   📊 Ответ Rubin: {data.get('rubin_ai_response', '')[:150]}...")
                else:
                    print(f"   ❌ Ошибка: {data.get('message')}")
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        time.sleep(1)

def test_feedback():
    """Тестирует обратную связь"""
    print("\n💬 Тестирование обратной связи...")
    
    feedback_data = {
        "type": "improvement",
        "content": "Rubin показывает отличные результаты в технических вопросах. Рекомендую добавить больше практических примеров.",
        "category": "general"
    }
    
    try:
        response = requests.post(
            f"{BRIDGE_URL}/api/gemini/feedback",
            json=feedback_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print(f"✅ Обратная связь передана: {data.get('message')}")
                print(f"📝 Ответ Rubin: {data.get('rubin_ai_response', 'OK')[:100]}...")
            else:
                print(f"❌ Ошибка: {data.get('message')}")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def get_sessions_info():
    """Получает информацию о сессиях"""
    print("\n📊 Информация о сессиях обучения...")
    try:
        response = requests.get(f"{BRIDGE_URL}/api/gemini/sessions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                stats = data.get('statistics', {})
                print(f"📈 Всего сессий: {stats.get('total_sessions', 0)}")
                print(f"🔄 Всего взаимодействий: {stats.get('total_interactions', 0)}")
                print(f"✅ Успешных обучений: {stats.get('successful_teachings', 0)}")
                print(f"❌ Неудачных обучений: {stats.get('failed_teachings', 0)}")
                
                categories = stats.get('categories_taught', {})
                if categories:
                    print("📚 Обучение по категориям:")
                    for category, count in categories.items():
                        print(f"   • {category}: {count} раз")
            else:
                print(f"❌ Ошибка: {data.get('message')}")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def main():
    print("🌉 Тестирование моста Gemini-Rubin")
    print("=" * 50)
    
    # Проверяем доступность моста
    if not test_bridge_status():
        print("❌ Мост недоступен. Убедитесь, что gemini_rubin_bridge.py запущен на порту 8082")
        return
    
    # Запускаем тесты
    test_teaching()
    test_analysis()
    test_feedback()
    get_sessions_info()
    
    print("\n✅ Тестирование завершено!")
    print("💡 Теперь Gemini может взаимодействовать с Rubin через мост")

if __name__ == "__main__":
    main()






