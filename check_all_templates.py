#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка всех работающих серверов на шаблоны
"""

import requests
import json
import time

def check_all_servers():
    """Проверка всех работающих серверов"""
    
    print("🔍 ПРОВЕРКА ВСЕХ СЕРВЕРОВ НА ШАБЛОНЫ")
    print("=" * 50)
    
    # Тестовые вопросы
    test_questions = [
        "привет",
        "помощь", 
        "что ты умеешь",
        "python",
        "транзистор"
    ]
    
    # Все серверы
    servers = [
        {"name": "General API", "url": "http://localhost:8085/api/chat"},
        {"name": "Электротехника", "url": "http://localhost:8087/api/electrical/explain"},
        {"name": "Радиомеханика", "url": "http://localhost:8089/api/radiomechanics/explain"},
        {"name": "Контроллеры", "url": "http://localhost:9000/api/controllers/topic/general"},
        {"name": "Математика", "url": "http://localhost:8086/api/chat"},
        {"name": "Программирование", "url": "http://localhost:8088/api/programming/explain"},
        {"name": "Нейросеть", "url": "http://localhost:8090/api/neuro/chat"}
    ]
    
    results = {}
    
    for server in servers:
        print(f"\n🔍 Тестируем {server['name']}:")
        print("-" * 30)
        
        server_results = []
        
        for question in test_questions:
            try:
                # Разные payload для разных серверов
                if "electrical" in server['url']:
                    payload = {"concept": question}
                elif "radiomechanics" in server['url']:
                    payload = {"concept": question}
                elif "controllers" in server['url']:
                    payload = {"data": {"message": question}}
                elif "neuro" in server['url']:
                    payload = {"message": question}
                else:
                    payload = {"message": question}
                
                response = requests.post(
                    server['url'],
                    json=payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Извлекаем ответ в зависимости от структуры
                    if 'response' in data:
                        answer = data['response']
                    elif 'message' in data:
                        answer = data['message']
                    elif 'explanation' in data:
                        answer = data['explanation']
                    else:
                        answer = str(data)
                    
                    # Проверяем на шаблонные фразы
                    template_phrases = [
                        "Я специализируюсь на",
                        "Чем могу помочь",
                        "На основе найденной информации",
                        "Документ содержит техническую информацию",
                        "Я Rubin AI v2 - ваш помощник",
                        "Понял ваш запрос",
                        "Я могу помочь с вопросами",
                        "Моя специализация"
                    ]
                    
                    has_template = any(phrase in answer for phrase in template_phrases)
                    
                    result = {
                        "question": question,
                        "answer": answer[:100] + "..." if len(answer) > 100 else answer,
                        "has_template": has_template,
                        "status": "✅ OK" if not has_template else "❌ ШАБЛОН"
                    }
                    
                    server_results.append(result)
                    
                    print(f"❓ {question}")
                    print(f"📝 {answer[:80]}{'...' if len(answer) > 80 else ''}")
                    print(f"🔍 {'❌ ШАБЛОН' if has_template else '✅ ЕСТЕСТВЕННО'}")
                    print()
                    
                else:
                    print(f"❌ Ошибка {response.status_code} для вопроса: {question}")
                    
            except Exception as e:
                print(f"❌ Ошибка подключения к {server['name']}: {e}")
                break
            
            time.sleep(0.3)  # Небольшая пауза между запросами
        
        results[server['name']] = server_results
    
    # Итоговый отчет
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЙ ОТЧЕТ ПО ВСЕМ СЕРВЕРАМ")
    print("=" * 50)
    
    total_questions = 0
    total_templates = 0
    
    for server_name, server_results in results.items():
        if server_results:
            questions_count = len(server_results)
            template_count = sum(1 for r in server_results if r['has_template'])
            natural_count = questions_count - template_count
            
            total_questions += questions_count
            total_templates += template_count
            
            print(f"\n🔍 {server_name}:")
            print(f"   Всего вопросов: {questions_count}")
            print(f"   ✅ Естественные ответы: {natural_count}")
            print(f"   ❌ Шаблонные ответы: {template_count}")
            print(f"   📈 Процент естественности: {(natural_count/questions_count)*100:.1f}%")
    
    print(f"\n🎯 ОБЩИЙ РЕЗУЛЬТАТ:")
    print(f"   Всего вопросов: {total_questions}")
    print(f"   ✅ Естественные ответы: {total_questions - total_templates}")
    print(f"   ❌ Шаблонные ответы: {total_templates}")
    print(f"   📈 Общий процент естественности: {((total_questions - total_templates)/total_questions)*100:.1f}%")
    
    if total_templates == 0:
        print(f"\n🎉 ВСЕ ШАБЛОНЫ УДАЛЕНЫ! Система работает естественно!")
    else:
        print(f"\n⚠️ Осталось {total_templates} шаблонных ответов из {total_questions}")

if __name__ == "__main__":
    check_all_servers()










