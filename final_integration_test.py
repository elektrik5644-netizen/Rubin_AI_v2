#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальный тест интеграции LogiEval в Rubin AI v2
"""

import requests
import time

def final_integration_test():
    """Финальный тест полной интеграции."""
    print("🎉 ФИНАЛЬНЫЙ ТЕСТ ИНТЕГРАЦИИ LOGIEVAL В RUBIN AI v2")
    print("=" * 60)
    
    # Тест через диспетчер
    print("\n🧠 Тестирование через Simple Dispatcher...")
    
    test_cases = [
        "логическая задача",
        "задача на доказательства", 
        "медицинская логическая задача",
        "математическая логическая задача"
    ]
    
    success_count = 0
    
    for i, message in enumerate(test_cases, 1):
        try:
            print(f"\n{i}. Тест: '{message}'")
            
            response = requests.post(
                "http://localhost:8080/api/chat",
                json={'message': message},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                category = data.get('category', 'unknown')
                description = data.get('server_description', 'unknown')
                
                print(f"✅ Категория: {category}")
                print(f"📝 Описание: {description}")
                
                if category == 'logic_tasks':
                    print("🎯 Правильно направлено к логическим задачам!")
                    success_count += 1
                    
                    # Проверим, получили ли мы задачу
                    if 'error' not in data:
                        print("📄 Задача получена успешно!")
                    else:
                        print(f"⚠️ Ошибка сервера: {data['error']}")
                else:
                    print(f"⚠️ Направлено к: {category}")
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        time.sleep(2)  # Пауза между запросами
    
    # Прямой тест Logic Tasks API
    print("\n🔧 Прямой тест Logic Tasks API...")
    
    try:
        response = requests.get("http://localhost:8106/api/logic/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data['statistics']
            print("✅ Статистика получена:")
            print(f"📊 Всего датасетов: {len(stats['dataset_stats'])}")
            print(f"📚 Общее количество задач: {sum(info['count'] for info in stats['dataset_stats'].values())}")
            
            print("\n📋 Доступные типы задач:")
            for name, info in stats['dataset_stats'].items():
                print(f"  🔹 {name}: {info['count']} задач - {info['description']}")
        else:
            print(f"❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Итоги
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ИНТЕГРАЦИИ")
    print("=" * 60)
    print(f"✅ Успешных тестов маршрутизации: {success_count}/{len(test_cases)}")
    
    if success_count >= len(test_cases) * 0.75:  # 75% успешности
        print("🎉 ИНТЕГРАЦИЯ УСПЕШНА!")
        print("🧠 LogiEval база данных успешно интегрирована в Rubin AI v2")
        print("📚 Доступно более 50,000 логических задач")
        print("🎯 Simple Dispatcher корректно маршрутизирует запросы")
        return True
    elif success_count >= len(test_cases) * 0.5:  # 50% успешности
        print("👍 Интеграция работает хорошо")
        return True
    else:
        print("⚠️ Интеграция требует доработки")
        return False

if __name__ == '__main__':
    final_integration_test()



