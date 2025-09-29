#!/usr/bin/env python3
"""
Отладка проблемы с поиском
"""

import sys
sys.path.append('.')
from rubin_ultimate_system import RubinUltimateSystem

def debug_search_issue():
    """Отладка проблемы с поиском"""
    
    print("🔍 ОТЛАДКА ПРОБЛЕМЫ С ПОИСКОМ")
    print("=" * 40)
    
    # Создаем экземпляр системы
    ai = RubinUltimateSystem()
    
    # Проверяем содержимое базы данных
    print("📊 Содержимое базы данных:")
    for i, doc in enumerate(ai.database_content):
        print(f"{i+1}. {doc['filename']} ({doc['category']})")
        print(f"   Содержание: {doc['content'][:100]}...")
        print()
    
    # Тестируем поиск с разными запросами
    queries = ['атом', 'химия', 'физика', 'математика']
    for query in queries:
        results = ai.search_content(query)
        print(f"🔍 Поиск '{query}': {len(results)} результатов")
        for result in results:
            print(f"   • {result['filename']} ({result['category']})")
        print()
    
    # Тестируем генерацию ответа
    print("💬 Тестирование генерации ответа:")
    response = ai.generate_response('Что такое атом?')
    print(f"Категория: {response['category']}")
    print(f"Источники найдены: {response['sources_found']}")
    print(f"Ответ: {response['response'][:200]}...")

if __name__ == "__main__":
    debug_search_issue()

















