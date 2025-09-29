#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест keyword-based fallback для физических запросов
"""

from neural_smart_dispatcher import categorize_message_keywords

def test_keyword_fallback():
    """Тестирует keyword-based fallback для физических запросов"""
    
    # Тестируем физические запросы
    test_messages = [
        'что такое фотон?',
        'объясни электрон',
        'расскажи про атом',
        'что такое квантовая механика?',
        'физика частиц',
        'что такое атом?',
        'объясни квант',
        'расскажи про физику'
    ]

    print('=== ТЕСТ KEYWORD-BASED FALLBACK ===')
    for message in test_messages:
        category = categorize_message_keywords(message)
        print(f'Сообщение: "{message}" → Категория: {category}')

if __name__ == '__main__':
    test_keyword_fallback()



