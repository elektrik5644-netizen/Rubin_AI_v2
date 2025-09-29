#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для системы директив Rubin AI
"""

import requests
import json

SMART_DISPATCHER_URL = "http://localhost:8080/api/chat"

def test_directive_command(command: str, user_id: str = "test_user"):
    """Тестирует команду директив"""
    payload = {"message": command, "user_id": user_id}
    try:
        response = requests.post(SMART_DISPATCHER_URL, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Команда: {command}")
            print(f"Ответ: {json.dumps(data, ensure_ascii=False, indent=2)}")
            print("-" * 50)
        else:
            print(f"❌ Ошибка HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def test_regular_message(message: str):
    """Тестирует обычное сообщение"""
    payload = {"message": message}
    try:
        response = requests.post(SMART_DISPATCHER_URL, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"📝 Сообщение: {message}")
            print(f"Ответ: {data.get('response', 'Нет ответа')}")
            if 'directives_applied' in data:
                print(f"📋 Применены директивы: {data['directives_applied']}")
            print("-" * 50)
        else:
            print(f"❌ Ошибка HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def main():
    print("🧪 Тестирование системы директив Rubin AI")
    print("=" * 50)
    
    # Тест команд директив
    print("1. Тестирование команд директив:")
    test_directive_command("помощь по директивам")
    test_directive_command("прими директиву всегда добавляй примеры к ответам")
    test_directive_command("прими директиву при анализе графиков проверяй тренд")
    test_directive_command("прими директиву в электротехнике объясняй формулы подробнее")
    test_directive_command("список директив")
    test_directive_command("статистика директив")
    
    print("\n2. Тестирование обычных сообщений с применением директив:")
    test_regular_message("что такое резистор")
    test_regular_message("как работает конденсатор")
    test_regular_message("проанализируй график")
    test_regular_message("расскажи про закон Ома")
    
    print("\n3. Тестирование удаления директив:")
    test_directive_command("удали директиву dir_20250923_233000")  # Замените на реальный ID

if __name__ == "__main__":
    main()






