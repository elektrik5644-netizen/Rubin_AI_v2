#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка статуса всех API модулей Rubin AI
"""

import requests
import time

def check_server_status():
    """Проверяет статус всех серверов"""
    servers = [
        ('Smart Dispatcher', 'http://localhost:8080/api/health'),
        ('General API', 'http://localhost:8085/api/health'),
        ('Математика', 'http://localhost:8086/health'),
        ('Электротехника', 'http://localhost:8087/api/electrical/status'),
        ('Программирование', 'http://localhost:8088/health'),
        ('Радиомеханика', 'http://localhost:8089/api/radiomechanics/status'),
        ('Контроллеры', 'http://localhost:9000/api/controllers/topic/general'),
        ('Нейронная сеть', 'http://localhost:8090/api/neuro/projects')
    ]
    
    print("🔌 ПРОВЕРКА ВСЕХ API МОДУЛЕЙ")
    print("=" * 50)
    
    online_count = 0
    total_count = len(servers)
    
    for name, url in servers:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"✅ {name} - ОНЛАЙН")
                online_count += 1
            else:
                print(f"⚠️ {name} - ОТВЕЧАЕТ НО ОШИБКА ({response.status_code})")
        except requests.exceptions.ConnectTimeout:
            print(f"❌ {name} - НЕДОСТУПЕН (таймаут)")
        except requests.exceptions.ConnectionError:
            print(f"❌ {name} - НЕДОСТУПЕН (нет соединения)")
        except Exception as e:
            print(f"❌ {name} - ОШИБКА: {e}")
    
    print("=" * 50)
    print(f"📊 СТАТИСТИКА: {online_count}/{total_count} серверов онлайн")
    
    if online_count == total_count:
        print("🎉 ВСЕ СЕРВЕРЫ РАБОТАЮТ!")
    elif online_count > total_count // 2:
        print("✅ Большинство серверов работает")
    else:
        print("⚠️ Многие серверы недоступны")

def test_physics_formulas():
    """Тестирует новые физические формулы"""
    print("\n⚡ ТЕСТИРОВАНИЕ ФИЗИЧЕСКИХ ФОРМУЛ")
    print("=" * 50)
    
    test_cases = [
        "Найти напряжение при токе 2 А и сопротивлении 5 Ом",
        "Найти кинетическую энергию тела массой 2 кг, движущегося со скоростью 10 м/с",
        "Найти мощность при напряжении 12 В и токе 3 А"
    ]
    
    for i, problem in enumerate(test_cases, 1):
        print(f"\n❓ Задача {i}: {problem}")
        try:
            # Тестируем через General API
            response = requests.post(
                'http://localhost:8085/api/chat',
                json={'message': problem},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Ответ: {result.get('response', 'Нет ответа')[:100]}...")
            else:
                print(f"❌ Ошибка API: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    check_server_status()
    
    # Ждем немного для запуска серверов
    print("\n⏳ Ожидание запуска серверов...")
    time.sleep(5)
    
    check_server_status()
    
    # Тестируем новые возможности
    test_physics_formulas()










