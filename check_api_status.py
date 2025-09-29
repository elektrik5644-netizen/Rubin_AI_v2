#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time

def check_api_status():
    """Проверяет статус всех API серверов Rubin AI"""
    
    # Определяем модули для проверки
    modules = {
        'Smart Dispatcher': 8080,
        'Gemini Bridge': 8082,
        'General API': 8085,
        'Mathematics API': 8086,
        'Electrical API': 8087,
        'Programming API': 8088,
        'Radiomechanics API': 8089,
        'Neuro API': 8090,
        'Controllers API': 9000,
        'Search Engine API': 8102,
        'System Utils API': 8103,
        'GAI API': 8104,
        'Ethical Core API': 8105,
        # Arduino Nano API удален - теперь встроенный модуль в Smart Dispatcher
    }
    
    print("🔍 ПРОВЕРКА СТАТУСА API СЕРВЕРОВ RUBIN AI")
    print("=" * 50)
    
    working_count = 0
    total_count = len(modules)
    
    for name, port in modules.items():
        try:
            response = requests.get(f'http://localhost:{port}/api/health', timeout=3)
            if response.status_code == 200:
                status = "✅ РАБОТАЕТ"
                working_count += 1
            else:
                status = f"❌ ОШИБКА ({response.status_code})"
        except requests.exceptions.ConnectionError:
            status = "❌ НЕДОСТУПЕН"
        except requests.exceptions.Timeout:
            status = "⏳ TIMEOUT"
        except Exception as e:
            status = f"❌ ОШИБКА ({str(e)[:20]}...)"
        
        print(f"{name:<20} ({port}): {status}")
    
    print("=" * 50)
    print(f"📊 СТАТИСТИКА: {working_count}/{total_count} серверов работают")
    
    if working_count < total_count:
        print("\n🚨 НЕДОСТУПНЫЕ СЕРВИСЫ НУЖНО ЗАПУСТИТЬ:")
        for name, port in modules.items():
            try:
                response = requests.get(f'http://localhost:{port}/api/health', timeout=2)
                if response.status_code != 200:
                    print(f"  - {name} (порт {port})")
            except:
                print(f"  - {name} (порт {port})")
    
    return working_count, total_count

if __name__ == "__main__":
    check_api_status()
