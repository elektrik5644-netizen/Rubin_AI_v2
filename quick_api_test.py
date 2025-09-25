#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Quick API Test - Быстрое тестирование API серверов Rubin AI v2
"""

import requests
import json
import time
from datetime import datetime

def quick_test():
    """Быстрое тестирование всех серверов"""
    print("🚀 Rubin AI v2 - Быстрое тестирование API")
    print("=" * 50)
    
    servers = {
        'Smart Dispatcher': 8080,
        'General API': 8085,
        'Mathematics': 8086,
        'Electrical': 8087,
        'Programming': 8088,
        'Neuro': 8090,
        'Controllers': 9000,
        'PLC Analysis': 8099,
        'Advanced Math': 8100,
        'Data Processing': 8101,
        'Search Engine': 8102,
        'System Utils': 8103,
        'GAI Server': 8104,
        'Unified Manager': 8084,
        'Ethical Core': 8105
    }
    
    results = {}
    online_count = 0
    
    for name, port in servers.items():
        try:
            # Определяем правильный эндпоинт для каждого сервера
            if name == 'Neuro':
                endpoints = ['/api/health']  # Правильный эндпоинт для нейронной сети
            elif name == 'Electrical':
                endpoints = ['/api/electrical/status']
            elif name == 'Controllers':
                endpoints = ['/api/controllers/topic/general']
            elif name == 'PLC Analysis':
                endpoints = ['/api/plc/health']
            elif name == 'Advanced Math':
                endpoints = ['/api/advanced_math/health']
            elif name == 'Data Processing':
                endpoints = ['/api/data_processing/health']
            elif name == 'Search Engine':
                endpoints = ['/api/search/health']
            elif name == 'System Utils':
                endpoints = ['/api/system/health']
            elif name == 'GAI Server':
                endpoints = ['/api/gai/health']
            elif name == 'Unified Manager':
                endpoints = ['/api/system/health']
            elif name == 'Ethical Core':
                endpoints = ['/api/ethical/health']
            else:
                endpoints = ['/api/health', '/health']  # Общие эндпоинты для остальных
            
            for endpoint in endpoints:
                try:
                    url = f"http://localhost:{port}{endpoint}"
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        results[name] = "✅ ОНЛАЙН"
                        online_count += 1
                        break
                except:
                    continue
            else:
                results[name] = "❌ ОФФЛАЙН"
                
        except Exception as e:
            results[name] = f"❌ ОШИБКА: {str(e)[:30]}"
    
    # Выводим результаты
    print("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("-" * 50)
    
    for name, status in results.items():
        print(f"{status} {name}")
    
    print("-" * 50)
    print(f"📈 Статистика: {online_count}/{len(servers)} серверов онлайн")
    
    success_rate = (online_count / len(servers)) * 100
    
    if success_rate >= 90:
        print("🎉 ОТЛИЧНО! Система работает превосходно!")
    elif success_rate >= 75:
        print("👍 ХОРОШО! Система работает хорошо.")
    elif success_rate >= 50:
        print("⚠️ УДОВЛЕТВОРИТЕЛЬНО! Есть проблемы.")
    else:
        print("🚨 КРИТИЧНО! Система требует ремонта.")
    
    return results

def test_smart_dispatcher():
    """Тестирует Smart Dispatcher"""
    print("\n🎯 Тестирование Smart Dispatcher...")
    
    try:
        url = "http://localhost:8080/api/chat"
        test_message = "Привет, как дела?"
        
        payload = {'message': test_message}
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Smart Dispatcher работает корректно!")
                print(f"📝 Ответ: {data.get('response', 'Нет ответа')[:100]}...")
                return True
            else:
                print(f"⚠️ Smart Dispatcher отвечает, но с ошибкой: {data.get('error', 'Неизвестная ошибка')}")
                return False
        else:
            print(f"❌ Smart Dispatcher HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Smart Dispatcher недоступен: {e}")
        return False

if __name__ == "__main__":
    # Быстрое тестирование
    results = quick_test()
    
    # Тестирование Smart Dispatcher
    smart_dp_ok = test_smart_dispatcher()
    
    print("\n" + "=" * 50)
    print("🏁 Тестирование завершено!")
    print("=" * 50)
