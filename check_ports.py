#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт проверки состояния портов Rubin AI
"""

import requests
import socket
import time
from datetime import datetime

# Конфигурация портов Rubin AI
PORTS_CONFIG = {
    'Smart Dispatcher': 8080,
    'Unified Manager': 8084,
    'General API': 8085,
    'Mathematics API': 8086,
    'Electrical API': 8087,
    'Programming API': 8088,
    'Neuro Repository': 8090,
    'Vector Search': 8091,
    'PLC Analysis': 8099,
    'Advanced Math': 8100,
    'Data Processing': 8101,
    'Search Engine': 8102,
    'System Utils': 8103,
    'Enhanced GAI': 8104,
    'Ethical Core': 8105,
    'Controllers': 9000,
    'Gemini Bridge': 8082
}

def check_port_open(host, port, timeout=3):
    """Проверяет, открыт ли порт"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_http_health(port, timeout=5):
    """Проверяет HTTP здоровье сервиса"""
    try:
        response = requests.get(f"http://localhost:{port}/api/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def check_service_status(name, port):
    """Проверяет статус сервиса"""
    port_open = check_port_open('localhost', port)
    
    if not port_open:
        return {
            'name': name,
            'port': port,
            'status': 'OFFLINE',
            'port_open': False,
            'http_health': False,
            'response_time': None
        }
    
    # Проверяем HTTP здоровье
    start_time = time.time()
    http_healthy = check_http_health(port)
    response_time = round((time.time() - start_time) * 1000, 2)
    
    return {
        'name': name,
        'port': port,
        'status': 'ONLINE' if http_healthy else 'DEGRADED',
        'port_open': True,
        'http_health': http_healthy,
        'response_time': response_time
    }

def main():
    print("🔍 Проверка состояния портов Rubin AI")
    print("=" * 60)
    print(f"⏰ Время проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    online_count = 0
    degraded_count = 0
    offline_count = 0
    
    for name, port in PORTS_CONFIG.items():
        result = check_service_status(name, port)
        results.append(result)
        
        if result['status'] == 'ONLINE':
            online_count += 1
        elif result['status'] == 'DEGRADED':
            degraded_count += 1
        else:
            offline_count += 1
    
    # Выводим результаты
    for result in results:
        status_icon = "✅" if result['status'] == 'ONLINE' else "⚠️" if result['status'] == 'DEGRADED' else "❌"
        
        print(f"{status_icon} {result['name']:<20} Порт {result['port']:<5} {result['status']:<8}", end="")
        
        if result['port_open']:
            print(f"HTTP: {'OK' if result['http_health'] else 'FAIL'}", end="")
            if result['response_time']:
                print(f" ({result['response_time']}ms)")
            else:
                print()
        else:
            print("Порт закрыт")
    
    print()
    print("=" * 60)
    print(f"📊 Сводка:")
    print(f"   ✅ Онлайн:     {online_count}")
    print(f"   ⚠️  Деградированы: {degraded_count}")
    print(f"   ❌ Офлайн:     {offline_count}")
    print(f"   📈 Всего:      {len(PORTS_CONFIG)}")
    
    # Определяем общий статус системы
    if offline_count == 0 and degraded_count == 0:
        system_status = "🟢 ОТЛИЧНО"
    elif offline_count == 0:
        system_status = "🟡 ХОРОШО"
    elif offline_count < len(PORTS_CONFIG) / 2:
        system_status = "🟠 УДОВЛЕТВОРИТЕЛЬНО"
    else:
        system_status = "🔴 КРИТИЧНО"
    
    print(f"   🎯 Общий статус: {system_status}")
    
    # Рекомендации
    print()
    print("💡 Рекомендации:")
    if offline_count > 0:
        offline_services = [r['name'] for r in results if r['status'] == 'OFFLINE']
        print(f"   • Запустить недоступные сервисы: {', '.join(offline_services)}")
    
    if degraded_count > 0:
        degraded_services = [r['name'] for r in results if r['status'] == 'DEGRADED']
        print(f"   • Проверить работу сервисов: {', '.join(degraded_services)}")
    
    if online_count == len(PORTS_CONFIG):
        print("   • Все сервисы работают нормально! 🎉")

if __name__ == "__main__":
    main()