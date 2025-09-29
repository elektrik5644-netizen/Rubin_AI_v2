#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Итоговый отчет о состоянии Rubin AI
"""

import requests
import time
import json
from datetime import datetime

def check_server(port, name, endpoint="/api/health"):
    """Проверяет сервер и возвращает детальную информацию"""
    try:
        url = f"http://localhost:{port}{endpoint}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            return {
                "status": "ONLINE",
                "port": port,
                "name": name,
                "response_time": response.elapsed.total_seconds(),
                "details": "Сервер работает нормально"
            }
        else:
            return {
                "status": "ERROR",
                "port": port,
                "name": name,
                "response_time": response.elapsed.total_seconds(),
                "details": f"HTTP {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "status": "OFFLINE",
            "port": port,
            "name": name,
            "response_time": None,
            "details": "Сервер не запущен"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "TIMEOUT",
            "port": port,
            "name": name,
            "response_time": None,
            "details": "Превышено время ожидания"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "port": port,
            "name": name,
            "response_time": None,
            "details": str(e)
        }

def test_smart_dispatcher():
    """Тестирует Smart Dispatcher"""
    test_questions = [
        "Как дела?",
        "Реши уравнение x^2 + 5x + 6 = 0",
        "Расскажи о контроллерах",
        "Что такое PyTorch?"
    ]
    
    results = []
    for question in test_questions:
        try:
            response = requests.post(
                "http://localhost:8080/api/chat",
                json={"message": question},
                timeout=10
            )
            if response.status_code == 200:
                results.append("OK")
            else:
                results.append(f"HTTP {response.status_code}")
        except Exception as e:
            results.append("ERROR")
    
    return results

def main():
    print("=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ О СОСТОЯНИИ RUBIN AI")
    print("=" * 80)
    print(f"Время проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Список серверов
    servers = [
        (8080, "Smart Dispatcher", "/api/health"),
        (8081, "Enhanced Rubin API", "/api/health"),
        (8085, "General Server", "/api/health"),
        (8086, "Math Server", "/health"),
        (8087, "Electrical Server", "/api/health"),
        (8088, "Programming Server", "/api/health"),
        (8089, "Radiomechanics Server", "/api/health"),
        (9000, "Controllers Server", "/api/health"),
        (8090, "Neuro Server", "/api/health"),
        (8091, "Learning Server", "/api/learning/health"),
        (8092, "PyTorch Server", "/api/health")
    ]
    
    # Проверка серверов
    print("ПРОВЕРКА СЕРВЕРОВ:")
    print("-" * 80)
    
    online_count = 0
    server_results = []
    
    for port, name, endpoint in servers:
        result = check_server(port, name, endpoint)
        server_results.append(result)
        
        status_symbol = "✓" if result["status"] == "ONLINE" else "✗"
        print(f"{status_symbol} {name:<25} (порт {port:<4}) - {result['status']:<8} - {result['details']}")
        
        if result["status"] == "ONLINE":
            online_count += 1
    
    print("-" * 80)
    print(f"ОБЩИЙ СТАТУС: {online_count}/{len(servers)} серверов работают")
    print()
    
    # Тест Smart Dispatcher
    print("ТЕСТ SMART DISPATCHER:")
    print("-" * 80)
    
    test_results = test_smart_dispatcher()
    test_questions = [
        "Как дела?",
        "Реши уравнение x^2 + 5x + 6 = 0",
        "Расскажи о контроллерах",
        "Что такое PyTorch?"
    ]
    
    for i, (question, result) in enumerate(zip(test_questions, test_results)):
        print(f"Тест {i+1}: {question[:30]}... - {result}")
    
    print()
    
    # Рекомендации
    print("РЕКОМЕНДАЦИИ:")
    print("-" * 80)
    
    offline_servers = [r for r in server_results if r["status"] == "OFFLINE"]
    error_servers = [r for r in server_results if r["status"] == "ERROR"]
    
    if offline_servers:
        print("Необходимо запустить:")
        for server in offline_servers:
            print(f"  - {server['name']} (порт {server['port']})")
        print()
    
    if error_servers:
        print("Требуют внимания:")
        for server in error_servers:
            print(f"  - {server['name']} (порт {server['port']}) - {server['details']}")
        print()
    
    if online_count >= len(servers) * 0.9:
        print("✓ Система работает отлично!")
    elif online_count >= len(servers) * 0.7:
        print("⚠ Система работает с небольшими ограничениями")
    elif online_count >= len(servers) * 0.5:
        print("⚠ Система работает с ограничениями")
    else:
        print("✗ Критические проблемы с системой")
    
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПроверка прервана пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")










