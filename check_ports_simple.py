#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простая проверка портов Rubin AI без эмодзи
"""

import requests
import time
import sys

def check_port(port, name, endpoint="/api/health", timeout=2):
    """Проверяет доступность порта"""
    try:
        url = f"http://localhost:{port}{endpoint}"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, f"OK - {name} (порт {port})"
        else:
            return False, f"ERROR - {name} (порт {port}) - HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"OFFLINE - {name} (порт {port})"
    except requests.exceptions.Timeout:
        return False, f"TIMEOUT - {name} (порт {port})"
    except Exception as e:
        return False, f"ERROR - {name} (порт {port}) - {str(e)}"

def main():
    print("=" * 60)
    print("ПРОВЕРКА ПОРТОВ RUBIN AI")
    print("=" * 60)
    
    # Список серверов для проверки (порт, имя, endpoint)
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
    
    online_count = 0
    total_count = len(servers)
    
    print(f"Проверяю {total_count} серверов...")
    print("-" * 60)
    
    for port, name, endpoint in servers:
        is_online, message = check_port(port, name, endpoint)
        print(message)
        if is_online:
            online_count += 1
        time.sleep(0.1)  # Небольшая пауза между проверками
    
    print("-" * 60)
    print(f"РЕЗУЛЬТАТ: {online_count}/{total_count} серверов работают")
    
    if online_count == total_count:
        print("СТАТУС: Все серверы работают!")
    elif online_count >= total_count * 0.8:
        print("СТАТУС: Система работает с небольшими ограничениями")
    elif online_count >= total_count * 0.5:
        print("СТАТУС: Система работает с ограничениями")
    else:
        print("СТАТУС: Критические проблемы с системой")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПроверка прервана пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
