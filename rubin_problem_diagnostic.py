#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔧 ДИАГНОСТИКА И ИСПРАВЛЕНИЕ ПРОБЛЕМ RUBIN AI
=============================================
"""

import requests
import json
import time
from datetime import datetime

class RubinProblemDiagnostic:
    """Диагностика проблем Rubin AI"""
    
    def __init__(self):
        self.servers = {
            'smart_dispatcher': {'port': 8080, 'endpoint': '/api/health', 'status': 'unknown'},
            'general_api': {'port': 8085, 'endpoint': '/api/health', 'status': 'unknown'},
            'mathematics': {'port': 8086, 'endpoint': '/health', 'status': 'unknown'},
            'electrical': {'port': 8087, 'endpoint': '/api/electrical/status', 'status': 'unknown'},
            'programming': {'port': 8088, 'endpoint': '/api/programming/explain', 'status': 'unknown'},
            'radiomechanics': {'port': 8089, 'endpoint': '/api/radiomechanics/status', 'status': 'unknown'},
            'controllers': {'port': 9000, 'endpoint': '/api/health', 'status': 'unknown'},
            'neuro': {'port': 8090, 'endpoint': '/api/health', 'status': 'unknown'},
            'learning': {'port': 8091, 'endpoint': '/api/learning/health', 'status': 'unknown'}
        }
        
        self.problems = []
        self.solutions = []
    
    def diagnose_all_servers(self):
        """Диагностика всех серверов"""
        print("🔍 ДИАГНОСТИКА ПРОБЛЕМ RUBIN AI")
        print("=" * 50)
        
        for server_name, config in self.servers.items():
            print(f"\n🔍 Проверяю {server_name}...")
            is_healthy, status, details = self.check_server_health(server_name, config)
            self.servers[server_name]['status'] = 'healthy' if is_healthy else 'unhealthy'
            
            if not is_healthy:
                self.problems.append({
                    'server': server_name,
                    'port': config['port'],
                    'status': status,
                    'details': details
                })
                print(f"❌ {server_name}: {status}")
            else:
                print(f"✅ {server_name}: Работает")
    
    def check_server_health(self, server_name, config):
        """Проверка здоровья сервера"""
        port = config['port']
        endpoint = config['endpoint']
        url = f"http://localhost:{port}{endpoint}"
        
        try:
            if server_name == 'programming':
                # Для programming сервера используем POST запрос
                response = requests.post(url, json={'concept': 'test'}, timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return True, "Здоров", response.json() if response.content else {}
            else:
                return False, f"HTTP {response.status_code}", f"Endpoint: {endpoint}"
        except requests.exceptions.ConnectionError:
            return False, "Недоступен", "Сервер не запущен"
        except requests.exceptions.Timeout:
            return False, "Таймаут", "Сервер не отвечает"
        except Exception as e:
            return False, f"Ошибка: {str(e)[:30]}", str(e)
    
    def generate_problem_report(self):
        """Генерация отчета о проблемах"""
        print(f"\n📋 ОТЧЕТ О ПРОБЛЕМАХ RUBIN AI")
        print("=" * 50)
        
        healthy_count = sum(1 for s in self.servers.values() if s['status'] == 'healthy')
        total_count = len(self.servers)
        
        print(f"📊 Общий статус: {healthy_count}/{total_count} серверов работают")
        
        if self.problems:
            print(f"\n❌ ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:")
            for i, problem in enumerate(self.problems, 1):
                print(f"{i}. {problem['server']} (порт {problem['port']}): {problem['status']}")
                print(f"   Детали: {problem['details']}")
        else:
            print("\n✅ Проблем не обнаружено!")
    
    def generate_solutions(self):
        """Генерация решений проблем"""
        print(f"\n🔧 РЕШЕНИЯ ПРОБЛЕМ:")
        print("=" * 30)
        
        for problem in self.problems:
            server = problem['server']
            port = problem['port']
            
            if server == 'general_api':
                solution = "Запустить: python general_server.py"
            elif server == 'electrical':
                solution = "Запустить: python electrical_server.py"
            elif server == 'programming':
                solution = "Запустить: python programming_server.py"
            elif server == 'radiomechanics':
                solution = "Запустить: python radiomechanics_server.py"
            elif server == 'controllers':
                solution = "Запустить: python controllers_server.py"
            elif server == 'learning':
                solution = "Запустить: python learning_server.py"
            else:
                solution = f"Проверить конфигурацию сервера {server}"
            
            print(f"🔧 {server}: {solution}")
    
    def test_smart_dispatcher_routing(self):
        """Тестирование маршрутизации Smart Dispatcher"""
        print(f"\n🧠 ТЕСТ МАРШРУТИЗАЦИИ SMART DISPATCHER:")
        print("-" * 40)
        
        test_questions = [
            "Как дела?",
            "Реши уравнение x^2 + 5x + 6 = 0",
            "Расскажи о контроллерах",
            "Что такое демо?"
        ]
        
        for question in test_questions:
            print(f"\n📝 Вопрос: {question}")
            try:
                response = requests.post('http://localhost:8080/api/chat', 
                                       json={'message': question}, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Ответ получен (HTTP {response.status_code})")
                    if 'response' in data:
                        print(f"📄 Ответ: {str(data['response'])[:100]}...")
                    else:
                        print(f"📄 Структура ответа: {list(data.keys())}")
                else:
                    print(f"❌ HTTP ошибка: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Ошибка: {e}")

def main():
    """Основная функция диагностики"""
    diagnostic = RubinProblemDiagnostic()
    
    # Диагностика всех серверов
    diagnostic.diagnose_all_servers()
    
    # Отчет о проблемах
    diagnostic.generate_problem_report()
    
    # Решения проблем
    diagnostic.generate_solutions()
    
    # Тест маршрутизации
    diagnostic.test_smart_dispatcher_routing()
    
    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    print("=" * 20)
    print("1. Запустить недостающие серверы")
    print("2. Проверить конфигурацию портов")
    print("3. Обновить Smart Dispatcher до v3.0")
    print("4. Настроить fallback механизмы")

if __name__ == "__main__":
    main()
