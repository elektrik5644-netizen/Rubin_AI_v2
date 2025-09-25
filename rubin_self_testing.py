#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система самотестирования Rubin AI
Позволяет Rubin AI рассказать о своих возможностях и провести диагностику всех модулей
"""

import requests
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import concurrent.futures

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubinSelfTesting:
    """Система самотестирования Rubin AI"""
    
    def __init__(self):
        self.servers = {
            'smart_dispatcher': {'port': 8080, 'name': 'Smart Dispatcher', 'endpoint': '/api/health'},
            'general_api': {'port': 8085, 'name': 'General API', 'endpoint': '/api/health'},
            'math_server': {'port': 8086, 'name': 'Math Server', 'endpoint': '/health'},
            'electrical_server': {'port': 8088, 'name': 'Electrical Server', 'endpoint': '/api/electrical/status'},
            'programming_server': {'port': 8089, 'name': 'Programming Server', 'endpoint': '/health'},
            'radiomechanics_server': {'port': 8090, 'name': 'Radiomechanics Server', 'endpoint': '/api/radiomechanics/status'},
            'controllers_server': {'port': 9000, 'name': 'Controllers Server', 'endpoint': '/api/controllers/topic/general'},
            'neural_network': {'port': 8091, 'name': 'Neural Network', 'endpoint': '/api/health'},
            'learning_system': {'port': 8092, 'name': 'Learning System', 'endpoint': '/api/learning/health'},
            'pytorch_system': {'port': 8093, 'name': 'PyTorch System', 'endpoint': '/api/health'},
            'knowledge_base': {'port': 8094, 'name': 'Knowledge Base', 'endpoint': '/api/knowledge/health'},
            'thinking_system': {'port': 8095, 'name': 'Thinking System', 'endpoint': '/api/thinking/health'},
            'cnn_system': {'port': 8096, 'name': 'CNN System', 'endpoint': '/api/cnn/health'},
            'rnn_system': {'port': 8097, 'name': 'RNN System', 'endpoint': '/api/rnn/health'},
            'gan_system': {'port': 8098, 'name': 'GAN System', 'endpoint': '/api/gan/health'},
            'argumentation_system': {'port': 8100, 'name': 'Argumentation System', 'endpoint': '/api/argumentation/health'},
            'thoughts_communicator': {'port': 8101, 'name': 'Thoughts Communicator', 'endpoint': '/api/thoughts/health'}
        }
        
        self.capabilities = {
            'smart_dispatcher': [
                "Интеллектуальная маршрутизация вопросов",
                "Определение типа задачи",
                "Направление к специализированным серверам",
                "Генерация подробных ответов с шаблонами"
            ],
            'general_api': [
                "Ответы на общие вопросы",
                "Обработка неспецифичных запросов",
                "Базовые знания и информация"
            ],
            'math_server': [
                "Решение математических задач",
                "Арифметические операции",
                "Алгебраические уравнения",
                "Геометрические расчеты",
                "Статистические вычисления"
            ],
            'electrical_server': [
                "Электротехнические расчеты",
                "Закон Ома и Кирхгофа",
                "Анализ электрических цепей",
                "Полупроводниковые приборы",
                "Электрические схемы"
            ],
            'programming_server': [
                "Программирование на различных языках",
                "Алгоритмы и структуры данных",
                "Паттерны проектирования",
                "Отладка кода",
                "Архитектура программ"
            ],
            'radiomechanics_server': [
                "Радиотехнические расчеты",
                "Антенны и радиоволны",
                "Модуляция и демодуляция",
                "Радиосвязь",
                "Электромагнитные поля"
            ],
            'controllers_server': [
                "Промышленная автоматизация",
                "ПЛК и ЧПУ",
                "Системы управления",
                "PID регуляторы",
                "Промышленные сети"
            ],
            'neural_network': [
                "Нейронные сети",
                "Машинное обучение",
                "Глубокое обучение",
                "Обработка данных",
                "Предсказательная аналитика"
            ],
            'learning_system': [
                "Адаптивное обучение",
                "Самообучение",
                "Улучшение на основе опыта",
                "Персонализация ответов"
            ],
            'pytorch_system': [
                "PyTorch фреймворк",
                "Тензорные операции",
                "Градиентный спуск",
                "Автоматическое дифференцирование"
            ],
            'knowledge_base': [
                "База знаний",
                "Хранение информации",
                "Поиск по знаниям",
                "Структурированные данные"
            ],
            'thinking_system': [
                "Система мышления",
                "Абдуктивное рассуждение",
                "Нетривиальные запросы",
                "Креативное решение задач"
            ],
            'cnn_system': [
                "Сверточные нейронные сети",
                "Обработка изображений",
                "Распознавание образов",
                "Компьютерное зрение"
            ],
            'rnn_system': [
                "Рекуррентные нейронные сети",
                "Обработка последовательностей",
                "LSTM и GRU",
                "Временные ряды"
            ],
            'gan_system': [
                "Генеративно-состязательные сети",
                "Генерация данных",
                "Синтетические образцы",
                "Творческие задачи"
            ],
            'argumentation_system': [
                "Система аргументации",
                "Доказательство правоты",
                "Признание ошибок",
                "Логические рассуждения"
            ],
            'thoughts_communicator': [
                "Общение о мыслях",
                "Обмен идеями",
                "Творческие инсайты",
                "Процесс мышления"
            ]
        }
        
        self.test_questions = {
            'math_server': "Реши уравнение: 2x + 5 = 13",
            'electrical_server': "Что такое закон Ома?",
            'programming_server': "Напиши функцию сортировки массива",
            'radiomechanics_server': "Как работает антенна?",
            'controllers_server': "Что такое ПЛК?",
            'neural_network': "Объясни принцип работы нейронной сети",
            'thinking_system': "Найди неочевидную связь между математикой и музыкой",
            'argumentation_system': "Докажи, что программирование - это искусство"
        }
        
        logger.info("🧪 Система самотестирования Rubin AI инициализирована")
    
    def run_full_self_test(self) -> Dict[str, Any]:
        """Запуск полного самотестирования"""
        logger.info("🚀 Запуск полного самотестирования Rubin AI...")
        
        start_time = datetime.now()
        
        # Проверяем доступность всех серверов
        server_status = self._check_all_servers()
        
        # Тестируем функциональность доступных серверов
        functionality_tests = self._test_functionality()
        
        # Генерируем отчет о возможностях
        capabilities_report = self._generate_capabilities_report()
        
        # Создаем итоговый отчет
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = {
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "server_status": server_status,
            "functionality_tests": functionality_tests,
            "capabilities_report": capabilities_report,
            "summary": self._generate_summary(server_status, functionality_tests)
        }
        
        logger.info(f"✅ Самотестирование завершено за {duration:.2f} секунд")
        return report
    
    def _check_all_servers(self) -> Dict[str, Any]:
        """Проверка доступности всех серверов"""
        logger.info("🔍 Проверка доступности серверов...")
        
        server_status = {}
        
        for server_id, server_info in self.servers.items():
            try:
                health_url = f"http://localhost:{server_info['port']}{server_info['endpoint']}"
                response = requests.get(health_url, timeout=3)
                
                if response.status_code == 200:
                    server_status[server_id] = {
                        "status": "online",
                        "port": server_info['port'],
                        "name": server_info['name'],
                        "response_time": response.elapsed.total_seconds(),
                        "health_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else None
                    }
                    logger.info(f"✅ {server_info['name']} (порт {server_info['port']}) - ОНЛАЙН")
                else:
                    server_status[server_id] = {
                        "status": "error",
                        "port": server_info['port'],
                        "name": server_info['name'],
                        "error": f"HTTP {response.status_code}",
                        "response_time": response.elapsed.total_seconds()
                    }
                    logger.warning(f"❌ {server_info['name']} (порт {server_info['port']}) - ОШИБКА HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                server_status[server_id] = {
                    "status": "offline",
                    "port": server_info['port'],
                    "name": server_info['name'],
                    "error": "Connection refused"
                }
                logger.warning(f"❌ {server_info['name']} (порт {server_info['port']}) - НЕДОСТУПЕН")
                
            except requests.exceptions.Timeout:
                server_status[server_id] = {
                    "status": "timeout",
                    "port": server_info['port'],
                    "name": server_info['name'],
                    "error": "Request timeout"
                }
                logger.warning(f"⏰ {server_info['name']} (порт {server_info['port']}) - ТАЙМАУТ")
                
            except Exception as e:
                server_status[server_id] = {
                    "status": "error",
                    "port": server_info['port'],
                    "name": server_info['name'],
                    "error": str(e)
                }
                logger.error(f"💥 {server_info['name']} (порт {server_info['port']}) - ОШИБКА: {e}")
        
        return server_status
    
    def _test_functionality(self) -> Dict[str, Any]:
        """Тестирование функциональности доступных серверов"""
        logger.info("🧪 Тестирование функциональности серверов...")
        
        functionality_tests = {}
        
        for server_id, test_question in self.test_questions.items():
            if server_id not in self.servers:
                continue
                
            server_info = self.servers[server_id]
            
            try:
                # Определяем endpoint для тестирования
                if server_id == 'math_server':
                    test_url = f"http://localhost:{server_info['port']}/api/solve"
                elif server_id in ['electrical_server', 'programming_server', 'radiomechanics_server', 'controllers_server']:
                    test_url = f"http://localhost:{server_info['port']}/api/chat"
                elif server_id == 'neural_network':
                    test_url = f"http://localhost:{server_info['port']}/api/neural/process"
                elif server_id == 'thinking_system':
                    test_url = f"http://localhost:{server_info['port']}/api/thinking/analyze"
                elif server_id == 'argumentation_system':
                    test_url = f"http://localhost:{server_info['port']}/api/argumentation/create_argument"
                else:
                    test_url = f"http://localhost:{server_info['port']}/api/test"
                
                # Отправляем тестовый запрос
                response = requests.post(
                    test_url,
                    json={"message": test_question},
                    timeout=10
                )
                
                if response.status_code == 200:
                    functionality_tests[server_id] = {
                        "status": "working",
                        "test_question": test_question,
                        "response_time": response.elapsed.total_seconds(),
                        "response_preview": str(response.json())[:200] + "..." if len(str(response.json())) > 200 else str(response.json())
                    }
                    logger.info(f"✅ {server_info['name']} - ФУНКЦИОНАЛЬНОСТЬ РАБОТАЕТ")
                else:
                    functionality_tests[server_id] = {
                        "status": "error",
                        "test_question": test_question,
                        "error": f"HTTP {response.status_code}",
                        "response_time": response.elapsed.total_seconds()
                    }
                    logger.warning(f"❌ {server_info['name']} - ОШИБКА ФУНКЦИОНАЛЬНОСТИ HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                functionality_tests[server_id] = {
                    "status": "offline",
                    "test_question": test_question,
                    "error": "Server offline"
                }
                logger.warning(f"❌ {server_info['name']} - СЕРВЕР НЕДОСТУПЕН")
                
            except Exception as e:
                functionality_tests[server_id] = {
                    "status": "error",
                    "test_question": test_question,
                    "error": str(e)
                }
                logger.error(f"💥 {server_info['name']} - ОШИБКА: {e}")
        
        return functionality_tests
    
    def _generate_capabilities_report(self) -> Dict[str, Any]:
        """Генерация отчета о возможностях"""
        logger.info("📊 Генерация отчета о возможностях...")
        
        total_capabilities = 0
        capabilities_by_category = {}
        
        for server_id, capabilities_list in self.capabilities.items():
            capabilities_by_category[server_id] = {
                "name": self.servers.get(server_id, {}).get('name', server_id),
                "capabilities": capabilities_list,
                "count": len(capabilities_list)
            }
            total_capabilities += len(capabilities_list)
        
        return {
            "total_capabilities": total_capabilities,
            "capabilities_by_category": capabilities_by_category,
            "categories_count": len(self.capabilities)
        }
    
    def _generate_summary(self, server_status: Dict, functionality_tests: Dict) -> Dict[str, Any]:
        """Генерация итогового отчета"""
        
        # Подсчитываем статистику
        total_servers = len(server_status)
        online_servers = len([s for s in server_status.values() if s['status'] == 'online'])
        working_servers = len([s for s in functionality_tests.values() if s['status'] == 'working'])
        
        # Определяем основные возможности
        main_capabilities = []
        if server_status.get('smart_dispatcher', {}).get('status') == 'online':
            main_capabilities.append("Интеллектуальная маршрутизация")
        if server_status.get('math_server', {}).get('status') == 'online':
            main_capabilities.append("Математические вычисления")
        if server_status.get('electrical_server', {}).get('status') == 'online':
            main_capabilities.append("Электротехнические расчеты")
        if server_status.get('programming_server', {}).get('status') == 'online':
            main_capabilities.append("Программирование")
        if server_status.get('neural_network', {}).get('status') == 'online':
            main_capabilities.append("Нейронные сети")
        if server_status.get('thinking_system', {}).get('status') == 'online':
            main_capabilities.append("Система мышления")
        if server_status.get('argumentation_system', {}).get('status') == 'online':
            main_capabilities.append("Аргументация и споры")
        
        return {
            "total_servers": total_servers,
            "online_servers": online_servers,
            "working_servers": working_servers,
            "availability_percentage": (online_servers / total_servers * 100) if total_servers > 0 else 0,
            "functionality_percentage": (working_servers / total_servers * 100) if total_servers > 0 else 0,
            "main_capabilities": main_capabilities,
            "status": "excellent" if online_servers >= total_servers * 0.8 else "good" if online_servers >= total_servers * 0.6 else "needs_attention"
        }
    
    def generate_self_description(self) -> str:
        """Генерация описания возможностей Rubin AI"""
        
        report = self.run_full_self_test()
        summary = report['summary']
        
        description = f"""
🤖 **RUBIN AI - СИСТЕМА ИСКУССТВЕННОГО ИНТЕЛЛЕКТА**

**📊 ОБЩАЯ СТАТИСТИКА:**
• Всего модулей: {summary['total_servers']}
• Онлайн модулей: {summary['online_servers']}
• Рабочих модулей: {summary['working_servers']}
• Доступность: {summary['availability_percentage']:.1f}%
• Функциональность: {summary['functionality_percentage']:.1f}%

**🚀 ОСНОВНЫЕ ВОЗМОЖНОСТИ:**
"""
        
        for capability in summary['main_capabilities']:
            description += f"• {capability}\n"
        
        description += f"""
**🧠 СПЕЦИАЛИЗИРОВАННЫЕ МОДУЛИ:**

**📐 Математика:**
• Решение уравнений и задач
• Арифметические операции
• Геометрические расчеты
• Статистический анализ

**⚡ Электротехника:**
• Закон Ома и Кирхгофа
• Анализ электрических цепей
• Полупроводниковые приборы
• Электрические схемы

**💻 Программирование:**
• Различные языки программирования
• Алгоритмы и структуры данных
• Паттерны проектирования
• Отладка и оптимизация

**📡 Радиомеханика:**
• Радиотехнические расчеты
• Антенны и радиоволны
• Модуляция и передача сигналов
• Электромагнитные поля

**🏭 Промышленная автоматизация:**
• ПЛК и ЧПУ системы
• Системы управления
• PID регуляторы
• Промышленные сети

**🧠 Искусственный интеллект:**
• Нейронные сети (CNN, RNN, GAN)
• Машинное обучение
• Глубокое обучение
• Обработка данных

**💭 Система мышления:**
• Абдуктивное рассуждение
• Нетривиальные запросы
• Креативное решение задач
• Обучение на опыте

**⚖️ Аргументация:**
• Доказательство правоты
• Признание ошибок
• Логические рассуждения
• Структурированные споры

**🎯 ОСОБЕННОСТИ:**
• Интеллектуальная маршрутизация вопросов
• Адаптивное обучение
• Самоанализ и улучшение
• Интеграция всех модулей

**📈 СТАТУС СИСТЕМЫ: {summary['status'].upper()}**

Я готов помочь с любыми техническими вопросами, от простых вычислений до сложных систем искусственного интеллекта! 🚀
"""
        
        return description

# Глобальный экземпляр системы самотестирования
_self_testing_system = None

def get_self_testing_system():
    """Получение глобального экземпляра системы самотестирования"""
    global _self_testing_system
    if _self_testing_system is None:
        _self_testing_system = RubinSelfTesting()
    return _self_testing_system

if __name__ == "__main__":
    print("🧪 ДЕМОНСТРАЦИЯ СИСТЕМЫ САМОТЕСТИРОВАНИЯ RUBIN AI")
    print("=" * 60)
    
    # Создаем систему самотестирования
    self_testing = get_self_testing_system()
    
    # Генерируем описание возможностей
    print("\n🤖 ОПИСАНИЕ ВОЗМОЖНОСТЕЙ RUBIN AI:")
    print(self_testing.generate_self_description())
    
    print("\n✅ Демонстрация завершена!")
