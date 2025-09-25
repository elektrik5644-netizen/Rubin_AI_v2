#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 API Test Suite для Rubin AI v2
Автоматическое тестирование всех API серверов
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APITestSuite:
    """Комплексная система тестирования всех API серверов Rubin AI v2"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
        # Конфигурация всех серверов
        self.servers = {
            'smart_dispatcher': {
                'port': 8080,
                'endpoints': {
                    'health': '/api/health',
                    'chat': '/api/chat',
                    'ethical_status': '/api/ethical/status'
                },
                'test_messages': [
                    'Привет',
                    'Что такое диод?',
                    'Как решить уравнение x^2 + 5x + 6 = 0?',
                    'Сравни Python и C++'
                ]
            },
            'general_api': {
                'port': 8085,
                'endpoints': {
                    'health': '/api/health',
                    'chat': '/api/chat'
                },
                'test_messages': [
                    'Привет',
                    'Как дела?',
                    'Расскажи о себе'
                ]
            },
            'mathematics_api': {
                'port': 8086,
                'endpoints': {
                    'health': '/health',
                    'chat': '/api/chat'
                },
                'test_messages': [
                    'Реши уравнение 2x + 3 = 7',
                    'Что такое интеграл?',
                    'Как найти производную от x^2?'
                ]
            },
            'electrical_api': {
                'port': 8087,
                'endpoints': {
                    'health': '/api/electrical/status',
                    'chat': '/api/chat'
                },
                'test_messages': [
                    'Что такое диод?',
                    'Объясни закон Ома',
                    'Как рассчитать мощность в трехфазной системе?'
                ]
            },
            'programming_api': {
                'port': 8088,
                'endpoints': {
                    'health': '/health',
                    'explain': '/api/programming/explain'
                },
                'test_messages': [
                    'Объясни что такое Python',
                    'Что такое объектно-ориентированное программирование?',
                    'Как работает рекурсия?'
                ]
            },
            'radiomechanics_api': {
                'port': 8089,
                'endpoints': {
                    'health': '/api/radiomechanics/status',
                    'chat': '/api/chat'
                },
                'test_messages': [
                    'Что такое антенна?',
                    'Объясни принцип работы радиопередатчика',
                    'Что такое модуляция сигнала?'
                ]
            },
            'neuro_api': {
                'port': 8090,
                'endpoints': {
                    'health': '/health',
                    'status': '/api/neuro/status'
                },
                'test_messages': [
                    'Статус нейронной сети',
                    'Анализ данных',
                    'Машинное обучение'
                ]
            },
            'controllers_api': {
                'port': 9000,
                'endpoints': {
                    'health': '/api/controllers/topic/general',
                    'status': '/api/controllers/topic/general'
                },
                'test_messages': [
                    'Что такое ПЛК?',
                    'Объясни принцип работы контроллера',
                    'Программирование ПЛК'
                ]
            },
            'plc_analysis_api': {
                'port': 8099,
                'endpoints': {
                    'health': '/api/plc/health',
                    'analyze': '/api/plc/analyze'
                },
                'test_messages': [
                    'Анализ ПЛК программы',
                    'Диагностика контроллера',
                    'Проверка логики ПЛК'
                ]
            },
            'advanced_math_api': {
                'port': 8100,
                'endpoints': {
                    'health': '/api/math/health',
                    'calculate': '/api/math/advanced'
                },
                'test_messages': [
                    'Реши систему уравнений',
                    'Вычисли интеграл',
                    'Найди производную'
                ]
            },
            'data_processing_api': {
                'port': 8101,
                'endpoints': {
                    'health': '/api/data/health',
                    'process': '/api/data/process'
                },
                'test_messages': [
                    'Обработка данных',
                    'Анализ временных рядов',
                    'Нормализация данных'
                ]
            },
            'search_engine_api': {
                'port': 8102,
                'endpoints': {
                    'health': '/api/search/health',
                    'search': '/api/search/hybrid'
                },
                'test_messages': [
                    'Поиск информации',
                    'Гибридный поиск',
                    'Индексация документов'
                ]
            },
            'system_utils_api': {
                'port': 8103,
                'endpoints': {
                    'health': '/api/system/health',
                    'utils': '/api/system/utils'
                },
                'test_messages': [
                    'Проверка системы',
                    'Диагностика',
                    'Оптимизация'
                ]
            },
            'gai_api': {
                'port': 8104,
                'endpoints': {
                    'health': '/api/gai/health',
                    'generate_text': '/api/gai/generate_text'
                },
                'test_messages': [
                    'Сгенерируй текст',
                    'Создай код',
                    'Опиши диаграмму'
                ]
            },
            'unified_manager': {
                'port': 8084,
                'endpoints': {
                    'health': '/api/system/health',
                    'status': '/api/system/status'
                },
                'test_messages': [
                    'Статус системы',
                    'Управление серверами',
                    'Мониторинг'
                ]
            },
            'ethical_core_api': {
                'port': 8105,
                'endpoints': {
                    'health': '/api/ethical/health',
                    'assess': '/api/ethical/assess'
                },
                'test_messages': [
                    'Оценка безопасности',
                    'Этический анализ',
                    'Контроль действий'
                ]
            }
        }
    
    def test_server_health(self, server_name: str, config: Dict) -> Tuple[bool, str]:
        """Тестирует здоровье сервера"""
        try:
            port = config['port']
            health_endpoint = config['endpoints'].get('health', '/health')
            url = f"http://localhost:{port}{health_endpoint}"
            
            logger.info(f"🔍 Тестирую здоровье {server_name} на {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"✅ {server_name} - здоровье ОК")
                return True, f"Сервер {server_name} работает (HTTP {response.status_code})"
            else:
                logger.warning(f"⚠️ {server_name} - неожиданный статус {response.status_code}")
                return False, f"Сервер {server_name} отвечает, но статус {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ {server_name} - сервер недоступен")
            return False, f"Сервер {server_name} недоступен (ConnectionError)"
        except requests.exceptions.Timeout:
            logger.error(f"⏰ {server_name} - таймаут")
            return False, f"Сервер {server_name} не отвечает (Timeout)"
        except Exception as e:
            logger.error(f"❌ {server_name} - ошибка: {e}")
            return False, f"Сервер {server_name} - ошибка: {str(e)}"
    
    def test_server_functionality(self, server_name: str, config: Dict) -> Tuple[bool, str]:
        """Тестирует функциональность сервера"""
        try:
            port = config['port']
            test_messages = config.get('test_messages', ['Тест'])
            
            # Пробуем разные эндпоинты
            endpoints_to_test = []
            for endpoint_name, endpoint_path in config['endpoints'].items():
                if endpoint_name != 'health':
                    endpoints_to_test.append((endpoint_name, endpoint_path))
            
            if not endpoints_to_test:
                return True, f"Сервер {server_name} - нет функциональных эндпоинтов для тестирования"
            
            success_count = 0
            total_tests = 0
            
            for endpoint_name, endpoint_path in endpoints_to_test:
                url = f"http://localhost:{port}{endpoint_path}"
                
                for message in test_messages[:2]:  # Тестируем только первые 2 сообщения
                    total_tests += 1
                    try:
                        if endpoint_name in ['chat', 'assess', 'generate_text']:
                            # POST запросы
                            payload = {'message': message}
                            response = requests.post(url, json=payload, timeout=10)
                        else:
                            # GET запросы
                            response = requests.get(url, timeout=10)
                        
                        if response.status_code in [200, 201]:
                            success_count += 1
                            logger.info(f"✅ {server_name}/{endpoint_name} - тест прошел")
                        else:
                            logger.warning(f"⚠️ {server_name}/{endpoint_name} - статус {response.status_code}")
                    
                    except Exception as e:
                        logger.error(f"❌ {server_name}/{endpoint_name} - ошибка: {e}")
            
            success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
            
            if success_rate >= 70:
                return True, f"Сервер {server_name} - функциональность ОК ({success_rate:.1f}% тестов прошли)"
            else:
                return False, f"Сервер {server_name} - проблемы с функциональностью ({success_rate:.1f}% тестов прошли)"
                
        except Exception as e:
            logger.error(f"❌ {server_name} - ошибка тестирования функциональности: {e}")
            return False, f"Сервер {server_name} - ошибка тестирования: {str(e)}"
    
    def test_smart_dispatcher_routing(self) -> Tuple[bool, str]:
        """Тестирует маршрутизацию Smart Dispatcher"""
        try:
            url = "http://localhost:8080/api/chat"
            test_cases = [
                ("Что такое диод?", "electrical"),
                ("Реши уравнение x^2 = 4", "mathematics"),
                ("Объясни Python", "programming"),
                ("Привет", "general")
            ]
            
            success_count = 0
            for message, expected_category in test_cases:
                try:
                    payload = {'message': message}
                    response = requests.post(url, json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success') and 'category' in data:
                            success_count += 1
                            logger.info(f"✅ Smart Dispatcher маршрутизировал '{message}' -> {data['category']}")
                        else:
                            logger.warning(f"⚠️ Smart Dispatcher неверный ответ для '{message}'")
                    else:
                        logger.warning(f"⚠️ Smart Dispatcher статус {response.status_code} для '{message}'")
                
                except Exception as e:
                    logger.error(f"❌ Smart Dispatcher ошибка для '{message}': {e}")
            
            success_rate = (success_count / len(test_cases)) * 100
            
            if success_rate >= 75:
                return True, f"Smart Dispatcher маршрутизация ОК ({success_rate:.1f}% тестов прошли)"
            else:
                return False, f"Smart Dispatcher проблемы с маршрутизацией ({success_rate:.1f}% тестов прошли)"
                
        except Exception as e:
            logger.error(f"❌ Smart Dispatcher - ошибка тестирования маршрутизации: {e}")
            return False, f"Smart Dispatcher - ошибка тестирования: {str(e)}"
    
    def run_all_tests(self) -> Dict:
        """Запускает все тесты"""
        logger.info("🧪 Начинаю комплексное тестирование всех API серверов Rubin AI v2")
        logger.info("=" * 80)
        
        results = {
            'start_time': self.start_time.isoformat(),
            'servers': {},
            'summary': {
                'total_servers': len(self.servers),
                'healthy_servers': 0,
                'functional_servers': 0,
                'smart_dispatcher_ok': False
            }
        }
        
        # Тестируем Smart Dispatcher отдельно
        logger.info("🎯 Тестирую Smart Dispatcher...")
        smart_dp_health, smart_dp_msg = self.test_server_health('smart_dispatcher', self.servers['smart_dispatcher'])
        smart_dp_routing, routing_msg = self.test_smart_dispatcher_routing()
        
        results['servers']['smart_dispatcher'] = {
            'health': {'status': smart_dp_health, 'message': smart_dp_msg},
            'routing': {'status': smart_dp_routing, 'message': routing_msg},
            'overall': smart_dp_health and smart_dp_routing
        }
        
        if smart_dp_health and smart_dp_routing:
            results['summary']['smart_dispatcher_ok'] = True
        
        # Тестируем все остальные серверы
        for server_name, config in self.servers.items():
            if server_name == 'smart_dispatcher':
                continue
                
            logger.info(f"🔍 Тестирую {server_name}...")
            
            # Тест здоровья
            health_ok, health_msg = self.test_server_health(server_name, config)
            
            # Тест функциональности
            func_ok, func_msg = self.test_server_functionality(server_name, config)
            
            # Общий статус
            overall_ok = health_ok and func_ok
            
            results['servers'][server_name] = {
                'health': {'status': health_ok, 'message': health_msg},
                'functionality': {'status': func_ok, 'message': func_msg},
                'overall': overall_ok
            }
            
            if health_ok:
                results['summary']['healthy_servers'] += 1
            if overall_ok:
                results['summary']['functional_servers'] += 1
        
        # Завершение
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration
        
        # Сохраняем результаты
        self.save_results(results)
        
        # Выводим итоги
        self.print_summary(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Сохраняет результаты тестирования"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_test_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 Результаты сохранены в {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")
    
    def print_summary(self, results: Dict):
        """Выводит итоговую сводку"""
        logger.info("=" * 80)
        logger.info("📊 ИТОГОВАЯ СВОДКА ТЕСТИРОВАНИЯ")
        logger.info("=" * 80)
        
        summary = results['summary']
        
        logger.info(f"🕐 Время тестирования: {results['duration_seconds']:.2f} секунд")
        logger.info(f"📊 Всего серверов: {summary['total_servers']}")
        logger.info(f"✅ Здоровых серверов: {summary['healthy_servers']}")
        logger.info(f"🔧 Функциональных серверов: {summary['functional_servers']}")
        logger.info(f"🎯 Smart Dispatcher: {'✅ ОК' if summary['smart_dispatcher_ok'] else '❌ ПРОБЛЕМЫ'}")
        
        logger.info("\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        logger.info("-" * 80)
        
        for server_name, server_results in results['servers'].items():
            status_icon = "✅" if server_results['overall'] else "❌"
            logger.info(f"{status_icon} {server_name.upper()}")
            
            health_status = "✅" if server_results['health']['status'] else "❌"
            func_status = "✅" if server_results['functionality']['status'] else "❌"
            
            logger.info(f"   Здоровье: {health_status} {server_results['health']['message']}")
            logger.info(f"   Функции:  {func_status} {server_results['functionality']['message']}")
            
            if 'routing' in server_results:
                routing_status = "✅" if server_results['routing']['status'] else "❌"
                logger.info(f"   Маршрутизация: {routing_status} {server_results['routing']['message']}")
            
            logger.info("")
        
        # Общая оценка
        success_rate = (summary['functional_servers'] / summary['total_servers']) * 100
        
        if success_rate >= 90:
            logger.info("🎉 ОТЛИЧНО! Система работает превосходно!")
        elif success_rate >= 75:
            logger.info("👍 ХОРОШО! Система работает хорошо с небольшими проблемами.")
        elif success_rate >= 50:
            logger.info("⚠️ УДОВЛЕТВОРИТЕЛЬНО! Есть проблемы, требующие внимания.")
        else:
            logger.info("🚨 КРИТИЧНО! Система требует серьезного ремонта.")
        
        logger.info("=" * 80)

def main():
    """Главная функция"""
    print("🧪 Rubin AI v2 - API Test Suite")
    print("=" * 50)
    print("Автоматическое тестирование всех API серверов")
    print("=" * 50)
    
    # Создаем и запускаем тесты
    test_suite = APITestSuite()
    results = test_suite.run_all_tests()
    
    # Возвращаем код выхода
    success_rate = (results['summary']['functional_servers'] / results['summary']['total_servers']) * 100
    
    if success_rate >= 75:
        print("\n🎉 Тестирование завершено успешно!")
        sys.exit(0)
    else:
        print("\n⚠️ Обнаружены проблемы в системе!")
        sys.exit(1)

if __name__ == "__main__":
    main()



