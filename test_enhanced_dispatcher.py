#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для Enhanced Smart Dispatcher
"""

import unittest
import json
import requests
import time
from datetime import datetime
import logging

# Настройка логирования для тестов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEnhancedDispatcher(unittest.TestCase):
    """Тесты для Enhanced Smart Dispatcher"""
    
    def setUp(self):
        """Настройка тестов"""
        self.base_url = "http://localhost:8080"
        self.test_messages = [
            "привет",
            "2+2",
            "что такое транзистор",
            "закон кирхгофа",
            "напиши программу на python",
            "площадь круга радиусом 5",
            "sin(30)",
            "что такое pmac",
            "диагностика системы",
            "нейронная сеть"
        ]
    
    def test_health_check(self):
        """Тест проверки здоровья"""
        logger.info("🔍 Тестирование health check...")
        response = requests.get(f"{self.base_url}/api/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("neural_router", data)
        logger.info("✅ Health check прошел успешно")
    
    def test_neural_analysis(self):
        """Тест нейронного анализа"""
        logger.info("🧠 Тестирование нейронного анализа...")
        
        test_message = "что такое транзистор"
        response = requests.post(
            f"{self.base_url}/api/neural/analyze",
            json={"message": test_message}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("category", data)
        self.assertIn("confidence", data)
        self.assertIn("suggested_server", data)
        logger.info(f"✅ Нейронный анализ: категория={data['category']}, уверенность={data['confidence']:.2f}")
    
    def test_chat_endpoint(self):
        """Тест основного чат endpoint"""
        logger.info("💬 Тестирование чат endpoint...")
        
        for message in self.test_messages:
            logger.info(f"  Тестируем: '{message}'")
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"message": message}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Проверяем обязательные поля
            self.assertIn("routed_to", data)
            self.assertIn("confidence", data)
            self.assertIn("timestamp", data)
            self.assertIn("neural_analysis", data)
            
            logger.info(f"    ✅ Ответ: сервер={data['routed_to']}, уверенность={data['confidence']:.2f}")
            
            # Небольшая пауза между запросами
            time.sleep(0.1)
    
    def test_servers_list(self):
        """Тест списка серверов"""
        logger.info("📋 Тестирование списка серверов...")
        
        response = requests.get(f"{self.base_url}/api/servers")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("servers", data)
        self.assertIn("total_count", data)
        self.assertGreater(data["total_count"], 0)
        
        logger.info(f"✅ Найдено серверов: {data['total_count']}")
    
    def test_system_health(self):
        """Тест здоровья системы"""
        logger.info("🏥 Тестирование здоровья системы...")
        
        response = requests.get(f"{self.base_url}/api/system/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("total_servers", data)
        self.assertIn("healthy_servers", data)
        self.assertIn("health_percentage", data)
        
        logger.info(f"✅ Здоровье системы: {data['health_percentage']:.1f}%")
    
    def test_errors_endpoint(self):
        """Тест endpoint ошибок"""
        logger.info("❌ Тестирование endpoint ошибок...")
        
        response = requests.get(f"{self.base_url}/api/errors")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("errors", data)
        self.assertIn("total_count", data)
        
        logger.info(f"✅ Найдено ошибок: {data['total_count']}")
    
    def test_status_endpoint(self):
        """Тест endpoint статуса"""
        logger.info("📊 Тестирование endpoint статуса...")
        
        response = requests.get(f"{self.base_url}/api/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["neural_router"], "active")
        self.assertEqual(data["error_tracker"], "active")
        
        logger.info("✅ Статус системы: running")
    
    def test_empty_message(self):
        """Тест пустого сообщения"""
        logger.info("🔍 Тестирование пустого сообщения...")
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"message": ""}
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        
        logger.info("✅ Пустое сообщение обработано корректно")
    
    def test_special_characters(self):
        """Тест специальных символов"""
        logger.info("🔤 Тестирование специальных символов...")
        
        special_messages = [
            "2+2=4",
            "U=I×R",
            "sin(π/2)",
            "√16=4",
            "α + β = γ"
        ]
        
        for message in special_messages:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"message": message}
            )
            
            self.assertEqual(response.status_code, 200)
            logger.info(f"  ✅ Специальные символы: '{message}'")
    
    def test_long_message(self):
        """Тест длинного сообщения"""
        logger.info("📝 Тестирование длинного сообщения...")
        
        long_message = "Это очень длинное сообщение " * 50
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"message": long_message}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("routed_to", data)
        
        logger.info("✅ Длинное сообщение обработано")
    
    def test_performance(self):
        """Тест производительности"""
        logger.info("⚡ Тестирование производительности...")
        
        start_time = time.time()
        successful_requests = 0
        
        for i in range(10):
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"message": f"тест {i+1}"}
            )
            
            if response.status_code == 200:
                successful_requests += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10
        
        logger.info(f"✅ Производительность: {successful_requests}/10 запросов, среднее время: {avg_time:.3f}с")
        
        self.assertEqual(successful_requests, 10)
        self.assertLess(avg_time, 2.0)  # Среднее время должно быть меньше 2 секунд

class TestHandlers(unittest.TestCase):
    """Тесты для обработчиков"""
    
    def setUp(self):
        """Настройка тестов"""
        from handlers.general_handler import get_general_handler
        from handlers.electrical_handler import get_electrical_handler
        from handlers.mathematics_handler import get_mathematics_handler
        
        self.general_handler = get_general_handler()
        self.electrical_handler = get_electrical_handler()
        self.mathematics_handler = get_mathematics_handler()
    
    def test_general_handler(self):
        """Тест общего обработчика"""
        logger.info("🔧 Тестирование General Handler...")
        
        test_cases = [
            "привет",
            "помощь",
            "как дела",
            "что ты умеешь"
        ]
        
        for message in test_cases:
            result = self.general_handler.handle_request(message)
            
            self.assertIn("success", result)
            self.assertIn("response", result)
            self.assertIn("category", result)
            self.assertIn("confidence", result)
            
            logger.info(f"  ✅ '{message}' -> {result['category']} (уверенность: {result['confidence']:.2f})")
    
    def test_electrical_handler(self):
        """Тест электротехнического обработчика"""
        logger.info("⚡ Тестирование Electrical Handler...")
        
        test_cases = [
            "что такое транзистор",
            "закон ома",
            "закон кирхгофа",
            "что такое сервопривод"
        ]
        
        for message in test_cases:
            result = self.electrical_handler.handle_request(message)
            
            self.assertIn("success", result)
            self.assertIn("response", result)
            self.assertIn("category", result)
            
            logger.info(f"  ✅ '{message}' -> {result['category']} (уверенность: {result['confidence']:.2f})")
    
    def test_mathematics_handler(self):
        """Тест математического обработчика"""
        logger.info("🧮 Тестирование Mathematics Handler...")
        
        test_cases = [
            "2+2",
            "площадь круга радиусом 5",
            "sin(30)",
            "формула площади треугольника"
        ]
        
        for message in test_cases:
            result = self.mathematics_handler.handle_request(message)
            
            self.assertIn("success", result)
            self.assertIn("response", result)
            self.assertIn("category", result)
            
            logger.info(f"  ✅ '{message}' -> {result['category']} (уверенность: {result['confidence']:.2f})")

def run_integration_tests():
    """Запуск интеграционных тестов"""
    logger.info("🚀 Запуск интеграционных тестов...")
    
    # Проверяем доступность сервера
    try:
        response = requests.get("http://localhost:8080/api/health", timeout=5)
        if response.status_code != 200:
            logger.error("❌ Сервер недоступен")
            return False
    except requests.exceptions.RequestException:
        logger.error("❌ Не удается подключиться к серверу")
        return False
    
    # Запускаем тесты
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты диспетчера
    test_suite.addTest(loader.loadTestsFromTestCase(TestEnhancedDispatcher))
    
    # Добавляем тесты обработчиков
    test_suite.addTest(loader.loadTestsFromTestCase(TestHandlers))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Запуск тестов Enhanced Smart Dispatcher")
    print("=" * 50)
    
    success = run_integration_tests()
    
    if success:
        print("\n✅ Все тесты прошли успешно!")
    else:
        print("\n❌ Некоторые тесты не прошли")
    
    print("=" * 50)