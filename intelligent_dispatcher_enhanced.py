#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расширенный интеллектуальный диспетчер для Rubin AI v2.0
С поддержкой математического решателя
"""

import time
import random
import threading
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

# Импорт математических компонентов
from mathematical_solver.integrated_solver import IntegratedMathematicalSolver, MathIntegrationConfig

@dataclass
class ModuleMetrics:
    """Метрики производительности модуля"""
    module_name: str
    port: int
    request_count: int = 0
    avg_response_time: float = 0.0
    error_count: int = 0
    last_health_check: Optional[datetime] = None
    status: str = "unknown"  # online, offline, degraded
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

@dataclass
class RequestMetrics:
    """Метрики запроса"""
    timestamp: datetime
    module: str
    response_time: float
    success: bool
    category: str

class EnhancedIntelligentDispatcher:
    """Расширенный интеллектуальный диспетчер с поддержкой математических запросов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Конфигурация модулей
        self.modules = {
            'controllers': {'port': 8090, 'name': 'Контроллеры'},
            'electrical': {'port': 8087, 'name': 'Электротехника'},
            'radiomechanics': {'port': 8089, 'name': 'Радиомеханика'},
            'ai_chat': {'port': 8084, 'name': 'AI Чат'},
            'documents': {'port': 8085, 'name': 'Документы'},
            # Добавляем математический модуль
            'mathematics': {'port': 8089, 'name': 'Математический решатель', 'type': 'internal'}
        }
        
        # Инициализация математического решателя
        math_config = MathIntegrationConfig(
            enabled=True,
            confidence_threshold=0.7,
            fallback_to_general=True,
            log_requests=True,
            response_format="structured"
        )
        self.mathematical_solver = IntegratedMathematicalSolver(math_config)
        
        # Метрики и состояние
        self.module_metrics = {}
        self.request_metrics = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Инициализация метрик
        for module_name, module_info in self.modules.items():
            self.module_metrics[module_name] = ModuleMetrics(
                module_name=module_name,
                port=module_info['port']
            )
        
        self.logger.info("Enhanced Intelligent Dispatcher initialized with mathematical solver")
    
    def analyze_request_category(self, message: str) -> str:
        """
        Расширенный анализ категории запроса с поддержкой математических задач
        
        Args:
            message: Сообщение пользователя
            
        Returns:
            Категория запроса
        """
        message_lower = message.lower()
        
        # Сначала проверяем математические запросы
        if self.mathematical_solver.is_mathematical_request(message):
            detected_category = self.mathematical_solver.get_detected_category(message)
            if detected_category:
                return f"mathematics_{detected_category}"
            else:
                return "mathematics_general"
        
        # Существующие категории
        categories = {
            'controllers': ['пид', 'регулятор', 'plc', 'контроллер', 'автоматизация', 'программирование', 'логика'],
            'electrical': ['электричество', 'схема', 'ток', 'напряжение', 'резистор', 'конденсатор', 'закон ома'],
            'radiomechanics': ['антенна', 'сигнал', 'радио', 'модуляция', 'частота', 'передатчик', 'приемник'],
            'documents': ['документ', 'файл', 'поиск', 'база данных', 'информация']
        }
        
        # Подсчет совпадений
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            scores[category] = score
        
        # Возврат категории с наибольшим score
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'general'
    
    def get_available_modules(self, category: str) -> List[str]:
        """Получение списка доступных модулей для категории"""
        with self.lock:
            available = []
            
            # Обработка математических категорий
            if category.startswith('mathematics_'):
                # Математический решатель всегда доступен (внутренний модуль)
                available.append('mathematics')
                return available
            
            # Специализированные модули для категории
            if category in self.modules:
                if self.module_metrics[category].status == 'online':
                    available.append(category)
            
            # Fallback модули
            if category == 'general':
                # Для общих запросов используем AI чат
                if self.module_metrics['ai_chat'].status == 'online':
                    available.append('ai_chat')
                # Если AI чат недоступен, используем математический решатель как fallback
                elif category.startswith('mathematics_') or self._is_potentially_mathematical(category):
                    available.append('mathematics')
            
            return available if available else ['ai_chat']  # Последний fallback
    
    def _is_potentially_mathematical(self, message: str) -> bool:
        """Проверяет, может ли сообщение быть математическим"""
        math_indicators = ['число', 'вычисли', 'найди', 'сколько', 'реши', 'расчет', '+', '-', '*', '/', '=']
        return any(indicator in message.lower() for indicator in math_indicators)
    
    def process_request(self, message: str, category: str = None) -> Dict[str, any]:
        """
        Обрабатывает запрос с поддержкой математических задач
        
        Args:
            message: Сообщение пользователя
            category: Категория запроса (опционально)
            
        Returns:
            Результат обработки запроса
        """
        start_time = time.time()
        
        # Определяем категорию, если не указана
        if not category:
            category = self.analyze_request_category(message)
        
        self.logger.info(f"🔍 Анализирую вопрос: \"{message}\"")
        
        # Обработка математических запросов
        if category.startswith('mathematics_'):
            return self._process_mathematical_request(message, category)
        
        # Обработка обычных запросов
        available_modules = self.get_available_modules(category)
        
        if not available_modules:
            return self._create_fallback_response(message, "Нет доступных модулей")
        
        # Выбираем лучший модуль
        selected_module = self._select_best_module(available_modules)
        
        if selected_module == 'mathematics':
            return self._process_mathematical_request(message, category)
        
        # Перенаправляем к выбранному модулю
        self.logger.info(f"📡 Направляю к модулю: {self.modules[selected_module]['name']} (порт {self.modules[selected_module]['port']})")
        
        try:
            response = self._forward_to_module(message, selected_module)
            processing_time = time.time() - start_time
            
            # Обновляем метрики
            self._update_request_metrics(selected_module, processing_time, True, category)
            
            return {
                "response": response,
                "module": selected_module,
                "category": category,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка при обработке запроса: {e}")
            
            # Обновляем метрики ошибки
            self._update_request_metrics(selected_module, processing_time, False, category)
            
            # Fallback к математическому решателю, если возможно
            if self.mathematical_solver.is_mathematical_request(message):
                return self._process_mathematical_request(message, "mathematics_general")
            
            return self._create_fallback_response(message, str(e))
    
    def _process_mathematical_request(self, message: str, category: str) -> Dict[str, any]:
        """Обрабатывает математический запрос"""
        start_time = time.time()
        
        self.logger.info(f"🧮 Обрабатываю математический запрос: {category}")
        
        try:
            # Обрабатываем через математический решатель
            result = self.mathematical_solver.process_request(message)
            
            processing_time = time.time() - start_time
            
            # Если это не математический запрос, перенаправляем
            if result.get("should_route_to_other"):
                self.logger.info("📡 Запрос не математический, перенаправляю к другим модулям")
                return self.process_request(message, "general")
            
            # Форматируем ответ
            if result.get("success"):
                response_text = result.get("response", "Решение получено")
                self.logger.info(f"✅ Математическая задача решена за {processing_time:.3f}с")
            else:
                response_text = result.get("response", "Ошибка при решении задачи")
                self.logger.warning(f"⚠️ Ошибка при решении математической задачи: {result.get('error_message')}")
            
            # Обновляем метрики
            self._update_request_metrics("mathematics", processing_time, result.get("success", False), category)
            
            return {
                "response": response_text,
                "module": "mathematics",
                "category": category,
                "processing_time": processing_time,
                "success": result.get("success", False),
                "confidence": result.get("confidence", 0.0),
                "solution_data": result.get("solution_data", {})
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ошибка в математическом решателе: {e}")
            
            # Обновляем метрики ошибки
            self._update_request_metrics("mathematics", processing_time, False, category)
            
            # Fallback к общему AI чату
            return self._create_fallback_response(message, f"Ошибка математического решателя: {str(e)}")
    
    def _select_best_module(self, available_modules: List[str]) -> str:
        """Выбирает лучший модуль из доступных"""
        if not available_modules:
            return 'ai_chat'
        
        # Простой выбор - берем первый доступный
        # В будущем можно добавить более сложную логику
        return available_modules[0]
    
    def _forward_to_module(self, message: str, module_name: str) -> str:
        """Перенаправляет запрос к указанному модулю"""
        module_info = self.modules[module_name]
        
        # Для математического модуля используем внутренний решатель
        if module_name == 'mathematics':
            result = self.mathematical_solver.process_request(message)
            return result.get("response", "Ошибка математического решателя")
        
        # Для внешних модулей используем HTTP запросы
        try:
            url = f"http://localhost:{module_info['port']}/chat"
            response = requests.post(url, json={"message": message}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "Ответ получен")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Ошибка соединения с модулем {module_name}: {str(e)}")
    
    def _create_fallback_response(self, message: str, error: str) -> Dict[str, any]:
        """Создает fallback ответ при ошибках"""
        return {
            "response": f"Извините, произошла ошибка при обработке запроса: {error}",
            "module": "fallback",
            "category": "error",
            "processing_time": 0.0,
            "success": False,
            "error": error
        }
    
    def _update_request_metrics(self, module: str, response_time: float, success: bool, category: str):
        """Обновляет метрики запросов"""
        with self.lock:
            # Добавляем метрику запроса
            self.request_metrics.append(RequestMetrics(
                timestamp=datetime.now(),
                module=module,
                response_time=response_time,
                success=success,
                category=category
            ))
            
            # Обновляем метрики модуля
            if module in self.module_metrics:
                metrics = self.module_metrics[module]
                metrics.request_count += 1
                metrics.avg_response_time = (
                    (metrics.avg_response_time * (metrics.request_count - 1) + response_time) / 
                    metrics.request_count
                )
                if not success:
                    metrics.error_count += 1
    
    def get_system_status(self) -> Dict[str, any]:
        """Получает статус системы с математическим решателем"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "modules": {},
            "mathematical_solver": {},
            "overall_status": "operational"
        }
        
        # Статус обычных модулей
        with self.lock:
            for module_name, metrics in self.module_metrics.items():
                if module_name != 'mathematics':  # Математический модуль обрабатываем отдельно
                    status["modules"][module_name] = {
                        "status": metrics.status,
                        "port": metrics.port,
                        "request_count": metrics.request_count,
                        "avg_response_time": metrics.avg_response_time,
                        "error_count": metrics.error_count
                    }
        
        # Статус математического решателя
        try:
            math_status = self.mathematical_solver.get_solver_status()
            status["mathematical_solver"] = math_status
            
            if not math_status.get("status") == "operational":
                status["overall_status"] = "degraded"
                
        except Exception as e:
            status["mathematical_solver"] = {"status": "error", "error": str(e)}
            status["overall_status"] = "degraded"
        
        return status
    
    def health_check(self) -> Dict[str, any]:
        """Проверка здоровья системы"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "components": {}
        }
        
        # Проверяем математический решатель
        try:
            math_health = self.mathematical_solver.health_check()
            health["components"]["mathematical_solver"] = math_health
            
            if math_health["overall_status"] != "healthy":
                health["overall_health"] = "degraded"
                
        except Exception as e:
            health["components"]["mathematical_solver"] = {
                "overall_status": "unhealthy",
                "error": str(e)
            }
            health["overall_health"] = "unhealthy"
        
        # Проверяем другие модули (упрощенная проверка)
        for module_name, module_info in self.modules.items():
            if module_name != 'mathematics':
                health["components"][module_name] = {
                    "status": self.module_metrics[module_name].status
                }
        
        return health













