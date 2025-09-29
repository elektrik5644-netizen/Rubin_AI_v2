#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный интеллектуальный диспетчер для Rubin AI v2.0
С правильной маршрутизацией математических запросов
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

@dataclass
class RequestMetrics:
    """Метрики запроса"""
    timestamp: datetime
    module: str
    response_time: float
    success: bool
    category: str

class IntelligentDispatcherFixed:
    """Исправленный интеллектуальный диспетчер для маршрутизации запросов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Конфигурация модулей
        self.modules = {
            'mathematics': {'port': None, 'name': 'Математический решатель'},  # Локальный модуль
            'controllers': {'port': 8090, 'name': 'Контроллеры'},
            'electrical': {'port': 8087, 'name': 'Электротехника'},
            'radiomechanics': {'port': 8089, 'name': 'Радиомеханика'},
            'documents': {'port': 8088, 'name': 'Документы'},
            'general': {'port': 8084, 'name': 'Общие ответы'}
        }
        
        # Инициализация математического решателя
        self.math_handler = None
        self._initialize_math_handler()
        
        # Метрики модулей
        self.module_metrics: Dict[str, ModuleMetrics] = {}
        self.request_history: deque = deque(maxlen=1000)
        
        # Настройки
        self.load_balanced = True
        self.health_check_interval = 30  # секунд
        
        # Потокобезопасность
        self.lock = threading.Lock()
        
        # Инициализация метрик
        self._initialize_metrics()
        
        # Запуск мониторинга
        self._start_monitoring()
    
    def _initialize_math_handler(self):
        """Инициализация математического обработчика"""
        try:
            from mathematical_solver.integrated_solver import IntegratedMathematicalSolver, MathIntegrationConfig
            
            config = MathIntegrationConfig(
                enabled=True,
                confidence_threshold=0.7,
                fallback_to_general=False,
                log_requests=True,
                response_format="structured"
            )
            
            self.math_handler = IntegratedMathematicalSolver(config)
            self.logger.info("✅ Математический решатель инициализирован")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Не удалось импортировать интегрированный решатель: {e}")
            try:
                # Fallback к простому решателю
                from mathematical_problem_solver import MathematicalProblemSolver
                self.math_handler = MathematicalProblemSolver()
                self.logger.info("✅ Простой математический решатель инициализирован")
            except ImportError as e2:
                self.logger.error(f"❌ Не удалось инициализировать математический решатель: {e2}")
                self.math_handler = None
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации математического решателя: {e}")
            self.math_handler = None
    
    def _initialize_metrics(self):
        """Инициализация метрик для всех модулей"""
        for module_id, config in self.modules.items():
            self.module_metrics[module_id] = ModuleMetrics(
                module_name=config['name'],
                port=config['port']
            )
    
    def _start_monitoring(self):
        """Запуск фонового мониторинга"""
        def monitor_loop():
            while True:
                try:
                    self._update_health_status()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Ошибка мониторинга: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _update_health_status(self):
        """Обновление статуса здоровья модулей"""
        with self.lock:
            for module_id, config in self.modules.items():
                if config['port'] is None:  # Локальный модуль (математика)
                    self.module_metrics[module_id].status = "online"
                    self.module_metrics[module_id].last_health_check = datetime.now()
                    continue
                
                try:
                    response = requests.get(
                        f"http://localhost:{config['port']}/health",
                        timeout=3
                    )
                    if response.status_code == 200:
                        self.module_metrics[module_id].status = "online"
                    else:
                        self.module_metrics[module_id].status = "degraded"
                    self.module_metrics[module_id].last_health_check = datetime.now()
                except Exception:
                    self.module_metrics[module_id].status = "offline"
    
    def is_mathematical_request(self, message: str) -> bool:
        """Проверка, является ли запрос математическим"""
        if not message or not isinstance(message, str):
            return False
        
        message_lower = message.lower().strip()
        
        # Простые математические операции
        import re
        math_patterns = [
            r'^\d+\s*[+\-*/]\s*\d+.*[=?]?$',  # 2+4, 3-1, 5*2, 8/2
            r'^\d+\s*[+\-*/]\s*\d+$',          # 2+4, 3-1 (без знака равенства)
            r'\d+\s*[+\-*/]\s*\d+',            # в тексте
            r'сколько.*\d+',                    # сколько яблок, сколько деревьев
            r'вычисли\s+\d+',                   # вычисли 2+3
            r'реши\s+\d+',                      # реши 5-2
            r'найди\s+\d+',                     # найди 3*4
            r'яблок.*столе.*осталось',          # задача про яблоки
            r'деревьев.*яблон.*груш',           # задача про деревья
            r'скорость.*путь.*время',           # физические задачи
            r'угол.*градус.*смежн',            # геометрические задачи
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, message_lower):
                return True
        
        # Ключевые слова
        math_keywords = [
            'сколько', 'вычисли', 'найди', 'реши', 'задача',
            'скорость', 'время', 'расстояние', 'путь',
            'угол', 'градус', 'смежные', 'сумма',
            'деревьев', 'яблон', 'груш', 'слив',
            'м/с', 'км/ч', '°', '+', '-', '*', '/', '=',
            'акула', 'преодолевает', 'длиной'
        ]
        
        for keyword in math_keywords:
            if keyword in message_lower:
                return True
        
        return False
    
    def analyze_request_category(self, message: str) -> str:
        """Анализ категории запроса с приоритетом математики"""
        
        # ПРИОРИТЕТ 1: Математические запросы
        if self.is_mathematical_request(message):
            self.logger.info(f"🧮 Обнаружен математический запрос: {message[:50]}...")
            return "mathematics"
        
        # ПРИОРИТЕТ 2: Специализированные домены
        message_lower = message.lower()
        
        # Контроллеры и PMAC
        controller_keywords = ['pmac', 'контроллер', 'ось', 'движение', 'траектория']
        if any(keyword in message_lower for keyword in controller_keywords):
            return "controllers"
        
        # Электротехника
        electrical_keywords = ['ом', 'вольт', 'ампер', 'сопротивление', 'ток', 'напряжение']
        if any(keyword in message_lower for keyword in electrical_keywords):
            return "electrical"
        
        # Радиомеханика
        radio_keywords = ['антенна', 'частота', 'радио', 'сигнал', 'передатчик']
        if any(keyword in message_lower for keyword in radio_keywords):
            return "radiomechanics"
        
        # Документы
        doc_keywords = ['документ', 'файл', 'загрузить', 'сохранить']
        if any(keyword in message_lower for keyword in doc_keywords):
            return "documents"
        
        # По умолчанию - общие ответы
        return "general"
    
    def route_request(self, message: str) -> Dict:
        """Маршрутизация запроса с приоритетом математики"""
        start_time = time.time()
        
        try:
            # Определяем категорию запроса
            category = self.analyze_request_category(message)
            self.logger.info(f"📡 Направляю к модулю: {self.modules[category]['name']}")
            
            # Обработка математических запросов
            if category == "mathematics":
                return self._handle_mathematical_request(message, start_time)
            
            # Обработка других запросов
            return self._handle_remote_request(message, category, start_time)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка маршрутизации: {e}")
            return {
                "success": False,
                "error": f"Ошибка маршрутизации: {str(e)}",
                "provider": "Intelligent Dispatcher",
                "category": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_mathematical_request(self, message: str, start_time: float) -> Dict:
        """Обработка математических запросов"""
        
        if not self.math_handler:
            return {
                "success": False,
                "error": "Математический решатель недоступен",
                "provider": "Mathematical Solver",
                "category": "mathematics",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Используем интегрированный решатель если доступен
            if hasattr(self.math_handler, 'process_request'):
                result = self.math_handler.process_request(message)
                
                processing_time = (time.time() - start_time) * 1000
                
                if result.get("success"):
                    solution_data = result.get("solution_data", {})
                    return {
                        "success": True,
                        "response": solution_data.get("final_answer", "Не удалось получить ответ"),
                        "provider": "Mathematical Solver (Integrated)",
                        "category": "mathematics",
                        "confidence": solution_data.get("confidence", 0.0),
                        "explanation": solution_data.get("explanation", ""),
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error_message", "Неизвестная ошибка"),
                        "provider": "Mathematical Solver (Integrated)",
                        "category": "mathematics",
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Fallback к простому решателю
            elif hasattr(self.math_handler, 'solve_problem'):
                solution = self.math_handler.solve_problem(message)
                processing_time = (time.time() - start_time) * 1000
                
                if solution:
                    return {
                        "success": True,
                        "response": solution.final_answer,
                        "provider": "Mathematical Solver (Simple)",
                        "category": "mathematics",
                        "confidence": solution.confidence,
                        "explanation": solution.explanation,
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "Не удалось решить задачу",
                        "provider": "Mathematical Solver (Simple)",
                        "category": "mathematics",
                        "processing_time_ms": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
            
            else:
                return {
                    "success": False,
                    "error": "Неизвестный тип математического решателя",
                    "provider": "Mathematical Solver",
                    "category": "mathematics",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Ошибка решения математической задачи: {e}")
            processing_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": f"Ошибка обработки: {str(e)}",
                "provider": "Mathematical Solver",
                "category": "mathematics",
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_remote_request(self, message: str, category: str, start_time: float) -> Dict:
        """Обработка удаленных запросов"""
        
        config = self.modules[category]
        port = config['port']
        
        if port is None:
            return {
                "success": False,
                "error": f"Модуль {config['name']} недоступен",
                "provider": config['name'],
                "category": category,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Проверяем доступность модуля
            if self.module_metrics[category].status != "online":
                return {
                    "success": False,
                    "error": f"Модуль {config['name']} (порт {port}) недоступен",
                    "provider": config['name'],
                    "category": category,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Отправляем запрос к модулю
            response = requests.post(
                f"http://localhost:{port}/api/chat",
                json={"message": message},
                timeout=10
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", "Ответ получен"),
                    "provider": config['name'],
                    "category": category,
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Ошибка модуля {config['name']}: {response.status_code}",
                    "provider": config['name'],
                    "category": category,
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": f"Ошибка подключения к {config['name']}: {str(e)}",
                "provider": config['name'],
                "category": category,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict:
        """Получение статуса системы"""
        with self.lock:
            status = {
                "timestamp": datetime.now().isoformat(),
                "modules": {},
                "mathematical_solver": self.math_handler is not None
            }
            
            for module_id, metrics in self.module_metrics.items():
                status["modules"][module_id] = {
                    "name": metrics.module_name,
                    "port": metrics.port,
                    "status": metrics.status,
                    "request_count": metrics.request_count,
                    "avg_response_time": metrics.avg_response_time,
                    "error_count": metrics.error_count,
                    "last_health_check": metrics.last_health_check.isoformat() if metrics.last_health_check else None
                }
            
            return status

# Глобальный экземпляр диспетчера
dispatcher = IntelligentDispatcherFixed()

def route_request(message: str) -> Dict:
    """Глобальная функция маршрутизации"""
    return dispatcher.route_request(message)

def get_system_status() -> Dict:
    """Глобальная функция получения статуса"""
    return dispatcher.get_system_status()

if __name__ == "__main__":
    # Тестирование диспетчера
    test_messages = [
        "2+4",
        "Сколько будет 3*5?",
        "2 яблока на столе одно укатилось, сколько осталось",
        "Как настроить PMAC контроллер?",
        "Расчет сопротивления по закону Ома"
    ]
    
    for message in test_messages:
        print(f"\n🔍 Тестирование: {message}")
        result = route_request(message)
        print(f"📡 Категория: {result.get('category', 'unknown')}")
        print(f"✅ Успех: {result.get('success', False)}")
        if result.get('success'):
            print(f"💬 Ответ: {result.get('response', 'Нет ответа')}")
        else:
            print(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")


















