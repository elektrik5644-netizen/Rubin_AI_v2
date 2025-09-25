#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеллектуальный диспетчер для Rubin AI v2.0
Адаптация принципов LocalAI для оптимальной маршрутизации запросов
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

class IntelligentDispatcher:
    """Интеллектуальный диспетчер для маршрутизации запросов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Конфигурация модулей
        self.modules = {
            'controllers': {'port': 9000, 'name': 'Контроллеры'},
            'electrical': {'port': 8087, 'name': 'Электротехника'},
            'mathematics': {'port': 8086, 'name': 'Математика'},
            'programming': {'port': 8088, 'name': 'Программирование'},
            'neuro': {'port': 8090, 'name': 'Нейронная сеть'},
            'advanced_math': {'port': 8100, 'name': 'Продвинутая математика'},
            'data_processing': {'port': 8101, 'name': 'Обработка данных'},
            'search_engine': {'port': 8102, 'name': 'Поисковая система'},
            'system_utils': {'port': 8103, 'name': 'Системные утилиты'},
            'gai_server': {'port': 8104, 'name': 'GAI сервер'},
            'ethical_core': {'port': 8105, 'name': 'Этическое ядро'},
            'general': {'port': 8085, 'name': 'Общие ответы'}
        }
        
        # Инициализация обработчиков
        self.math_handler = None
        self.programming_handler = None
        self.electrical_handler = None
        self.enhanced_categorizer = None
        
        self._initialize_handlers()
        
        # Метрики модулей
        self.module_metrics: Dict[str, ModuleMetrics] = {
            "general": ModuleMetrics(module_name="Simple Chat", port=8085),
            "mathematics": ModuleMetrics(module_name="Math Server", port=8086),
            "programming": ModuleMetrics(module_name="Programming", port=8088),
            "electrical": ModuleMetrics(module_name="Electrical", port=8087),
            "neuro": ModuleMetrics(module_name="Neuro", port=8090),
            "advanced_math": ModuleMetrics(module_name="Advanced Math", port=8100),
            "data_processing": ModuleMetrics(module_name="Data Processing", port=8101),
            "search_engine": ModuleMetrics(module_name="Search Engine", port=8102),
            "system_utils": ModuleMetrics(module_name="System Utils", port=8103),
            "gai_server": ModuleMetrics(module_name="GAI Server", port=8104),
            "ethical_core": ModuleMetrics(module_name="Ethical Core", port=8105),
            "controllers": ModuleMetrics(module_name="Controllers", port=9000)
        }
        self.request_history: deque = deque(maxlen=1000)
        self.request_table: Dict[str, int] = defaultdict(int)
        
        # Настройки
        self.load_balanced = True
        self.health_check_interval = 5  # Уменьшаем для более быстрого обнаружения
        self.performance_window = 300  # 5 минут
        
        # Потокобезопасность
        self.lock = threading.Lock()
        
        # Инициализация метрик
        self._initialize_metrics()
        
        # Запуск мониторинга
        self._start_monitoring()
    
    def _initialize_handlers(self):
        """Инициализация всех обработчиков"""
        # Математический обработчик
        try:
            from mathematical_solver.request_handler import MathematicalRequestHandler
            self.math_handler = MathematicalRequestHandler()
            self.logger.info("✅ Математический обработчик инициализирован")
        except ImportError as e:
            self.logger.warning(f"⚠️ Не удалось инициализировать математический обработчик: {e}")
            self.math_handler = None
        
        # Улучшенный категоризатор
        try:
            from enhanced_request_categorizer import get_enhanced_categorizer
            self.enhanced_categorizer = get_enhanced_categorizer()
            self.logger.info("✅ Улучшенный категоризатор инициализирован")
        except ImportError as e:
            self.logger.warning(f"⚠️ Не удалось инициализировать улучшенный категоризатор: {e}")
            self.enhanced_categorizer = None
        
        # Обработчик программирования
        try:
            from programming_knowledge_handler import get_programming_handler
            self.programming_handler = get_programming_handler()
            self.logger.info("✅ Обработчик программирования инициализирован")
        except ImportError as e:
            self.logger.warning(f"⚠️ Не удалось инициализировать обработчик программирования: {e}")
            self.programming_handler = None
        
        # Обработчик электротехники
        try:
            from electrical_knowledge_handler import get_electrical_handler
            self.electrical_handler = get_electrical_handler()
            self.logger.info("✅ Обработчик электротехники инициализирован")
        except ImportError as e:
            self.logger.warning(f"⚠️ Не удалось инициализировать обработчик электротехники: {e}")
            self.electrical_handler = None
    
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
                    self._update_module_health()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Ошибка мониторинга: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("Мониторинг модулей запущен")
    
    def _update_module_health(self):
        """Обновление состояния здоровья модулей"""
        with self.lock:
            for module_id, metrics in self.module_metrics.items():
                # Если порт None, это локальный модуль, который всегда считается онлайн
                if metrics.port is None:
                    metrics.last_health_check = datetime.now()
                    metrics.status = 'online'
                    metrics.avg_response_time = 0.0 # Локальные модули не имеют времени ответа через сеть
                    self.logger.debug(f"Локальный модуль {module_id} всегда онлайн")
                    continue

                try:
                    # Проверка здоровья модуля
                    health_data = self._check_module_health(module_id)
                    
                    # Обновление метрик
                    metrics.last_health_check = datetime.now()
                    metrics.status = health_data['status']
                    
                    if health_data['status'] == 'online':
                        metrics.avg_response_time = health_data.get('response_time', 0.0)
                    else:
                        metrics.error_count += 1
                    
                    self.logger.debug(f"Модуль {module_id}: {metrics.status}, время ответа: {metrics.avg_response_time:.3f}с")
                    
                except Exception as e:
                    self.logger.error(f"Ошибка проверки модуля {module_id}: {e}")
                    metrics.status = 'offline'
                    metrics.error_count += 1
    
    def _check_module_health(self, module_id: str) -> Dict:
        """Проверка здоровья конкретного модуля"""
        metrics = self.module_metrics[module_id]
        
        # Определяем правильный endpoint для каждого модуля
        endpoints = {
            'general': '/api/health',
            'mathematics': '/health',
            'programming': '/api/health',
            'electrical': '/api/health',
            'neuro': '/api/health',
            'advanced_math': '/api/math/health',
            'data_processing': '/api/data/health',
            'search_engine': '/api/search/health',
            'system_utils': '/api/system/health',
            'gai_server': '/api/gai/health',
            'ethical_core': '/api/ethical/health',
            'controllers': '/api/health'
        }
        
        endpoint = endpoints.get(module_id, '/api/health')
        url = f"http://localhost:{metrics.port}{endpoint}"
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=3)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.logger.debug(f"✅ Модуль {module_id} на порту {metrics.port} ответил успешно. Статус: {response.status_code}")
                return {
                    'status': 'online',
                    'response_time': response_time,
                    'data': response.json() if response.content else {}
                }
            else:
                self.logger.warning(f"⚠️ Модуль {module_id} на порту {metrics.port} ответил с ошибкой: {response.status_code}")
                return {'status': 'degraded', 'response_time': response_time}
                
        except requests.exceptions.Timeout:
            self.logger.error(f"❌ Таймаут при проверке модуля {module_id} на порту {metrics.port}")
            return {'status': 'offline', 'response_time': 3.0}
        except requests.exceptions.ConnectionError as ce:
            self.logger.error(f"❌ Ошибка соединения с модулем {module_id} на порту {metrics.port}: {ce}")
            return {'status': 'offline', 'response_time': 0.0}
        except Exception as e:
            self.logger.error(f"❌ Неожиданная ошибка при проверке модуля {module_id} на порту {metrics.port}: {e}")
            return {'status': 'offline', 'response_time': 0.0}
    
    def analyze_request_category(self, message: str) -> str:
        """Улучшенный анализ категории запроса"""
        # Используем улучшенный категоризатор если доступен
        if self.enhanced_categorizer:
            try:
                category = self.enhanced_categorizer.categorize(message)
                confidence = self.enhanced_categorizer.get_confidence(message, category)
                
                self.logger.info(f"🎯 Улучшенная категоризация: '{message[:50]}...' → {category} (уверенность: {confidence:.2f})")
                
                # Если уверенность высокая, используем результат
                if confidence >= 0.3:
                    return category
                else:
                    self.logger.warning(f"⚠️ Низкая уверенность категоризации ({confidence:.2f}), используем fallback")
            except Exception as e:
                self.logger.error(f"❌ Ошибка улучшенного категоризатора: {e}")
        
        # Fallback: проверяем на математические запросы
        if self.math_handler:
            try:
                if self.math_handler.detector.is_mathematical_request(message):
                    math_category = self.math_handler.detector.detect_math_category(message)
                    if math_category:
                        self.logger.info(f"🧮 Обнаружен математический запрос: {math_category}")
                        return f"mathematics_{math_category}"
                    else:
                        self.logger.info("🧮 Обнаружен общий математический запрос")
                        return "mathematics_general"
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка определения математической категории: {e}")
        
        # Fallback: старая логика категоризации
        message_lower = message.lower()
        categories = {
            'programming': ['c++', 'python', 'программирование', 'сравни', 'compare', 'язык программирования', 'алгоритм', 'сортировка', 'массив', 'функции', 'продвинутые', 'специфические'],
            'electrical': ['защита', 'короткое замыкание', 'цепи', 'электрические', 'электричество', 'схема', 'ток', 'напряжение', 'резистор', 'транзистор', 'диод', 'контактор', 'реле', 'мощность'],
            'controllers': ['пид', 'регулятор', 'plc', 'контроллер', 'автоматизация', 'логика'],
            'radiomechanics': ['антенна', 'сигнал', 'радио', 'модуляция', 'частота', 'передатчик', 'приемник']
        }
        
        # Подсчет совпадений
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            scores[category] = score
        
        # Возврат категории с наибольшим score
        if scores and max(scores.values()) > 0:
            best_category = max(scores, key=scores.get)
            self.logger.info(f"📊 Fallback категоризация: '{message[:50]}...' → {best_category}")
            return best_category
        
        self.logger.info(f"❓ Неопределенная категория: '{message[:50]}...' → general")
        return 'general'
    
    def get_available_modules(self, category: str) -> List[str]:
        """Получение списка доступных модулей для категории"""
        with self.lock:
            available = []
            
            # Специализированные модули для категории
            if category in self.modules:
                if self.module_metrics[category].status == 'online':
                    available.append(category)
            
            # Fallback на общие ответы ТОЛЬКО если нет специализированных модулей
            if not available and 'general' in self.modules and self.module_metrics['general'].status == 'online':
                available.append('general')
            
            return available
    
    def select_least_used_module(self, category: str) -> Optional[str]:
        """Выбор наименее загруженного модуля"""
        available_modules = self.get_available_modules(category)
        
        if not available_modules:
            return None
        
        with self.lock:
            # Поиск модуля с минимальным количеством запросов
            min_requests = min(self.request_table.get(module, 0) for module in available_modules)
            least_used = [m for m in available_modules if self.request_table.get(m, 0) == min_requests]
            
            # Если несколько модулей с одинаковой нагрузкой, выбираем случайный
            return random.choice(least_used) if least_used else None
    
    def select_optimal_module(self, category: str) -> Optional[str]:
        """Выбор оптимального модуля на основе метрик производительности"""
        available_modules = self.get_available_modules(category)
        
        if not available_modules:
            return None
        
        with self.lock:
            # Вычисление score для каждого модуля
            scored_modules = []
            
            for module in available_modules:
                metrics = self.module_metrics[module]
                
                # Базовый score на основе количества запросов
                load_score = self.request_table.get(module, 0)
                
                # Score на основе времени ответа (чем меньше, тем лучше)
                response_score = metrics.avg_response_time * 10
                
                # Score на основе ошибок (чем больше ошибок, тем хуже)
                error_score = metrics.error_count * 5
                
                # Общий score (чем меньше, тем лучше)
                total_score = load_score + response_score + error_score
                
                scored_modules.append((module, total_score, metrics))
            
            # Сортировка по score
            scored_modules.sort(key=lambda x: x[1])
            
            # Логирование выбора
            if scored_modules:
                best_module, best_score, best_metrics = scored_modules[0]
                self.logger.debug(f"Выбран модуль {best_module} (score: {best_score:.2f}, "
                                f"запросов: {self.request_table.get(best_module, 0)}, "
                                f"время ответа: {best_metrics.avg_response_time:.3f}с)")
                
                return best_module
        
        return None
    
    def random_module(self, category: str) -> Optional[str]:
        """Случайный выбор модуля"""
        available_modules = self.get_available_modules(category)
        return random.choice(available_modules) if available_modules else None
    
    def record_request(self, module: str, response_time: float, success: bool, category: str):
        """Запись метрик запроса"""
        with self.lock:
            # Обновление счетчика запросов
            self.request_table[module] += 1
            
            # Обновление метрик модуля
            if module in self.module_metrics:
                metrics = self.module_metrics[module]
                metrics.request_count += 1
                
                # Обновление среднего времени ответа
                if success:
                    if metrics.avg_response_time == 0:
                        metrics.avg_response_time = response_time
                    else:
                        # Экспоненциальное скользящее среднее
                        alpha = 0.1
                        metrics.avg_response_time = (alpha * response_time + 
                                                   (1 - alpha) * metrics.avg_response_time)
                else:
                    metrics.error_count += 1
            
            # Добавление в историю
            request_metric = RequestMetrics(
                timestamp=datetime.now(),
                module=module,
                response_time=response_time,
                success=success,
                category=category
            )
            self.request_history.append(request_metric)
    
    def route_request(self, message: str, request_data: Dict) -> Tuple[Optional[str], str]:
        """Маршрутизация запроса"""
        start_time = time.time()
        
        # Анализ категории
        category = self.analyze_request_category(message)
        self.logger.info(f"Категория запроса: {category}")
        
        # Выбор модуля
        if self.load_balanced:
            selected_module = self.select_optimal_module(category)
        else:
            selected_module = self.random_module(category)
        
        if not selected_module:
            self.logger.warning(f"Нет доступных модулей для категории {category}")
            return None, "no_available_modules"
        
        # Маршрутизация запроса
        try:
            response = self._forward_request(selected_module, request_data)
            response_time = time.time() - start_time
            
            # Запись метрик
            self.record_request(selected_module, response_time, True, category)
            
            self.logger.info(f"Запрос успешно обработан модулем {selected_module} за {response_time:.3f}с")
            return response, selected_module
            
        except Exception as e:
            response_time = time.time() - start_time
            self.record_request(selected_module, response_time, False, category)
            
            self.logger.error(f"Ошибка обработки запроса модулем {selected_module}: {e}")
            return None, "error"
    
    def _forward_request(self, module: str, request_data: Dict) -> Dict:
        """Пересылка запроса к выбранному модулю с fallback механизмами"""
        
        # Интегрированные модули (обрабатываются локально)
        if module == 'programming':
            return self._handle_programming_request(request_data)
        elif module == 'electrical':
            # Сначала пробуем внешний модуль, потом fallback на интегрированный
            if module in self.module_metrics:
                port = self.module_metrics[module].port
                if port and self._check_module_availability(port):
                    try:
                        return self._forward_to_external_module(module, request_data)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Внешний модуль {module} недоступен: {e}, используем интегрированный")
            
            # Fallback на интегрированный обработчик
            return self._handle_electrical_request(request_data)
        elif module.startswith('mathematics'):
            return self._handle_mathematical_request(request_data)
        elif module == 'general':
            return self._handle_general_request(request_data)
        else:
            # Внешние модули
            return self._forward_to_external_module(module, request_data)
    
    def _forward_to_external_module(self, module: str, request_data: Dict) -> Dict:
        """Пересылка запроса к внешнему модулю"""
        metrics = self.module_metrics[module]
        
        # Специализированные модули - используем правильные endpoint'ы
        endpoint_map = {
            'controllers': f'/api/controllers/topic/general',
            'electrical': f'/api/electrical/explain',
            'radiomechanics': f'/api/radiomechanics/explain'
        }
        
        endpoint = endpoint_map.get(module, f'/api/{module}/explain')
        url = f"http://localhost:{metrics.port}{endpoint}"
        
        # Подготовка данных запроса в зависимости от модуля
        if module == 'controllers':
            request_payload = {'concept': request_data.get('message', '')}
        elif module in ['electrical', 'radiomechanics']:
            request_payload = {'concept': request_data.get('message', '')}
        else:
            request_payload = {'query': request_data.get('message', '')}
        
        self.logger.info(f"🌐 Отправляем запрос к внешнему модулю {module} на {url}")
        
        response = requests.post(
            url,
            json=request_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            self.logger.info(f"✅ Внешний модуль {module} ответил успешно")
            return response.json()
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    def _handle_mathematical_request(self, request_data: Dict) -> Dict:
        """Обработка математических запросов"""
        message = request_data.get('message', '')
        
        if self.math_handler:
            try:
                self.logger.info(f"Обрабатываем математический запрос: {message[:50]}...")
                response = self.math_handler.handle_request(message)
                
                if response.get('solution_data'):
                    # Форматируем математический ответ
                    solution = response['solution_data']
                    formatted_response = f"""🧮 **Математическое решение:**

**Ответ:** {solution.get('final_answer', 'Не найден')}

**Пошаговое решение:**
"""
                    for i, step in enumerate(solution.get('steps', []), 1):
                        formatted_response += f"{i}. {step}\n"
                    
                    if solution.get('explanation'):
                        formatted_response += f"\n💡 **Объяснение:** {solution['explanation']}"
                    
                    formatted_response += f"\n📊 **Уверенность:** {solution.get('confidence', 0):.1%}"
                    
                    return {
                        'response': formatted_response,
                        'provider': 'Mathematical Solver',
                        'category': 'mathematics',
                        'solution_data': solution,
                        'success': True
                    }
                else:
                    return response
                    
            except Exception as e:
                self.logger.error(f"Ошибка математического обработчика: {e}")
                return {
                    'response': f'Произошла ошибка при решении математической задачи: {str(e)}',
                    'provider': 'Mathematical Solver',
                    'category': 'mathematics',
                    'success': False,
                    'error': str(e)
                }
        else:
            return {
                'response': 'Математический решатель недоступен. Попробуйте переформулировать вопрос.',
                'provider': 'System',
                'category': 'mathematics',
                'success': False
            }

    def _handle_programming_request(self, request_data: Dict) -> Dict:
        """Обработка программных запросов"""
        message = request_data.get('message', '')
        
        if self.programming_handler:
            try:
                self.logger.info(f"💻 Обрабатываем программный запрос: {message[:50]}...")
                response = self.programming_handler.handle_request(message)
                self.logger.info(f"✅ Программный запрос обработан успешно")
                return response
            except Exception as e:
                self.logger.error(f"❌ Ошибка обработчика программирования: {e}")
                return {
                    'response': f'Произошла ошибка при обработке программного вопроса: {str(e)}',
                    'provider': 'Programming Handler',
                    'category': 'programming',
                    'success': False,
                    'error': str(e)
                }
        else:
            return {
                'response': 'Обработчик программирования недоступен. Попробуйте переформулировать вопрос.',
                'provider': 'System',
                'category': 'programming',
                'success': False
            }

    def _handle_electrical_request(self, request_data: Dict) -> Dict:
        """Обработка электротехнических запросов"""
        message = request_data.get('message', '')
        
        if self.electrical_handler:
            try:
                self.logger.info(f"⚡ Обрабатываем электротехнический запрос: {message[:50]}...")
                response = self.electrical_handler.handle_request(message)
                self.logger.info(f"✅ Электротехнический запрос обработан успешно")
                return response
            except Exception as e:
                self.logger.error(f"❌ Ошибка обработчика электротехники: {e}")
                return {
                    'response': f'Произошла ошибка при обработке электротехнического вопроса: {str(e)}',
                    'provider': 'Electrical Handler',
                    'category': 'electrical',
                    'success': False,
                    'error': str(e)
                }
        else:
            return {
                'response': 'Обработчик электротехники недоступен. Попробуйте переформулировать вопрос.',
                'provider': 'System',
                'category': 'electrical',
                'success': False
            }

    def _check_module_availability(self, port: int) -> bool:
        """Проверка доступности модуля по порту"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # Таймаут 2 секунды
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.debug(f"Ошибка проверки порта {port}: {e}")
            return False

    def _handle_general_request(self, request_data: Dict) -> Dict:
        """Обработка общих запросов с использованием гибридного поиска и локального LLM"""
        message = request_data.get('message', '')
        message_lower = message.lower()
        
        # Проверяем на общие вопросы и приветствия - НЕ ищем в базе данных
        if any(phrase in message_lower for phrase in [
            "какой хороший день", "хороший день", "как дела", "как поживаешь",
            "что нового", "как настроение", "как жизнь", "привет", "hello", "hi"
        ]):
            self.logger.info("Обнаружен общий вопрос/приветствие, используем fallback ответ")
            # Импортируем локальный LLM провайдер для fallback
            try:
                from providers.local_llm_provider import LocalLLMProvider
                local_llm = LocalLLMProvider()
                local_llm.initialize()
                
                # Генерируем fallback ответ
                llm_response = local_llm.get_response(message, {'task_type': 'general_chat', 'documents': []})
                
                if llm_response.get('success', True):
                    return {
                        'response': llm_response['content'],
                        'provider': 'Local LLM (Fallback)',
                        'category': 'general',
                        'search_results_count': 0
                    }
            except Exception as e:
                self.logger.error(f"Ошибка fallback ответа: {e}")
        
        # Импортируем локальный LLM провайдер
        try:
            from providers.local_llm_provider import LocalLLMProvider
            local_llm = LocalLLMProvider()
            local_llm.initialize()
        except ImportError as e:
            self.logger.error(f"Не удалось импортировать LocalLLMProvider: {e}")
            local_llm = None
        
        # Импортируем процессор данных
        try:
            from data_processor import get_data_processor
            data_processor = get_data_processor()
        except ImportError as e:
            self.logger.error(f"Не удалось импортировать DataProcessor: {e}")
            data_processor = None
        
        # Простые общие ответы
        general_responses = {
            'привет': 'Привет! Я Rubin AI, ваш помощник по техническим вопросам.',
            'как дела': 'У меня все хорошо! Готов помочь с техническими вопросами.',
            'спасибо': 'Пожалуйста! Рад был помочь.',
            'помощь': 'Я могу помочь с вопросами по контроллерам, электротехнике, радиомеханике и поиску документов.'
        }
        
        message_lower = message.lower()
        for key, response in general_responses.items():
            if key in message_lower:
                return {
                    'response': response,
                    'category': 'general',
                    'provider': 'General Assistant'
                }
        
        # Используем последовательный поиск для общих запросов
        try:
            # Импортируем последовательный поиск
            from sequential_search_engine import SequentialSearchEngine
            
            # Создаем экземпляр последовательного поиска
            sequential_search_engine = SequentialSearchEngine()
            
            self.logger.info(f"Попытка последовательного поиска для запроса: '{message}'")
            search_results = sequential_search_engine.search(message, limit=3)
            self.logger.info(f"Последовательный поиск вернул {len(search_results) if search_results else 0} результатов")
            
            if search_results and len(search_results) > 0:
                # Обрабатываем результаты поиска через процессор данных
                if data_processor:
                    try:
                        self.logger.info("🔄 Обработка результатов поиска через процессор данных")
                        processed_content = data_processor.process_search_results(search_results)
                        
                        # Проверяем качество обработанного контента
                        if processed_content.quality_score >= 0.3:
                            self.logger.info(f"✅ Контент прошел фильтрацию: качество {processed_content.quality_score:.2f}")
                            
                            # Используем локальный LLM для генерации ответа
                            if local_llm:
                                try:
                                    self.logger.info("Генерируем ответ с помощью локального LLM")
                                    context = {
                                        'task_type': 'general_chat',
                                        'documents': search_results,
                                        'processed_content': data_processor.prepare_for_llm(processed_content)
                                    }
                                    llm_response = local_llm.get_response(message, context)
                                    
                                    if llm_response.get('success', True):
                                        # Проверяем качество ответа от LLM
                                        response_validation = data_processor.validate_llm_response(llm_response['content'])
                                        
                                        if response_validation['valid']:
                                            self.logger.info("Локальный LLM успешно сгенерировал качественный ответ")
                                            return {
                                                'response': llm_response['content'],
                                                'provider': 'Local LLM + Data Processor + Sequential Search',
                                                'category': 'general',
                                                'search_results_count': len(search_results),
                                                'processed_sections': len(processed_content.valid_sections),
                                                'quality_score': processed_content.quality_score,
                                                'response_quality': response_validation['quality_score']
                                            }
                                        else:
                                            self.logger.warning(f"Ответ LLM не прошел валидацию: {response_validation['reason']}")
                                    else:
                                        self.logger.warning(f"Ошибка локального LLM: {llm_response.get('error', 'Unknown error')}")
                                except Exception as e:
                                    self.logger.error(f"Ошибка локального LLM: {e}")
                        else:
                            self.logger.warning(f"Контент не прошел фильтрацию: качество {processed_content.quality_score:.2f}")
                    except Exception as e:
                        self.logger.error(f"Ошибка процессора данных: {e}")
                
                # Fallback: используем локальный LLM без обработки
                if local_llm:
                    try:
                        self.logger.info("Генерируем ответ с помощью локального LLM (fallback)")
                        context = {
                            'task_type': 'general_chat',
                            'documents': search_results
                        }
                        llm_response = local_llm.get_response(message, context)
                        
                        if llm_response.get('success', True):
                            self.logger.info("Локальный LLM успешно сгенерировал ответ")
                            return {
                                'response': llm_response['content'],
                                'provider': 'Local LLM + Sequential Search',
                                'category': 'general',
                                'search_results_count': len(search_results),
                                'documents_used': len(search_results)
                            }
                        else:
                            self.logger.warning(f"Ошибка локального LLM: {llm_response.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.logger.error(f"Ошибка локального LLM: {e}")
                
                # Fallback: показываем найденные документы как раньше
                self.logger.info("Используем fallback форматирование документов")
                response_text = "**Найденная информация:**\n\n"
                
                for i, result in enumerate(search_results[:2], 1):
                    response_text += f"**{i}. {result.get('file_name', result.get('title', 'Документ'))}**\n"
                    content = result.get('content_preview', result.get('content', ''))
                    # Показываем больше текста, но не более 1000 символов
                    if len(content) > 1000:
                        response_text += f"{content[:1000]}...\n\n"
                    else:
                        response_text += f"{content}\n\n"
                
                response_text += "\n*Ответ основан на документах из базы знаний Rubin AI*"
                
                self.logger.info(f"Последовательный поиск нашел {len(search_results)} результатов, возвращаем ответ")
                
                return {
                    'response': response_text,
                    'provider': 'Sequential Search',
                    'category': 'general',
                    'search_results_count': len(search_results)
                }
            else:
                self.logger.warning("Последовательный поиск не нашел результатов")
                
        except Exception as e:
            self.logger.error(f"Ошибка последовательного поиска: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Используем локальный LLM для fallback ответа
        if local_llm:
            try:
                self.logger.info("Генерируем fallback ответ с помощью локального LLM")
                context = {
                    'task_type': 'general_chat',
                    'documents': []
                }
                llm_response = local_llm.get_response(message, context)
                
                if llm_response.get('success', True):
                    self.logger.info("Локальный LLM успешно сгенерировал fallback ответ")
                    return {
                        'response': llm_response['content'],
                        'provider': 'Local LLM',
                        'category': 'general',
                        'search_results_count': 0
                    }
            except Exception as e:
                self.logger.error(f"Ошибка локального LLM fallback: {e}")
        
        # Дефолтный ответ
        return {
            'response': 'Я понимаю ваш вопрос, но для более точного ответа уточните, пожалуйста, область (контроллеры, электротехника, радиомеханика).',
            'category': 'general',
            'provider': 'General Assistant'
        }
    
    def get_statistics(self) -> Dict:
        """Получение статистики диспетчера"""
        with self.lock:
            stats = {
                'total_requests': sum(self.request_table.values()),
                'module_stats': {},
                'recent_requests': len([r for r in self.request_history 
                                      if r.timestamp > datetime.now() - timedelta(minutes=5)]),
                'load_balanced': self.load_balanced
            }
            
            for module_id, metrics in self.module_metrics.items():
                stats['module_stats'][module_id] = {
                    'name': metrics.module_name,
                    'port': metrics.port,
                    'status': metrics.status,
                    'request_count': self.request_table.get(module_id, 0),
                    'avg_response_time': metrics.avg_response_time,
                    'error_count': metrics.error_count,
                    'last_health_check': metrics.last_health_check.isoformat() if metrics.last_health_check else None
                }
            
            return stats
    
    def set_load_balancing(self, enabled: bool):
        """Включение/отключение балансировки нагрузки"""
        self.load_balanced = enabled
        self.logger.info(f"Балансировка нагрузки: {'включена' if enabled else 'отключена'}")

# Глобальный экземпляр диспетчера
dispatcher = IntelligentDispatcher()

def get_intelligent_dispatcher() -> IntelligentDispatcher:
    """Получение глобального экземпляра диспетчера"""
    return dispatcher

if __name__ == "__main__":
    # Тестирование диспетчера
    logging.basicConfig(level=logging.INFO)
    
    test_requests = [
        "Как работает ПИД-регулятор?",
        "Объясни закон Ома",
        "Что такое антенна?",
        "Привет, как дела?",
        "Найди документы по автоматизации"
    ]
    
    for request in test_requests:
        print(f"\n--- Тест: {request} ---")
        response, module = dispatcher.route_request(request, {'message': request})
        print(f"Модуль: {module}")
        print(f"Ответ: {response}")
    
    # Статистика
    print(f"\n--- Статистика ---")
    stats = dispatcher.get_statistics()
    print(f"Всего запросов: {stats['total_requests']}")
    print(f"Недавних запросов (5 мин): {stats['recent_requests']}")
    
    for module_id, module_stats in stats['module_stats'].items():
        print(f"{module_stats['name']}: {module_stats['status']}, "
              f"запросов: {module_stats['request_count']}, "
              f"время ответа: {module_stats['avg_response_time']:.3f}с")
