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

# --- ИЗМЕНЕНИЕ 1: ДОБАВЛЕН ИМПОРТ НЕЙРОСЕТИ ---
try:
    from neural_rubin import get_neural_rubin
    NEURAL_NETWORK_AVAILABLE = True
    logging.info("🧠 Neural Rubin AI (нейросеть) успешно импортирован.")
except ImportError as e:
    NEURAL_NETWORK_AVAILABLE = False
    logging.warning(f"⚠️ Не удалось импортировать нейросеть: {e}. Диспетчер будет работать в режиме поиска по ключевым словам.")


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
        
        self._initialize_handlers()
        
        self.module_metrics: Dict[str, ModuleMetrics] = {}
        self.request_history: deque = deque(maxlen=1000)
        self.request_table: Dict[str, int] = defaultdict(int)
        
        self.load_balanced = True
        self.health_check_interval = 5
        self.performance_window = 300
        
        self.lock = threading.Lock()
        
        self._initialize_metrics()
        self._start_monitoring()
    
    def _initialize_handlers(self):
        self.math_handler = None
        self.programming_handler = None
        self.electrical_handler = None
        self.enhanced_categorizer = None
        self.logger.info("Обработчики инициализированы (пусто).")

    def _initialize_metrics(self):
        for module_id, config in self.modules.items():
            self.module_metrics[module_id] = ModuleMetrics(
                module_name=config['name'],
                port=config['port']
            )
    
    def _start_monitoring(self):
        # Мониторинг в этом примере отключен для простоты
        self.logger.info("Мониторинг модулей в данной версии отключен.")

    # --- ИЗМЕНЕНИЕ 2: ЛОГИКА КАТЕГОРИЗАЦИИ ЗАМЕНЕНА НА НЕЙРОСЕТЕВУЮ ---
    def analyze_request_category(self, message: str) -> str:
        """Категоризация сообщения с помощью нейронной сети с fallback."""
        if not NEURAL_NETWORK_AVAILABLE:
            self.logger.warning("⚠️ Нейросеть недоступна, используется fallback на ключевые слова.")
            return self._fallback_keyword_categorization(message)

        try:
            neural_ai = get_neural_rubin()
            category, confidence = neural_ai.classify_question(message)
            self.logger.info(f"🧠 Нейросеть классифицировала: '{message[:30]}...' → {category} (уверенность: {confidence:.2f})")
            
            if confidence < 0.5: # Порог уверенности
                self.logger.warning(f"⚠️ Низкая уверенность нейросети ({confidence:.2f}), используется fallback.")
                return self._fallback_keyword_categorization(message)
            
            # Сопоставление категорий нейросети с модулями диспетчера
            neural_to_dispatcher_map = {
                'физика': 'general', # Модуля физики нет, отправляем в общий
                'наука': 'general',
                'математика': 'mathematics',
                'электротехника': 'electrical',
                'программирование': 'programming',
                'контроллеры': 'controllers',
                'общие_вопросы': 'general'
            }
            
            return neural_to_dispatcher_map.get(category, 'general')

        except Exception as e:
            self.logger.error(f"❌ Ошибка нейронной классификации: {e}. Используем fallback.")
            return self._fallback_keyword_categorization(message)

    def _fallback_keyword_categorization(self, message: str) -> str:
        """Резервная категоризация по ключевым словам."""
        message_lower = message.lower()
        categories = {
            'programming': ['c++', 'python', 'программирование', 'алгоритм', 'сортировка'],
            'electrical': ['защита', 'ток', 'напряжение', 'резистор', 'транзистор', 'диод'],
            'controllers': ['пид', 'регулятор', 'plc', 'контроллер', 'автоматизация'],
        }
        for category, keywords in categories.items():
            if any(keyword in message_lower for keyword in keywords):
                self.logger.info(f"📊 Fallback категоризация: '{message[:30]}...' → {category}")
                return category
        
        self.logger.info(f"❓ Неопределенная категория (Fallback): '{message[:30]}...' → general")
        return 'general'

    def route_request(self, message: str, request_data: Dict) -> Tuple[Optional[str], str]:
        """Маршрутизация запроса"""
        start_time = time.time()
        
        category = self.analyze_request_category(message)
        self.logger.info(f"Категория запроса: {category}")
        
        selected_module = category # В этой упрощенной версии категория = модуль
        
        if not selected_module or selected_module not in self.modules:
            self.logger.warning(f"Нет доступных модулей для категории {category}")
            return None, "no_available_modules"
        
        try:
            response = self._forward_request(selected_module, request_data)
            response_time = time.time() - start_time
            self.logger.info(f"Запрос успешно обработан модулем {selected_module} за {response_time:.3f}с")
            return response, selected_module
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса модулем {selected_module}: {e}")
            return None, "error"
    
    def _forward_request(self, module: str, request_data: Dict) -> Dict:
        """Пересылка запроса к выбранному модулю"""
        metrics = self.modules[module]
        endpoint = '/api/chat' # Предполагаем, что у всех целевых серверов есть этот endpoint
        # В реальной системе здесь была бы более сложная логика определения эндпоинта
        
        # Используем host.docker.internal для связи между контейнерами
        url = f"http://host.docker.internal:{metrics['port']}{endpoint}"
        
        payload = {'message': request_data.get('message', '')}
        
        self.logger.info(f"🌐 Отправляем запрос к модулю {module} на {url}")
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            self.logger.info(f"✅ Модуль {module} ответил успешно")
            return response.json()
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

# Код ниже (Flask app) остается для запуска и обработки HTTP запросов
# Он будет использовать обновленный класс IntelligentDispatcher

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
dispatcher = IntelligentDispatcher()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        response, module = dispatcher.route_request(message, data)
        
        if response:
            return jsonify({
                'success': True,
                'routed_to': module,
                'response': response,
                'neural_analysis': NEURAL_NETWORK_AVAILABLE
            })
        else:
            return jsonify({'success': False, 'error': 'Не удалось обработать запрос'}), 500
            
    except Exception as e:
        dispatcher.logger.error(f"❌ Ошибка в /api/chat: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'neural_network': NEURAL_NETWORK_AVAILABLE})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("🚀 Запуск ИСПРАВЛЕННОГО нейронного диспетчера...")
    app.run(host='0.0.0.0', port=8081, debug=False) # Используем порт 8081, как в отчете пользователя