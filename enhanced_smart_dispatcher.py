#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Enhanced Smart Dispatcher с нейронной сетью
Улучшенный маршрутизатор с интеграцией нейронной сети и отображением ошибок
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import logging
import json
import time
import os
from datetime import datetime
import threading
import queue

# Импорт Qdrant адаптера
try:
    from enhanced_qdrant_adapter import EnhancedQdrantAdapter
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Настройка детального логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_dispatcher.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Дополнительные логгеры для разных компонентов
neural_logger = logging.getLogger('neural_router')
routing_logger = logging.getLogger('routing')
kb_logger = logging.getLogger('knowledge_base')
error_logger = logging.getLogger('error_tracker')

app = Flask(__name__)
CORS(app)

# Очередь для ошибок
error_queue = queue.Queue()

# Конфигурация серверов
SERVERS = {
    "general": {
        "port": 8085,
        "endpoint": "/api/chat",
        "method": "POST",
        "keywords": ["привет", "как дела", "что нового", "общее", "помощь", "справка"],
        "priority": 1
    },
    "mathematics": {
        "port": 8086,
        "endpoint": "/health",
        "method": "GET",
        "keywords": ["математика", "математический", "вычисление", "формула", "уравнение", "интеграл", "производная"],
        "priority": 2
    },
    "electrical": {
        "port": 8087,
        "endpoint": "/api/electrical/solve",
        "method": "POST",
        "keywords": ["электротехника", "электричество", "ток", "напряжение", "мощность", "схема", "диод", "резистор", "конденсатор", "транзистор", "кирхгофа", "закон", "сервопривод"],
        "priority": 2
    },
    "programming": {
        "port": 8088,
        "endpoint": "/api/programming/explain",
        "method": "GET",
        "keywords": ["программирование", "код", "алгоритм", "python", "javascript", "java", "c++"],
        "priority": 2
    },
    "radiomechanics": {
        "port": 8089,
        "endpoint": "/api/chat",
        "method": "POST",
        "keywords": ["радиомеханика", "радио", "передатчик", "приемник", "антенна", "частота"],
        "priority": 2
    },
    # "neuro": {
    #     "port": 8090,
    #     "endpoint": "/api/neuro/health",
    #     "method": "GET",
    #     "keywords": ["нейросеть", "нейронная сеть", "машинное обучение", "искусственный интеллект", "ai"],
    #     "priority": 3
    # },
    # "controllers": {
    #     "port": 9000,
    #     "endpoint": "/api/controllers/topic/general",
    #     "method": "GET",
    #     "keywords": ["контроллер", "pmac", "plc", "сервопривод", "серводвигатель", "привод", "двигатель"],
    #     "priority": 2
    # },
    "plc_analysis": {
        "port": 8099,
        "endpoint": "/api/plc/health",
        "method": "GET",
        "keywords": ["plc анализ", "ladder logic", "программа plc", "отладка plc"],
        "priority": 2
    },
    # Отключенные модули - не существуют
    # "advanced_math": {
    #     "port": 8100,
    #     "endpoint": "/api/advanced_math/health",
    #     "method": "GET",
    #     "keywords": ["продвинутая математика", "сложные вычисления", "высшая математика"],
    #     "priority": 2
    # },
    # "data_processing": {
    #     "port": 8101,
    #     "endpoint": "/api/data_processing/health",
    #     "method": "GET",
    #     "keywords": ["обработка данных", "анализ данных", "статистика", "визуализация"],
    #     "priority": 2
    # },
    # "search_engine": {
    #     "port": 8102,
    #     "endpoint": "/api/search/health",
    #     "method": "GET",
    #     "keywords": ["поиск", "гибридный поиск", "семантический поиск"],
    #     "priority": 2
    # },
    # "system_utils": {
    #     "port": 8103,
    #     "endpoint": "/api/system/health",
    #     "method": "GET",
    #     "keywords": ["системные утилиты", "диагностика", "мониторинг", "оптимизация", "проблемы", "ошибки", "здоровье системы"],
    #     "priority": 1
    # },
    "gai_server": {
        "port": 8104,
        "endpoint": "/api/gai/health",
        "method": "GET",
        "keywords": ["генеративный ai", "создание контента", "генерация текста", "генерация кода"],
        "priority": 2
    },
    "unified_manager": {
        "port": 8084,
        "endpoint": "/api/system/health",
        "method": "GET",
        "keywords": ["управление системой", "менеджер", "администрирование"],
        "priority": 1
    },
    "ethical_core": {
        "port": 8105,
        "endpoint": "/api/ethical/health",
        "method": "GET",
        "keywords": ["этика", "безопасность", "этическое ядро"],
        "priority": 1
    }
}

class NeuralRouter:
    """🧠 Нейронный маршрутизатор для интеллектуальной категоризации"""
    
    def __init__(self):
        # Инициализация настоящей нейронной сети
        try:
            from neural_rubin import NeuralRubinAI
            self.neural_ai = NeuralRubinAI()
            self.neural_available = True
            neural_logger.info("✅ Настоящая нейронная сеть инициализирована")
        except Exception as e:
            neural_logger.warning(f"⚠️ Не удалось загрузить нейронную сеть: {e}")
            self.neural_ai = None
            self.neural_available = False
        
        # Инициализация Qdrant для векторного поиска
        if QDRANT_AVAILABLE:
            try:
                self.qdrant_adapter = EnhancedQdrantAdapter(use_embedded=False)
                self.qdrant_available = True
                neural_logger.info("✅ Qdrant адаптер инициализирован")
            except Exception as e:
                neural_logger.warning(f"⚠️ Не удалось инициализировать Qdrant: {e}")
                self.qdrant_adapter = None
                self.qdrant_available = False
        else:
            self.qdrant_adapter = None
            self.qdrant_available = False
        
        # Fallback база знаний для простой классификации
        self.knowledge_base = {
            "технические": ["pmac", "plc", "сервопривод", "электротехника", "радиомеханика", "контроллер", "транзистор", "диод", "резистор", "конденсатор", "закон кирхгофа", "закон ома"],
            "математические": ["математика", "вычисление", "формула", "уравнение", "интеграл", "производная", "2+2", "сложение", "вычитание", "умножение", "деление", "площадь", "объем"],
            "программные": ["программирование", "код", "алгоритм", "python", "javascript", "java", "c++", "напиши программу"],
            "системные": ["диагностика", "мониторинг", "оптимизация", "проблемы", "ошибки"],
            "ai_связанные": ["нейросеть", "машинное обучение", "искусственный интеллект", "ai"],
            "общие": ["привет", "как дела", "что нового", "помощь", "справка", "здравствуй", "добро пожаловать"]
        }
    
    def analyze_message(self, message):
        """Анализ сообщения с помощью настоящей нейронной сети и векторного поиска"""
        neural_logger.info(f"🧠 Анализ сообщения: '{message[:50]}...'")
        
        # Сначала пробуем векторный поиск в Qdrant
        if self.qdrant_available and self.qdrant_adapter:
            try:
                # Поиск похожих документов в базе знаний
                similar_docs = self.qdrant_adapter.search_similar(
                    collection_name="rubin_knowledge",
                    query_text=message,
                    limit=3,
                    score_threshold=0.7
                )
                
                if similar_docs:
                    neural_logger.info(f"🔍 Найдено {len(similar_docs)} похожих документов в Qdrant")
                    # Возвращаем категорию с высокой уверенностью
                    return "knowledge_base", 0.9
                    
            except Exception as e:
                neural_logger.warning(f"⚠️ Ошибка векторного поиска: {e}")
        
        # Пытаемся использовать настоящую нейронную сеть
        if self.neural_available and self.neural_ai:
            try:
                category, confidence = self.neural_ai.classify_question(message)
                neural_logger.info(f"  🎯 Нейронная классификация: {category} (уверенность: {confidence:.2f})")
                
                # Маппинг категорий нейросети на наши категории
                neural_to_our_mapping = {
                    'математика': 'математические',
                    'электротехника': 'технические', 
                    'программирование': 'программные',
                    'физика': 'технические',
                    'общие_вопросы': 'общие',
                    'техника': 'технические',
                    'наука': 'технические',
                    'другое': 'общие',
                    'time_series': 'математические',
                    'graph_analysis': 'математические',
                    'data_visualization': 'математические',
                    'formula_calculation': 'математические'
                }
                
                mapped_category = neural_to_our_mapping.get(category, 'общие')
                return mapped_category, confidence
                
            except Exception as e:
                neural_logger.warning(f"⚠️ Ошибка нейронной классификации: {e}, используем fallback")
        
        # Fallback: простая классификация по ключевым словам
        message_lower = message.lower()
        category_scores = {}
        for category, keywords in self.knowledge_base.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                category_scores[category] = score
                neural_logger.debug(f"  Категория '{category}': {score} совпадений")
        
        # Определение приоритетной категории
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category] / len(message_lower.split())
            neural_logger.info(f"  🎯 Fallback категория: {best_category} (уверенность: {confidence:.2f})")
            return best_category, confidence
        
        neural_logger.info("  🔄 Категория не определена, используем 'общие'")
        return "общие", 0.1
    
    def suggest_route(self, message):
        """Предложение маршрута на основе нейронного анализа"""
        category, confidence = self.analyze_message(message)
        neural_logger.info(f"🔀 Предложение маршрута для категории: {category}")
        
        # Маппинг категорий на серверы
        category_mapping = {
            "технические": ["electrical", "controllers", "radiomechanics", "plc_analysis"],
            "математические": ["mathematics", "advanced_math"],
            "программные": ["programming"],
            "системные": ["system_utils", "unified_manager"],
            "ai_связанные": ["neuro", "gai_server"],
            "общие": ["general"]
        }
        
        suggested_servers = category_mapping.get(category, ["general"])
        suggested_server = suggested_servers[0]
        neural_logger.info(f"  🎯 Предложенный сервер: {suggested_server} (уверенность: {confidence:.2f})")
        return suggested_server, confidence

class ErrorTracker:
    """📊 Трекер ошибок для мониторинга системы"""
    
    def __init__(self):
        self.errors = []
        self.server_status = {}
    
    def log_error(self, server_name, error_type, message, details=None):
        """Логирование ошибки"""
        error = {
            "timestamp": datetime.now().isoformat(),
            "server": server_name,
            "type": error_type,
            "message": message,
            "details": details
        }
        self.errors.append(error)
        error_queue.put(error)
        logger.error(f"❌ {server_name}: {error_type} - {message}")
    
    def update_server_status(self, server_name, status, response_time=None):
        """Обновление статуса сервера"""
        self.server_status[server_name] = {
            "status": status,
            "last_check": datetime.now().isoformat(),
            "response_time": response_time
        }
    
    def get_errors(self, limit=50):
        """Получение последних ошибок"""
        return self.errors[-limit:]
    
    def get_system_health(self):
        """Получение общего здоровья системы"""
        total_servers = len(SERVERS)
        healthy_servers = sum(1 for status in self.server_status.values() if status["status"] == "healthy")
        
        return {
            "total_servers": total_servers,
            "healthy_servers": healthy_servers,
            "unhealthy_servers": total_servers - healthy_servers,
            "health_percentage": (healthy_servers / total_servers * 100) if total_servers > 0 else 0,
            "last_check": datetime.now().isoformat()
        }

# Инициализация компонентов
neural_router = NeuralRouter()
error_tracker = ErrorTracker()

def check_server_health(server_name, config):
    """Проверка здоровья сервера"""
    try:
        url = f"http://host.docker.internal:{config['port']}{config['endpoint']}"
        start_time = time.time()
        
        if config['method'] == 'GET':
            response = requests.get(url, timeout=3)
        else:
            response = requests.post(url, json={}, timeout=3)
        
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            error_tracker.update_server_status(server_name, "healthy", response_time)
            return True
        else:
            error_tracker.log_error(server_name, "HTTP_ERROR", f"Status {response.status_code}")
            error_tracker.update_server_status(server_name, "unhealthy")
            return False
            
    except requests.exceptions.ConnectionError:
        error_tracker.log_error(server_name, "CONNECTION_ERROR", "Сервер недоступен")
        error_tracker.update_server_status(server_name, "offline")
        return False
    except requests.exceptions.Timeout:
        error_tracker.log_error(server_name, "TIMEOUT_ERROR", "Превышено время ожидания")
        error_tracker.update_server_status(server_name, "timeout")
        return False
    except Exception as e:
        error_tracker.log_error(server_name, "UNKNOWN_ERROR", str(e))
        error_tracker.update_server_status(server_name, "error")
        return False

def categorize_message(message):
    """Улучшенная категоризация сообщения с использованием нейронной сети"""
    message_lower = message.lower()
    
    # ПРИОРИТЕТ: Сначала проверяем ключевые слова для точной маршрутизации
    # 1. Программирование (должно быть первым, чтобы не перехватывалось математикой)
    if any(word in message_lower for word in ["программирование", "код", "python", "javascript", "java", "c++", "алгоритм", "программа", "напиши программу", "напиши код", "создай программу"]):
        logger.info("✅ Обнаружены ключевые слова программирования, используем programming")
        return "programming", 0.9
    
    # 2. Электротехника
    if any(word in message_lower for word in ["резистор", "конденсатор", "транзистор", "диод", "ток", "напряжение", "электротехника", "кирхгофа", "закон ома", "электричество", "мощность", "схема"]):
        logger.info("✅ Обнаружены электротехнические ключевые слова, используем electrical")
        return "electrical", 0.9
    
    # 3. Математика (последней, чтобы не перехватывала программирование)
    if any(word in message_lower for word in ["реши", "уравнение", "вычисли", "математика", "2+2", "формула", "интеграл", "производная", "сложение", "вычитание", "умножение", "деление"]):
        logger.info("✅ Обнаружены математические ключевые слова, используем mathematics")
        return "mathematics", 0.9
    
    # Нейронный анализ
    suggested_server, confidence = neural_router.suggest_route(message)
    
    logger.info(f"🧠 Нейросеть предложила: {suggested_server} (уверенность: {confidence:.2f})")
    
    # ПРИОРИТЕТ: Для electrical, mathematics, general сразу используем внутренние обработчики
    priority_handlers = ["electrical", "mathematics", "general"]
    if suggested_server in priority_handlers:
        logger.info(f"✅ Используем приоритетный внутренний обработчик: {suggested_server}")
        return suggested_server, confidence
    
    # Проверка доступности предложенного сервера
    if suggested_server in SERVERS:
        config = SERVERS[suggested_server]
        if check_server_health(suggested_server, config):
            logger.info(f"✅ Сервер {suggested_server} доступен")
            return suggested_server, confidence
        else:
            logger.warning(f"⚠️ Предложенный сервер {suggested_server} недоступен")
            
            # Пытаемся найти альтернативный сервер из той же категории
            category, _ = neural_router.analyze_message(message)
            alternative_servers = get_alternative_servers(category, suggested_server)
            
            for alt_server in alternative_servers:
                if alt_server in SERVERS:
                    alt_config = SERVERS[alt_server]
                    if check_server_health(alt_server, alt_config):
                        logger.info(f"✅ Найден альтернативный сервер: {alt_server}")
                        return alt_server, confidence * 0.8  # Немного снижаем уверенность
            
            # Если альтернативы нет, используем general
            logger.warning(f"⚠️ Альтернативы не найдены, используем general")
            return "general", confidence * 0.3  # Сильно снижаем уверенность
    
    # Если нейросеть не предложила сервер, используем общий анализ
    logger.warning(f"⚠️ Нейросеть не предложила сервер для: {message[:50]}...")
    
    # Простой анализ ключевых слов как fallback
    for server_name, config in SERVERS.items():
        if server_name == "general":
            continue
        for keyword in config.get('keywords', []):
            if keyword.lower() in message_lower:
                if check_server_health(server_name, config):
                    logger.info(f"✅ Найден сервер по ключевым словам: {server_name}")
                    return server_name, 0.6  # Средняя уверенность для keyword matching
    
    # В крайнем случае используем general
    logger.info("🔄 Используем general сервер как fallback")
    return "general", 0.1

def get_alternative_servers(category, excluded_server):
    """Получает альтернативные серверы для категории"""
    category_mapping = {
        "технические": ["controllers", "electrical", "radiomechanics", "plc_analysis"],
        "математические": ["mathematics", "advanced_math"],
        "программные": ["programming"],
        "системные": ["system_utils", "unified_manager"],
        "ai_связанные": ["neuro", "gai_server"],
        "общие": ["general"]
    }
    
    alternatives = category_mapping.get(category, ["general"])
    # Убираем исключенный сервер из списка
    return [s for s in alternatives if s != excluded_server]

def forward_request(server_name, message):
    """Пересылка запроса на соответствующий сервер или внутренний обработчик"""
    if server_name not in SERVERS:
        error_tracker.log_error(server_name, "CONFIG_ERROR", "Сервер не найден в конфигурации")
        return {"error": "Сервер не найден в конфигурации"}
    
    # ПРИОРИТЕТ: Внутренние обработчики для electrical, mathematics, general
    priority_handlers = ["electrical", "mathematics", "general"]
    if server_name in priority_handlers:
        internal_result = try_internal_handlers(server_name, message)
        if internal_result:
            return internal_result
        # Если внутренний обработчик не сработал, не пытаемся использовать внешний сервер
        # для этих категорий, а переходим к general
        if server_name != "general":
            logger.warning(f"⚠️ Внутренний обработчик {server_name} не сработал, используем general")
            return try_internal_handlers("general", message) or {"error": "Все обработчики недоступны"}
    
    # Для остальных серверов пробуем внутренние обработчики
    internal_result = try_internal_handlers(server_name, message)
    if internal_result:
        return internal_result
    
    # Если внутренние обработчики не сработали, пробуем внешние серверы
    config = SERVERS[server_name]
    
    try:
        url = f"http://host.docker.internal:{config['port']}{config['endpoint']}"
        
        if config['method'] == 'GET':
            # Для GET запросов добавляем параметры
            params = {}
            if server_name == 'controllers':
                # Динамическое определение темы для контроллеров
                message_lower = message.lower()
                if 'pmac' in message_lower:
                    params['topic'] = 'pmac'
                elif 'plc' in message_lower:
                    params['topic'] = 'plc'
                elif 'сервопривод' in message_lower or 'серводвигатель' in message_lower:
                    params['topic'] = 'сервопривод'
                else:
                    params['topic'] = 'general'
            
            response = requests.get(url, params=params, timeout=5)
        else:
            # Для POST запросов
            payload = {'message': message}
            response = requests.post(url, json=payload, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_tracker.log_error(server_name, "HTTP_ERROR", f"Status {response.status_code}")
            return {"error": f"Ошибка сервера: {response.status_code}"}
            
    except requests.exceptions.ConnectionError:
        error_tracker.log_error(server_name, "CONNECTION_ERROR", "Сервер недоступен")
        return {"error": "Сервер недоступен"}
    except requests.exceptions.Timeout:
        error_tracker.log_error(server_name, "TIMEOUT_ERROR", "Превышено время ожидания")
        return {"error": "Превышено время ожидания"}
    except Exception as e:
        error_tracker.log_error(server_name, "UNKNOWN_ERROR", str(e))
        return {"error": f"Неизвестная ошибка: {str(e)}"}

def try_internal_handlers(server_name, message):
    """Попытка обработки запроса внутренними обработчиками"""
    try:
        if server_name == "general":
            from handlers.general_handler import get_general_handler
            handler = get_general_handler()
            result = handler.handle_request(message)
            logger.info(f"✅ Использован внутренний General Handler")
            return result
            
        elif server_name == "electrical":
            from handlers.electrical_handler import get_electrical_handler
            handler = get_electrical_handler()
            result = handler.handle_request(message)
            logger.info(f"✅ Использован внутренний Electrical Handler")
            return result
            
        elif server_name == "mathematics":
            from handlers.mathematics_handler import get_mathematics_handler
            handler = get_mathematics_handler()
            result = handler.handle_request(message)
            logger.info(f"✅ Использован внутренний Mathematics Handler")
            return result
            
        else:
            logger.info(f"🔄 Нет внутреннего обработчика для {server_name}")
            return None
            
    except ImportError as e:
        logger.warning(f"⚠️ Не удалось импортировать обработчик для {server_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Ошибка в внутреннем обработчике {server_name}: {e}")
        return None

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья маршрутизатора"""
    return jsonify({
        "status": "healthy",
        "message": "Enhanced Smart Dispatcher работает",
        "timestamp": datetime.now().isoformat(),
        "neural_router": "active",
        "error_tracker": "active"
    }), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "Пустое сообщение"}), 400
        
        # --- УЛУЧШЕННЫЙ ПОИСК В БАЗЕ ЗНАНИЙ ---
        try:
            kb_logger.info(f"🔍 Поиск в базе знаний для: '{message[:30]}...'")
            kb_response = requests.post(
                "http://host.docker.internal:8093/api/knowledge/search",
                json={'query': message, 'limit': 3},  # Увеличиваем лимит для лучшего выбора
                timeout=5  # Увеличиваем таймаут
            )
            kb_logger.debug(f"  📡 Ответ базы знаний: HTTP {kb_response.status_code}")
            
            if kb_response.status_code == 200:
                kb_data = kb_response.json()
                kb_logger.debug(f"  📊 Получено результатов: {len(kb_data.get('results', []))}")
                
                if kb_data.get('results'):
                    # Ищем лучший результат с высоким порогом уверенности
                    best_result = None
                    for i, result in enumerate(kb_data['results']):
                        kb_logger.debug(f"    Результат {i+1}: score={result['score']:.2f}, title='{result.get('title', 'N/A')}'")
                        if result['score'] > 0.7:  # Снижаем порог для лучшего покрытия
                            best_result = result
                            break
                    
                    if best_result:
                        kb_logger.info(f"✅ Найден релевантный ответ в базе знаний: {best_result.get('title')} (score: {best_result['score']:.2f})")
                        
                        # Определяем категорию на основе найденного контента
                        content_category = best_result.get('category', 'knowledge_base')
                        
                        # Формируем улучшенный ответ
                        response_content = best_result.get('content', '')
                        if best_result.get('title'):
                            response_content = f"**{best_result['title']}**\n\n{response_content}"
                        
                        return jsonify({
                            'success': True,
                            'category': content_category,
                            'response': {
                                'content': response_content,
                                'title': best_result.get('title'),
                                'source': 'Knowledge Base',
                                'score': best_result['score']
                            },
                            'server': 'knowledge_base:8093',
                            'neural_used': False,
                            'confidence': best_result['score'],
                            'timestamp': datetime.now().isoformat(),
                            'knowledge_base_hit': True
                        })
                    else:
                        kb_logger.info("🔍 База знаний найдена, но нет достаточно релевантных результатов")
                else:
                    kb_logger.info("🔍 База знаний пуста или не содержит результатов")
            else:
                kb_logger.warning(f"⚠️ База знаний недоступна: HTTP {kb_response.status_code}")
        except requests.exceptions.ConnectionError:
            kb_logger.warning("⚠️ База знаний недоступна: нет соединения")
        except requests.exceptions.Timeout:
            kb_logger.warning("⚠️ База знаний недоступна: превышено время ожидания")
        except Exception as e:
            kb_logger.warning(f"⚠️ Ошибка при поиске в базе знаний: {e}")
        # --- КОНЕЦ УЛУЧШЕННОГО ПОИСКА ---
        
        # Категоризация с нейронной сетью
        routing_logger.info(f"🔀 Начинаем маршрутизацию для: '{message[:30]}...'")
        server_name, confidence = categorize_message(message)
        routing_logger.info(f"  🎯 Выбран сервер: {server_name} (уверенность: {confidence:.2f})")
        
        # Пересылка запроса
        routing_logger.info(f"  📤 Пересылаем запрос на сервер: {server_name}")
        result = forward_request(server_name, message)
        routing_logger.info(f"  📥 Получен ответ от сервера: {server_name}")
        
        # Добавление метаданных
        result.update({
            "routed_to": server_name,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "neural_analysis": True
        })
        
        routing_logger.info(f"  ✅ Маршрутизация завершена успешно")
        return jsonify(result), 200
        
    except Exception as e:
        error_tracker.log_error("dispatcher", "CHAT_ERROR", str(e))
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500

@app.route('/api/servers', methods=['GET'])
def list_servers():
    """Список всех серверов"""
    return jsonify({
        "servers": SERVERS,
        "total_count": len(SERVERS),
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/errors', methods=['GET'])
def get_errors():
    """Получение ошибок"""
    limit = request.args.get('limit', 50, type=int)
    errors = error_tracker.get_errors(limit)
    
    return jsonify({
        "errors": errors,
        "total_count": len(errors),
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/system/health', methods=['GET'])
def system_health():
    """Общее здоровье системы"""
    health = error_tracker.get_system_health()
    health.update({
        "server_status": error_tracker.server_status,
        "timestamp": datetime.now().isoformat()
    })
    
    return jsonify(health), 200

@app.route('/api/neural/analyze', methods=['POST'])
def neural_analyze():
    """Нейронный анализ сообщения"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "Пустое сообщение"}), 400
        
        category, confidence = neural_router.analyze_message(message)
        suggested_server, route_confidence = neural_router.suggest_route(message)
        
        return jsonify({
            "message": message,
            "category": category,
            "confidence": confidence,
            "suggested_server": suggested_server,
            "route_confidence": route_confidence,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        error_tracker.log_error("neural_analyzer", "ANALYSIS_ERROR", str(e))
        return jsonify({"error": f"Ошибка анализа: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Статус маршрутизатора"""
    return jsonify({
        "status": "running",
        "neural_router": "active",
        "error_tracker": "active",
        "servers_configured": len(SERVERS),
        "uptime": "active",
        "timestamp": datetime.now().isoformat()
    }), 200

if __name__ == "__main__":
    logger.info("🧠 Enhanced Smart Dispatcher с нейронной сетью запущен")
    logger.info("🔗 URL: http://localhost:8080")
    logger.info("📊 Endpoints:")
    logger.info("  - POST /api/chat - Основной чат")
    logger.info("  - GET /api/health - Проверка здоровья")
    logger.info("  - GET /api/servers - Список серверов")
    logger.info("  - GET /api/errors - Ошибки системы")
    logger.info("  - GET /api/system/health - Здоровье системы")
    logger.info("  - POST /api/neural/analyze - Нейронный анализ")
    logger.info("  - GET /api/status - Статус маршрутизатора")
    
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
