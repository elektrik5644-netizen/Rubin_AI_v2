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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Очередь для ошибок
error_queue = queue.Queue()

# Конфигурация серверов
SERVERS = {
    "general": {
        "port": 8085,
        "endpoint": "/api/health",
        "method": "GET",
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
        "endpoint": "/api/electrical/status",
        "method": "GET",
        "keywords": ["электротехника", "электричество", "ток", "напряжение", "мощность", "схема", "диод"],
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
    "neuro": {
        "port": 8090,
        "endpoint": "/api/neuro/health",
        "method": "GET",
        "keywords": ["нейросеть", "нейронная сеть", "машинное обучение", "искусственный интеллект", "ai"],
        "priority": 3
    },
    "controllers": {
        "port": 9000,
        "endpoint": "/api/controllers/topic/general",
        "method": "GET",
        "keywords": ["контроллер", "pmac", "plc", "сервопривод", "серводвигатель", "привод", "двигатель"],
        "priority": 2
    },
    "plc_analysis": {
        "port": 8099,
        "endpoint": "/api/plc/health",
        "method": "GET",
        "keywords": ["plc анализ", "ladder logic", "программа plc", "отладка plc"],
        "priority": 2
    },
    "advanced_math": {
        "port": 8100,
        "endpoint": "/api/advanced_math/health",
        "method": "GET",
        "keywords": ["продвинутая математика", "сложные вычисления", "высшая математика"],
        "priority": 2
    },
    "data_processing": {
        "port": 8101,
        "endpoint": "/api/data_processing/health",
        "method": "GET",
        "keywords": ["обработка данных", "анализ данных", "статистика", "визуализация"],
        "priority": 2
    },
    "search_engine": {
        "port": 8102,
        "endpoint": "/api/search/health",
        "method": "GET",
        "keywords": ["поиск", "гибридный поиск", "семантический поиск"],
        "priority": 2
    },
    "system_utils": {
        "port": 8103,
        "endpoint": "/api/system/health",
        "method": "GET",
        "keywords": ["системные утилиты", "диагностика", "мониторинг", "оптимизация", "проблемы", "ошибки", "здоровье системы"],
        "priority": 1
    },
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
        self.knowledge_base = {
            "технические": ["pmac", "plc", "сервопривод", "электротехника", "радиомеханика", "контроллер"],
            "математические": ["математика", "вычисление", "формула", "уравнение", "интеграл", "производная"],
            "программные": ["программирование", "код", "алгоритм", "python", "javascript", "java"],
            "системные": ["диагностика", "мониторинг", "оптимизация", "проблемы", "ошибки"],
            "ai_связанные": ["нейросеть", "машинное обучение", "искусственный интеллект", "ai"],
            "общие": ["привет", "как дела", "что нового", "помощь", "справка"]
        }
    
    def analyze_message(self, message):
        """Анализ сообщения с помощью нейронной логики"""
        message_lower = message.lower()
        
        # Подсчет совпадений по категориям
        category_scores = {}
        for category, keywords in self.knowledge_base.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                category_scores[category] = score
        
        # Определение приоритетной категории
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category] / len(message_lower.split())
            return best_category, confidence
        
        return "общие", 0.1
    
    def suggest_route(self, message):
        """Предложение маршрута на основе нейронного анализа"""
        category, confidence = self.analyze_message(message)
        
        # Маппинг категорий на серверы
        category_mapping = {
            "технические": ["controllers", "electrical", "radiomechanics", "plc_analysis"],
            "математические": ["mathematics", "advanced_math"],
            "программные": ["programming"],
            "системные": ["system_utils", "unified_manager"],
            "ai_связанные": ["neuro", "gai_server"],
            "общие": ["general"]
        }
        
        suggested_servers = category_mapping.get(category, ["general"])
        return suggested_servers[0], confidence

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
        url = f"http://localhost:{config['port']}{config['endpoint']}"
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
    """Категоризация сообщения с использованием нейронной сети"""
    message_lower = message.lower()
    
    # Нейронный анализ
    suggested_server, confidence = neural_router.suggest_route(message)
    
    # Проверка доступности предложенного сервера
    if suggested_server in SERVERS:
        config = SERVERS[suggested_server]
        if check_server_health(suggested_server, config):
            return suggested_server, confidence
    
    # Fallback: поиск по ключевым словам
    best_match = None
    best_score = 0
    
    for server_name, config in SERVERS.items():
        score = 0
        for keyword in config['keywords']:
            if keyword in message_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = server_name
    
    if best_match and best_score > 0:
        return best_match, best_score / len(message_lower.split())
    
    return "general", 0.1

def forward_request(server_name, message):
    """Пересылка запроса на соответствующий сервер"""
    if server_name not in SERVERS:
        error_tracker.log_error(server_name, "CONFIG_ERROR", "Сервер не найден в конфигурации")
        return {"error": "Сервер не найден в конфигурации"}
    
    config = SERVERS[server_name]
    
    try:
        url = f"http://localhost:{config['port']}{config['endpoint']}"
        
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
        
        # Категоризация с нейронной сетью
        server_name, confidence = categorize_message(message)
        
        # Пересылка запроса
        result = forward_request(server_name, message)
        
        # Добавление метаданных
        result.update({
            "routed_to": server_name,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "neural_analysis": True
        })
        
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
    
    app.run(port=8080, debug=True, use_reloader=False)
