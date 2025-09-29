#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой диспетчер для правильной маршрутизации запросов
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import logging
import os
import json
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app)

# Импорт менеджера директив
try:
    from directives_manager import check_and_apply_directives, process_directives_command
    DIRECTIVES_AVAILABLE = True
except ImportError:
    DIRECTIVES_AVAILABLE = False

# Контекстная память для диалогов
CONVERSATION_HISTORY = {
    "sessions": {},
    "global_context": {
        "last_topics": [],
        "frequent_categories": {},
        "user_preferences": {}
    }
}

# Конфигурация серверов
SERVERS = {
    'general': {
        'port': 8085,
        'keywords': ['общее', 'помощь', 'справка', 'что', 'как', 'почему', 'зачем', 'где', 'когда', 'кто']
    },
    'mathematics': {
        'port': 8086,
        'keywords': ['математика', 'матем', 'число', 'числа', 'считать', 'вычислить', 'формула', 'уравнение', 'квадратное уравнение', 'алгебра', 'геометрия', 'тригонометрия', 'логарифм', 'интеграл', 'производная', 'функция', 'график', 'графики', 'математический', 'вычисления', 'расчет', 'расчеты']
    },
    'electrical': {
        'port': 8087,
        'keywords': ['электротехника', 'электричество', 'электрический', 'ток', 'напряжение', 'сопротивление', 'резистор', 'конденсатор', 'катушка', 'индуктивность', 'емкость', 'схема', 'схемы', 'цепи', 'закон', 'кирхгофа', 'ом', 'ватт', 'вольт', 'ампер', 'мощность', 'коэффициент мощности', 'power factor', 'cos φ', 'реактивная мощность', 'модубус', 'modbus', 'rtu', 'протокол']
    },
    'programming': {
        'port': 8088,
        'keywords': ['программирование', 'программировать', 'код', 'кодить', 'python', 'java', 'c++', 'javascript', 'алгоритм', 'алгоритмы', 'переменные', 'переменная', 'логика', 'управления', 'if', 'endif', 'условия', 'циклы', 'функции', 'методы', 'классы', 'объекты', 'программа', 'программы', 'разработка', 'разработать']
    },
    'neuro': {
        'port': 8090,
        'keywords': ['нейронная', 'нейронные', 'сеть', 'сети', 'машинное', 'обучение', 'обучается', 'обучать', 'тренировка', 'тренировать', 'искусственный', 'интеллект', 'ai', 'ml', 'deep learning', 'нейросеть', 'нейросети']
    },
    'controllers': {
        'port': 9000,
        'keywords': ['контроллер', 'контроллеры', 'плк', 'ПЛК', 'plc', 'автоматизация', 'автоматизировать', 'scada', 'скада', 'мониторинг', 'диспетчеризация', 'pmac', 'настройка', 'pid', 'сервопривод', 'привод', 'двигатель', 'мотор']
    },
    'gai': {
        'port': 8104,
        'keywords': ['генерация', 'генерировать', 'создать', 'создание', 'текст', 'код', 'диаграмма', 'диаграммы', 'изображение', 'изображения', 'контент', 'контента']
    }
}

def get_session_id():
    """Получает или создает ID сессии"""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return session_id

def add_to_history(session_id, message, category, response):
    """Добавляет сообщение в историю диалога"""
    if session_id not in CONVERSATION_HISTORY["sessions"]:
        CONVERSATION_HISTORY["sessions"][session_id] = {
            "messages": [],
            "last_category": None,
            "created_at": datetime.now().isoformat()
        }
    
    # Добавляем сообщение
    CONVERSATION_HISTORY["sessions"][session_id]["messages"].append({
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "category": category,
        "response": response[:200] + "..." if len(response) > 200 else response
    })
    
    # Обновляем последнюю категорию
    CONVERSATION_HISTORY["sessions"][session_id]["last_category"] = category
    
    # Обновляем глобальный контекст
    CONVERSATION_HISTORY["global_context"]["last_topics"].append(category)
    if len(CONVERSATION_HISTORY["global_context"]["last_topics"]) > 10:
        CONVERSATION_HISTORY["global_context"]["last_topics"].pop(0)
    
    # Обновляем частоту категорий
    if category in CONVERSATION_HISTORY["global_context"]["frequent_categories"]:
        CONVERSATION_HISTORY["global_context"]["frequent_categories"][category] += 1
    else:
        CONVERSATION_HISTORY["global_context"]["frequent_categories"][category] = 1

def get_context_for_message(session_id, message):
    """Генерирует контекстную подсказку для сообщения"""
    context_hint = ""
    
    if session_id in CONVERSATION_HISTORY["sessions"]:
        session_data = CONVERSATION_HISTORY["sessions"][session_id]
        last_category = session_data.get("last_category")
        recent_messages = session_data.get("messages", [])[-3:]
        
        if last_category:
            context_hint += f"[Контекст: последняя тема - {last_category}] "
        
        if recent_messages:
            recent_topics = [msg["category"] for msg in recent_messages]
            context_hint += f"[Недавние темы: {', '.join(recent_topics)}] "
    
    return context_hint

def categorize_message(message):
    """Категоризирует сообщение и определяет целевой сервер"""
    message_lower = message.lower()
    
    # Технические категории с приоритетом
    technical_categories = ['electrical', 'mathematics', 'programming', 'controllers', 'neuro', 'gai']
    
    # Проверяем технические категории в первую очередь
    for category in technical_categories:
        if category in SERVERS:
            keywords = SERVERS[category]['keywords']
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    logger.info(f"🔍 Техническая категория '{category}' найдена по ключевому слову '{keyword}'")
                    return category
    
    # Проверяем общие категории
    for category, config in SERVERS.items():
        if category in technical_categories:
            continue  # Уже проверили технические категории
        
        keywords = config['keywords']
        for keyword in keywords:
            if keyword.lower() in message_lower:
                logger.info(f"🔍 Общая категория '{category}' найдена по ключевому слову '{keyword}'")
                return category
    
    # Если ничего не найдено, возвращаем общую категорию
    logger.info("🔍 Категория не определена, используем 'general'")
    return 'general'

def _is_greeting(message):
    """Проверяет, является ли сообщение приветствием"""
    greetings = ['привет', 'hello', 'hi', 'здравствуй', 'добро пожаловать', 'как дела', 'как поживаешь']
    message_lower = message.lower().strip()
    return any(greeting in message_lower for greeting in greetings)

def _extract_text_from_result(result):
    """Извлекает текстовый ответ из результата"""
    if isinstance(result, str):
        return result
    elif isinstance(result, dict):
        if 'response' in result:
            response = result['response']
            if isinstance(response, dict) and 'explanation' in response:
                return response['explanation']
            elif isinstance(response, str):
                return response
        elif 'explanation' in result:
            return result['explanation']
        elif 'message' in result:
            return result['message']
        elif 'text' in result:
            return result['text']
        else:
            return str(result)
    elif isinstance(result, list) and len(result) > 0:
        return _extract_text_from_result(result[0])
    else:
        return str(result)

def forward_request(message, category):
    """Перенаправляет запрос на соответствующий сервер"""
    if category not in SERVERS:
        logger.error(f"❌ Неизвестная категория: {category}")
        return {"error": f"Неизвестная категория: {category}"}
    
    server_config = SERVERS[category]
    port = server_config['port']
    url = f"http://localhost:{port}"
    
    # Получаем ID сессии
    session_id = get_session_id()
    
    # Генерируем контекстную подсказку
    context_hint = get_context_for_message(session_id, message)
    contextual_message = f"{context_hint}{message}"
    
    logger.info(f"📡 Направляю к модулю: {category} (порт {port})")
    
    try:
        if category == 'controllers':
            # Для контроллеров используем специальный эндпоинт
            response = requests.post(
                f"{url}/api/controllers/topic/general",
                json={"message": contextual_message},
                timeout=30
            )
        else:
            # Для остальных модулей используем стандартный эндпоинт
            response = requests.post(
                f"{url}/api/chat",
                json={"message": contextual_message},
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            extracted_text = _extract_text_from_result(result)
            
            # Добавляем в историю
            add_to_history(session_id, message, category, extracted_text)
            
            # Применяем директивы если доступны
            if DIRECTIVES_AVAILABLE:
                context = {
                    "category": category,
                    "message": message,
                    "response": extracted_text,
                    "session_id": session_id
                }
                directive_results = check_and_apply_directives(context)
                if directive_results:
                    logger.info(f"📋 Применены директивы: {len(directive_results)}")
                    if isinstance(result, dict):
                        result["directives_applied"] = directive_results
            
            return extracted_text
        else:
            logger.error(f"❌ Ошибка HTTP {response.status_code} от {category}")
            return f"Ошибка запроса к {category}: HTTP {response.status_code}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ошибка сети при обращении к {category}: {e}")
        return f"Ошибка сети: Не удается подключиться к серверу {category}"
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при обращении к {category}: {e}")
        return f"Ошибка: {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной эндпоинт для чата"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Сообщение не может быть пустым"}), 400
        
        logger.info(f"🔍 Анализирую вопрос: \"{message}\"")
        
        # Проверяем команды директив
        if DIRECTIVES_AVAILABLE and any(cmd in message.lower() for cmd in [
            'прими директиву', 'список директив', 'удали директиву', 
            'статистика директив', 'помощь по директивам'
        ]):
            user_id = data.get('user_id', 'default')
            directive_result = process_directives_command(message, user_id)
            return jsonify(directive_result)
        
        # Определяем категорию
        category = categorize_message(message)
        
        # Обрабатываем приветствия
        if _is_greeting(message):
            greeting_response = "Привет! Готов помочь по программированию, электротехнике, автоматизации и математике. Чем заняться?"
            add_to_history(get_session_id(), message, 'general', greeting_response)
            return jsonify({"response": greeting_response})
        
        # Перенаправляем запрос
        result = forward_request(message, category)
        
        return jsonify({"response": result})
    
    except Exception as e:
        logger.error(f"❌ Ошибка в chat(): {e}")
        return jsonify({"error": f"Внутренняя ошибка сервера: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья Smart Dispatcher"""
    module_status = {}
    for name, config in SERVERS.items():
        try:
            response = requests.get(f"http://localhost:{config['port']}/api/health", timeout=5)
            module_status[name] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'port': config['port'],
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            module_status[name] = {
                'status': 'unreachable',
                'port': config['port'],
                'error': str(e)
            }
    
    unhealthy_count = sum(1 for status in module_status.values() if status['status'] != 'healthy')
    overall_status = 'healthy' if unhealthy_count == 0 else 'degraded' if unhealthy_count < len(SERVERS) else 'critical'
    
    return jsonify({
        'service': 'Smart Dispatcher',
        'status': overall_status,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'modules': module_status,
        'total_modules': len(SERVERS),
        'healthy_modules': len(SERVERS) - unhealthy_count,
        'unhealthy_modules': unhealthy_count
    })

@app.route('/api/chat/history', methods=['GET'])
def get_history():
    """Получение истории диалогов"""
    session_id = request.args.get('session_id')
    if session_id and session_id in CONVERSATION_HISTORY["sessions"]:
        return jsonify({
            'success': True,
            'session_id': session_id,
            'history': CONVERSATION_HISTORY["sessions"][session_id]
        })
    else:
        return jsonify({
            'success': True,
            'sessions': list(CONVERSATION_HISTORY["sessions"].keys()),
            'global_context': CONVERSATION_HISTORY["global_context"]
        })

@app.route('/api/chat/context', methods=['GET'])
def get_context():
    """Получение контекста диалога"""
    session_id = request.args.get('session_id')
    if session_id:
        context_hint = get_context_for_message(session_id, "")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'context': context_hint,
            'session_data': CONVERSATION_HISTORY["sessions"].get(session_id, {})
        })
    else:
        return jsonify({
            'success': True,
            'global_context': CONVERSATION_HISTORY["global_context"]
        })

@app.route('/api/status', methods=['GET'])
def status():
    """Статус Smart Dispatcher"""
    return jsonify({
        'service': 'Smart Dispatcher',
        'status': 'online',
        'port': 8080,
        'version': '1.0',
        'modules': list(SERVERS.keys()),
        'directives_available': DIRECTIVES_AVAILABLE,
        'uptime': 'running',
        'requests_processed': 'active'
    })

if __name__ == '__main__':
    print("🚀 Запуск Smart Dispatcher...")
    print("📡 Порт: 8080")
    print("🔗 URL: http://localhost:8080")
    print("📋 Эндпоинты:")
    print("  - POST /api/chat - Основной чат")
    print("  - GET /api/health - Проверка здоровья")
    print("  - GET /api/chat/history - История диалогов")
    print("  - GET /api/chat/context - Контекст диалога")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8080, debug=False)






