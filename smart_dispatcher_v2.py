#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный диспетчер для правильной маршрутизации запросов с fallback механизмом
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Конфигурация серверов
SERVERS = {
    'learning': {
        'port': 8091,
        'endpoint': '/api/learning/chat',
        'keywords': ['обучение', 'изучение', 'прогресс', 'понимание', 'научился', 'сегодня', 'делали', 'работали', 'взаимодействие', 'как проходит', 'что изучил', 'что научился', 'понимаешь процесс', 'наш процесс', 'взаимодействие', 'делали сегодня'],
        'priority': 10,  # Высокий приоритет для вопросов об обучении
        'fallback': 'general',
        'fallback_keywords': ['обучение', 'изучение', 'прогресс']
    },
    'electrical': {
        'port': 8087,
        'endpoint': '/api/electrical/explain',
        'keywords': ['закон', 'кирхгофа', 'резистор', 'транзистор', 'диод', 'контактор', 'реле', 'мощность', 'ток', 'напряжение', 'схема', 'электрические', 'электричество', 'цепи'],
        'priority': 5,
        'fallback': 'mathematics',
        'fallback_keywords': ['напряжение', 'ток', 'мощность', 'энергия', 'кинетическая', 'потенциальная', 'физика', 'формула']
    },
    'radiomechanics': {
        'port': 8089,
        'endpoint': '/api/radiomechanics/explain',
        'keywords': ['антенна', 'сигнал', 'радио', 'модуляция', 'частота', 'передатчик', 'приемник'],
        'fallback': 'general',
        'fallback_keywords': ['радио', 'сигнал', 'антенна']
    },
    'controllers': {
        'port': 9000,
        'endpoint': '/api/controllers/topic/general',
        'keywords': ['пид', 'регулятор', 'plc', 'контроллер', 'автоматизация', 'логика', 'события', 'прерывания', 'events', 'interrupts', 'ascii', 'команды', 'протокол', 'отправка', 'получение', 'ответы', 'чпу', 'cnc', 'числовое', 'программное', 'управление', 'передача', 'данные', 'g-коды', 'координаты', 'pmac', 'многоосевой', 'движение', 'позиционирование', 'траектория', 'ось', 'оси', 'серводвигатель', 'шаговый', 'энкодер', 'обратная связь', 'сервопривод', 'настроить', 'настройка', 'конфигурация', 'параметры', 'i130', 'i130a', 'i130b', 'i130c', 'i130d', 'i130e', 'i130f', 'i130g', 'i130h', 'i130i', 'i130j', 'i130k', 'i130l', 'i130m', 'i130n', 'i130o', 'i130p', 'i130q', 'i130r', 'i130s', 'i130t', 'i130u', 'i130v', 'i130w', 'i130x', 'i130y', 'i130z'],
        'fallback': 'programming',
        'fallback_keywords': ['plc', 'контроллер', 'автоматизация', 'программирование', 'логика', 'управление', 'ошибка', 'анализ', 'файл']
    },
    'mathematics': {
        'port': 8086,
        'endpoint': '/api/chat',
        'keywords': ['уравнение', 'квадратное', 'математика', 'алгебра', 'геометрия', 'арифметика', '+', '-', '*', '/', '=', 'вычислить', 'посчитать', 'сложить', 'вычесть', 'умножить', 'разделить', 'число', 'цифра', 'результат', 'ответ'],
        'fallback': 'general',
        'fallback_keywords': ['математика', 'вычислить', 'посчитать']
    },
    'programming': {
        'port': 8088,
        'endpoint': '/api/programming/explain',
        'keywords': ['продвинутые', 'специфические', 'функции', 'алгоритмы', 'алгоритм', 'программирование', 'код', 'разработка', 'python', 'javascript', 'c++', 'java', 'автоматизация', 'промышленная', 'конвейер', 'управление', 'сортировка', 'ошибки', 'error', 'xml', 'обработка'],
        'fallback': 'general',
        'fallback_keywords': ['программирование', 'код', 'алгоритм', 'python', 'ошибка']
    },
    'general': {
        'port': 8085,
        'endpoint': '/api/chat',
        'keywords': ['привет', 'hello', 'hi', 'здравствуй', 'помощь', 'help', 'справка', 'статус', 'status', 'работает', 'онлайн', 'что', 'как', 'объясни', 'расскажи'],
        'fallback': None,  # general не имеет fallback
        'fallback_keywords': []
    }
}

def categorize_message(message):
    """Определяет категорию сообщения на основе ключевых слов"""
    message_lower = message.lower()
    scores = {}
    
    for category, config in SERVERS.items():
        score = 0
        for keyword in config['keywords']:
            if keyword.lower() in message_lower:
                score += 1
        scores[category] = score
    
    if scores and max(scores.values()) > 0:
        best_category = max(scores, key=scores.get)
        logger.info(f"📊 Категоризация: '{message[:50]}...' → {best_category} (score: {scores[best_category]})")
        return best_category
    
    # Если нет совпадений, возвращаем general как fallback
    logger.info(f"❓ Неопределенная категория: '{message[:50]}...' → general (fallback)")
    return 'general'

def check_server_health(category):
    """Проверяет доступность сервера"""
    if category not in SERVERS:
        return False
    
    config = SERVERS[category]
    url = f"http://localhost:{config['port']}/api/health"
    
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

def forward_request(category, message, max_retries=2):
    """Пересылает запрос к соответствующему серверу с fallback механизмом"""
    if category not in SERVERS:
        return None, "Неизвестная категория"
    
    config = SERVERS[category]
    url = f"http://localhost:{config['port']}{config['endpoint']}"
    
    # Подготавливаем данные в зависимости от сервера
    if category in ['electrical', 'radiomechanics', 'controllers', 'programming']:
        payload = {'concept': message}
    else:  # mathematics, general
        payload = {'message': message}
    
    try:
        logger.info(f"🌐 Отправляем запрос к {category} на {url}")
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            logger.info(f"✅ Сервер {category} ответил успешно")
            return response.json(), None
        else:
            logger.warning(f"⚠️ Сервер {category} вернул {response.status_code}")
            return None, f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        logger.error(f"❌ Ошибка соединения с {category}: {e}")
        
        # Проверяем, есть ли fallback
        if config.get('fallback') and max_retries > 0:
            fallback_category = config['fallback']
            fallback_keywords = config.get('fallback_keywords', [])
            
            # Проверяем, подходит ли сообщение для fallback
            message_lower = message.lower()
            if any(keyword.lower() in message_lower for keyword in fallback_keywords):
                logger.info(f"🔄 Fallback: отправляем запрос к {fallback_category}")
                return forward_request(fallback_category, message, max_retries - 1)
            else:
                logger.info(f"🔄 Fallback: отправляем запрос к general (универсальный fallback)")
                return forward_request('general', message, max_retries - 1)
        
        # Если нет fallback или исчерпаны попытки, возвращаем ошибку
        return None, str(e)

@app.route('/')
def index():
    """Главная страница - веб-интерфейс Rubin IDE"""
    try:
        return send_from_directory('.', 'RubinIDE.html')
    except FileNotFoundError:
        return jsonify({
            'name': 'Smart Dispatcher',
            'version': '2.0',
            'status': 'online',
            'servers': {name: f"localhost:{config['port']}" for name, config in SERVERS.items()},
            'note': 'RubinIDE.html not found'
        })

@app.route('/matrix/RubinDeveloper.html')
def rubin_developer():
    """Rubin Developer интерфейс"""
    return send_from_directory('matrix', 'RubinDeveloper.html')

@app.route('/test-rubin')
def test_rubin():
    """Тестовая страница для RubinDeveloper"""
    return send_from_directory('.', 'test_rubin_developer.html')

@app.route('/api/health')
def health_check():
    """Проверка здоровья диспетчера"""
    return jsonify({
        'status': 'healthy',
        'dispatcher': 'Smart Dispatcher v2.0',
        'servers': {name: 'online' if check_server_health(name) else 'offline' 
                   for name in SERVERS.keys()}
    })

@app.route('/api/dispatcher/info')
def dispatcher_info():
    """Информация о диспетчере"""
    return jsonify({
        'name': 'Smart Dispatcher',
        'version': '2.0',
        'status': 'online',
        'servers': {name: f"localhost:{config['port']}" for name, config in SERVERS.items()},
        'features': ['fallback_mechanism', 'health_check', 'intelligent_routing']
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Отсутствует поле message'}), 400
        
        message = data['message']
        if not message.strip():
            return jsonify({'error': 'Пустое сообщение'}), 400
        
        # Определяем категорию
        category = categorize_message(message)
        
        # Пересылаем запрос
        result, error = forward_request(category, message)
        
        if result:
            return jsonify({
                'success': True,
                'category': category,
                'response': result
            })
        else:
            return jsonify({
                'success': False,
                'error': error,
                'category': category
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка в chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/servers/status')
def servers_status():
    """Статус всех серверов"""
    status = {}
    for name in SERVERS.keys():
        status[name] = {
            'online': check_server_health(name),
            'port': SERVERS[name]['port'],
            'fallback': SERVERS[name].get('fallback', 'none')
        }
    
    return jsonify({
        'servers': status,
        'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown'
    })

if __name__ == '__main__':
    print("🚀 Smart Dispatcher v2.0 запущен")
    print("URL: http://localhost:8080")
    print("Доступные серверы:")
    for name, config in SERVERS.items():
        status = "✅" if check_server_health(name) else "❌"
        fallback = f" → {config.get('fallback', 'none')}" if config.get('fallback') else ""
        print(f"  - {name}: localhost:{config['port']} {status}{fallback}")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
