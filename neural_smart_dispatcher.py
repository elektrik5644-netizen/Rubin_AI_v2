#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интегрированный умный диспетчер с нейронной сетью для Rubin AI v2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Попытка импорта нейронной сети
try:
    from neural_rubin import get_neural_rubin
    NEURAL_NETWORK_AVAILABLE = True
    logger.info("🧠 Нейронная сеть доступна!")
except ImportError as e:
    NEURAL_NETWORK_AVAILABLE = False
    logger.warning(f"⚠️ Нейронная сеть недоступна: {e}")

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Обработка CORS preflight запросов
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
        return response

# Установка правильных заголовков для всех ответов
@app.after_request
def after_request(response):
    if response.content_type and 'application/json' in response.content_type:
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# Конфигурация серверов
SERVERS = {
    'electrical': {
        'port': 8087,
        'endpoint': '/api/electrical/explain',
        'keywords': ['закон', 'кирхгофа', 'резистор', 'транзистор', 'диод', 'контактор', 'реле', 'мощность', 'ток', 'напряжение', 'схема', 'электрические', 'электричество', 'цепи', 'шим', 'плата', 'модуляция', 'импульсная', 'широтно', 'скважность', 'переключение']
    },
    'radiomechanics': {
        'port': 8089,
        'endpoint': '/api/radiomechanics/explain',
        'keywords': ['антенна', 'сигнал', 'радио', 'модуляция', 'частота', 'передатчик', 'приемник']
    },
    'controllers': {
        'port': 9000,
        'endpoint': '/api/controllers/topic/general',
        'keywords': ['пид', 'регулятор', 'plc', 'контроллер', 'автоматизация', 'логика', 'события', 'прерывания', 'events', 'interrupts', 'ascii', 'команды', 'протокол', 'отправка', 'получение', 'ответы', 'чпу', 'cnc', 'числовое', 'программное', 'управление', 'передача', 'данные', 'g-коды', 'координаты']
    },
    'mathematics': {
        'port': 8086,
        'endpoint': '/api/chat',
        'keywords': ['уравнение', 'квадратное', 'математика', 'алгебра', 'геометрия', 'арифметика']
    },
    'programming': {
        'port': 8088,
        'endpoint': '/api/programming/explain',
        'keywords': ['продвинутые', 'специфические', 'функции', 'алгоритмы', 'алгоритм', 'программирование', 'код', 'разработка', 'python', 'javascript', 'c++', 'java', 'автоматизация', 'промышленная', 'конвейер', 'управление', 'сортировка', 'ошибки', 'error', 'xml', 'обработка', 'сценарии', 'сценарий', 'решение', 'проблем', 'проблемы']
    },
    'general': {
        'port': 8085,
        'endpoint': '/api/chat',
        'keywords': ['привет', 'hello', 'hi', 'здравствуй', 'помощь', 'help', 'справка', 'статус', 'status', 'работает', 'онлайн', 'что', 'как', 'объясни', 'расскажи']
    }
}

def categorize_message_neural(message):
    """Категоризация сообщения с помощью нейронной сети"""
    if not NEURAL_NETWORK_AVAILABLE:
        return categorize_message_keywords(message)
    
    try:
        neural_ai = get_neural_rubin()
        
        # Получаем категорию от нейронной сети
        category, confidence = neural_ai.classify_question(message)
        
        logger.info(f"🧠 Нейронная сеть классифицировала: '{message[:50]}...' → {category} (уверенность: {confidence:.2f})")
        
        # Маппинг категорий нейронной сети на наши серверы
        neural_to_server = {
            'математика': 'mathematics',
            'физика': 'mathematics',  # Физика тоже может быть математической
            'электротехника': 'electrical',
            'программирование': 'programming',
            'техника': 'controllers',
            'общие_вопросы': 'general',
            'другое': 'general'
        }
        
        server_category = neural_to_server.get(category, 'general')
        
        # Если уверенность низкая, используем keyword-based fallback
        if confidence < 0.6:
            logger.info(f"⚠️ Низкая уверенность нейронной сети ({confidence:.2f}), используем keyword-based fallback")
            return categorize_message_keywords(message)
        
        return server_category
        
    except Exception as e:
        logger.error(f"❌ Ошибка нейронной сети: {e}")
        return categorize_message_keywords(message)

def categorize_message_keywords(message):
    """Категоризация сообщения по ключевым словам (fallback)"""
    message_lower = message.lower()
    
    # Подсчитываем совпадения для каждой категории
    scores = {}
    for category, config in SERVERS.items():
        score = 0
        for keyword in config['keywords']:
            if keyword in message_lower:
                score += 1
        scores[category] = score
    
    # Находим категорию с наибольшим количеством совпадений
    if scores and max(scores.values()) > 0:
        best_category = max(scores, key=scores.get)
        logger.info(f"📊 Keyword-based категоризация: '{message[:50]}...' → {best_category} (score: {scores[best_category]})")
        return best_category
    
    # Если нет совпадений, возвращаем general как fallback
    logger.info(f"❓ Неопределенная категория: '{message[:50]}...' → general (fallback)")
    return 'general'

def forward_request(category, message):
    """Пересылает запрос к соответствующему серверу"""
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
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ Сервер {category} ответил успешно")
            return response.json(), category
        else:
            logger.error(f"❌ Сервер {category} вернул ошибку: {response.status_code}")
            return None, f"Ошибка сервера {category}: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Ошибка соединения с {category}: {e}")
        return None, f"Ошибка соединения с {category}: {e}"

@app.route('/')
def index():
    return jsonify({
        'name': 'Smart Dispatcher with Neural Network',
        'version': '2.0',
        'status': 'online',
        'neural_network': 'available' if NEURAL_NETWORK_AVAILABLE else 'unavailable',
        'servers': list(SERVERS.keys())
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Категоризация сообщения
        if NEURAL_NETWORK_AVAILABLE:
            category = categorize_message_neural(message)
        else:
            category = categorize_message_keywords(message)
        
        # Пересылка запроса
        response, server_category = forward_request(category, message)
        
        if response:
            return jsonify({
                'success': True,
                'category': server_category,
                'response': response,
                'server': f'localhost:{SERVERS[server_category]["port"]}',
                'neural_used': NEURAL_NETWORK_AVAILABLE
            })
        else:
            return jsonify({
                'success': False,
                'error': server_category,
                'neural_used': NEURAL_NETWORK_AVAILABLE
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка в чате: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/neural-status')
def neural_status():
    """Статус нейронной сети"""
    if not NEURAL_NETWORK_AVAILABLE:
        return jsonify({
            'available': False,
            'message': 'Нейронная сеть недоступна'
        })
    
    try:
        neural_ai = get_neural_rubin()
        stats = neural_ai.get_neural_stats()
        
        return jsonify({
            'available': True,
            'stats': stats,
            'message': 'Нейронная сеть активна'
        })
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e),
            'message': 'Ошибка нейронной сети'
        })

@app.route('/api/neural-feedback', methods=['POST'])
def neural_feedback():
    """Обучение нейронной сети на основе обратной связи"""
    if not NEURAL_NETWORK_AVAILABLE:
        return jsonify({
            'success': False,
            'message': 'Нейронная сеть недоступна для обучения'
        }), 400
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        correct_category = data.get('correct_category', '').strip()
        rating = data.get('rating', 0)
        
        if not question or not correct_category:
            return jsonify({
                'success': False,
                'message': 'Необходимы question и correct_category'
            }), 400
        
        # Получаем нейронную сеть и обучаем
        neural_ai = get_neural_rubin()
        success = neural_ai.learn_from_feedback(question, correct_category, rating)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Обучение завершено: "{question}" → {correct_category} (оценка: {rating}/5)'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Ошибка при обучении'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка обучения: {e}")
        return jsonify({
            'success': False,
            'message': f'Ошибка обучения: {str(e)}'
        }), 500

@app.route('/api/health')
def health():
    """Проверка состояния диспетчера"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'neural_network': NEURAL_NETWORK_AVAILABLE,
        'servers_count': len(SERVERS)
    })

if __name__ == '__main__':
    logger.info("Smart Dispatcher с нейронной сетью запущен")
    logger.info("URL: http://localhost:8080")
    logger.info("Доступные серверы:")
    for name, config in SERVERS.items():
        logger.info(f"  - {name}: localhost:{config['port']}")
    
    if NEURAL_NETWORK_AVAILABLE:
        logger.info("🧠 Нейронная сеть интегрирована!")
    else:
        logger.info("⚠️ Нейронная сеть недоступна, используется keyword-based fallback")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
