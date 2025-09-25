#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенный Smart Dispatcher для Rubin AI v2
Без Ethical Core, с простой логикой и надежной обработкой ошибок
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import logging
import os
import time
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Конфигурация серверов (упрощенная)
SERVERS = {
    'electrical': {
        'port': 8087,
        'endpoint': '/api/electrical/status',
        'method': 'GET',
        'keywords': ['закон', 'кирхгофа', 'резистор', 'транзистор', 'диод', 'контактор', 'реле', 'мощность', 'ток', 'напряжение', 'схема', 'электрические', 'электричество', 'цепи'],
        'description': 'Электротехника и электрические схемы'
    },
    'mathematics': {
        'port': 8086,
        'endpoint': '/health',
        'method': 'GET',
        'keywords': ['уравнение', 'математика', 'алгебра', 'геометрия', 'арифметика', '+', '-', '*', '/', '=', 'вычислить', 'посчитать'],
        'description': 'Математические вычисления'
    },
    'programming': {
        'port': 8088,
        'endpoint': '/api/programming/explain',
        'method': 'GET',
        'keywords': ['код', 'программа', 'алгоритм', 'python', 'java', 'c++', 'javascript', 'разработка', 'функция', 'класс', 'переменная'],
        'description': 'Программирование и разработка'
    },
    'general': {
        'port': 8085,
        'endpoint': '/api/chat',
        'method': 'POST',
        'keywords': ['привет', 'как дела', 'что делаешь', 'кто ты', 'помощь', 'справка'],
        'description': 'Общие вопросы и помощь'
    },
    'neuro': {
        'port': 8090,
        'endpoint': '/api/neuro/chat',
        'method': 'POST',
        'keywords': ['нейросеть', 'искусственный интеллект', 'мозг', 'обучение', 'анализ', 'нейронная сеть'],
        'description': 'Нейронные сети и ИИ'
    },
    'plc_analysis': {
        'port': 8099,
        'endpoint': '/api/plc/analyze',
        'method': 'POST',
        'keywords': ['plc', 'контроллер', 'программа', 'анализ', 'диагностика', 'программируемый логический контроллер'],
        'description': 'Анализ PLC программ'
    },
    'advanced_math': {
        'port': 8100,
        'endpoint': '/api/math/advanced',
        'method': 'POST',
        'keywords': ['интеграл', 'производная', 'дифференциал', 'система уравнений', 'сложная математика', 'математический анализ'],
        'description': 'Продвинутая математика'
    },
    'data_processing': {
        'port': 8101,
        'endpoint': '/api/data_processing/health',
        'method': 'GET',
        'keywords': ['данные', 'обработка', 'анализ', 'статистика', 'фильтрация', 'обработка данных'],
        'description': 'Обработка и анализ данных'
    },
    'search_engine': {
        'port': 8102,
        'endpoint': '/api/search/hybrid',
        'method': 'POST',
        'keywords': ['поиск', 'информация', 'найти', 'база знаний', 'документ', 'поиск информации'],
        'description': 'Поиск и информация'
    },
    'system_utils': {
        'port': 8103,
        'endpoint': '/api/system/utils',
        'method': 'POST',
        'keywords': ['система', 'статус', 'мониторинг', 'диагностика', 'утилиты', 'системные утилиты', 'проблемы', 'ошибки', 'здоровье системы', 'проблемы системы', 'системные ошибки', 'диагностика системы'],
        'description': 'Системные утилиты'
    },
    'gai': {
        'port': 8104,
        'endpoint': '/api/gai/generate_text',
        'method': 'POST',
        'keywords': ['сгенерировать', 'создать', 'написать', 'искусство', 'генерация', 'творчество'],
        'description': 'Генеративный ИИ'
    },
    'controllers': {
        'port': 9000,
        'endpoint': '/api/controllers/topic/general',
        'method': 'GET',
        'keywords': ['pmac', 'контроллер', 'автоматизация', 'движение', 'оси', 'позиционирование', 'траектория', 'servo', 'step', 'промышленная автоматизация', 'сервопривод', 'серводвигатель', 'привод', 'двигатель'],
        'description': 'Контроллеры и промышленная автоматизация'
    },
    'unified_manager': {
        'port': 8084,
        'endpoint': '/api/system/status',
        'method': 'GET',
        'keywords': ['управление', 'серверы', 'модули', 'статус системы', 'мониторинг системы'],
        'description': 'Управление системой'
    },
    'logic_tasks': {
        'port': 8106,
        'endpoint': '/api/logic/chat',
        'method': 'POST',
        'keywords': ['логика', 'логическая задача', 'доказательство', 'правила', 'аргумент', 'логическое рассуждение', 'задача', 'решить задачу'],
        'description': 'Логические задачи и рассуждения'
    }
}

# Статистика работы
STATS = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'server_stats': {},
    'start_time': datetime.now()
}

def categorize_message(message: str) -> str:
    """🧠 Интеллектуальная категоризация сообщения с весовыми коэффициентами"""
    lower_message = message.lower()
    
    # Весовые коэффициенты для более точной категоризации
    keyword_weights = {
        # Высокий приоритет - конкретные технические термины
        'high_priority': {
            'транзистор': 3, 'резистор': 3, 'конденсатор': 3, 'диод': 3,
            'pmac': 3, 'plc': 3, 'сервопривод': 3, 'серводвигатель': 3,
            'антенна': 3, 'передатчик': 3, 'приемник': 3,
            'python': 3, 'java': 3, 'c++': 3, 'javascript': 3,
            'интеграл': 3, 'производная': 3, 'система уравнений': 3
        },
        # Средний приоритет - общие технические термины
        'medium_priority': {
            'схема': 2, 'электричество': 2, 'ток': 2, 'напряжение': 2,
            'контроллер': 2, 'автоматизация': 2, 'двигатель': 2,
            'сигнал': 2, 'радио': 2, 'частота': 2,
            'код': 2, 'алгоритм': 2, 'программирование': 2,
            'уравнение': 2, 'математика': 2, 'вычислить': 2
        }
    }
    
    scores = {name: 0 for name in SERVERS}
    
    # Интеллектуальный подсчет с весовыми коэффициентами
    for server_name, config in SERVERS.items():
        for keyword in config['keywords']:
            if keyword in lower_message:
                # Определяем вес ключевого слова
                weight = 1  # Базовый вес
                for priority_level, keywords in keyword_weights.items():
                    if keyword in keywords:
                        weight = keywords[keyword]
                        break
                
                scores[server_name] += weight
    
    # Специальная логика для логических задач
    if any(word in lower_message for word in ['логическая задача', 'логика', 'доказательство', 'логическое рассуждение']):
        logger.info(f"🧠 Логическая задача обнаружена: '{message[:30]}...' → logic_tasks")
        return 'logic_tasks'
    
    # Приоритет для системных утилит
    if any(word in lower_message for word in ['проблемы', 'ошибки', 'здоровье системы', 'системные ошибки', 'диагностика системы']):
        logger.info(f"🛠️ Системная проблема обнаружена: '{message[:30]}...' → system_utils")
        return 'system_utils'
    
    # Приоритет для контроллеров
    if any(word in lower_message for word in ['pmac', 'plc', 'контроллер', 'сервопривод', 'серводвигатель', 'привод', 'двигатель']):
        logger.info(f"🎮 Контроллер обнаружен: '{message[:30]}...' → controllers")
        return 'controllers'
    
    # Интеллектуальный выбор на основе весов
    if scores:
        best_server = max(scores, key=scores.get)
        max_score = scores[best_server]
        
        # Дополнительная проверка для неоднозначности
        if max_score > 0:
            confidence = max_score / len(lower_message.split()) if lower_message.split() else 0
            logger.info(f"🧠 Интеллектуальная категоризация: '{message[:30]}...' → {best_server} (вес: {max_score}, уверенность: {confidence:.2f})")
            return best_server
    
    # Fallback на general с интеллектуальным анализом
    logger.info(f"🤔 Неопределенная категория: '{message[:30]}...' → general")
    return 'general'

def forward_request(category: str, message: str):
    """Перенаправляет запрос соответствующему серверу."""
    server_config = SERVERS.get(category)
    if not server_config:
        logger.error(f"❌ Неизвестная категория: {category}")
        return {"error": "Неизвестная категория", "category": category}, 400

    port = server_config['port']
    endpoint = server_config['endpoint']
    url = f"http://localhost:{port}{endpoint}"
    
    logger.info(f"🌐 Отправляем запрос к {category} на {url}")

    # Формирование payload в зависимости от типа сервера
    payload = {}
    if category in ['electrical', 'programming', 'general', 'mathematics', 'neuro', 'radiomechanics', 'plc_analysis', 'advanced_math', 'data_processing', 'search_engine', 'system_utils', 'gai', 'logic_tasks']:
        payload = {'message': message}
    elif category == 'controllers':
        # Определяем тему на основе сообщения
        if 'pmac' in message.lower():
            payload = {'topic': 'pmac'}
        elif 'plc' in message.lower():
            payload = {'topic': 'plc'}
        elif 'сервопривод' in message.lower() or 'servo' in message.lower():
            payload = {'topic': 'сервопривод'}
        elif 'энкодер' in message.lower():
            payload = {'topic': 'энкодер'}
        elif 'чпу' in message.lower():
            payload = {'topic': 'чпу'}
        elif 'pid' in message.lower():
            payload = {'topic': 'pid'}
        else:
            payload = {'topic': 'pmac'}  # По умолчанию PMAC
    elif category == 'unified_manager':
        payload = {'command': message}

    try:
        # Определение метода запроса на основе конфигурации
        method = server_config.get('method', 'POST')  # По умолчанию POST
        
        if method == 'GET':
            # Для GET запросов добавляем параметры в URL
            if payload:
                params = payload
                response = requests.get(url, params=params, timeout=10)
            else:
                response = requests.get(url, timeout=10)
        else:  # POST requests
            response = requests.post(url, json=payload, timeout=10)
        
        response.raise_for_status()
        logger.info(f"✅ Сервер {category} ответил успешно")
        
        # Обновление статистики
        STATS['successful_requests'] += 1
        if category not in STATS['server_stats']:
            STATS['server_stats'][category] = {'requests': 0, 'success': 0}
        STATS['server_stats'][category]['requests'] += 1
        STATS['server_stats'][category]['success'] += 1
        
        return response.json(), 200
        
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Ошибка соединения с {category}: Сервер недоступен")
        STATS['failed_requests'] += 1
        return {"error": f"Сервер {category} недоступен", "category": category}, 503
        
    except requests.exceptions.Timeout:
        logger.error(f"⏰ Таймаут при соединении с {category}")
        STATS['failed_requests'] += 1
        return {"error": f"Таймаут при соединении с {category}", "category": category}, 504
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP ошибка от {category}: {e.response.status_code} - {e.response.text}")
        STATS['failed_requests'] += 1
        return {"error": f"HTTP ошибка от {category}: {e.response.status_code}", "details": e.response.text, "category": category}, e.response.status_code
        
    except Exception as e:
        logger.error(f"❌ Неизвестная ошибка при запросе к {category}: {e}")
        STATS['failed_requests'] += 1
        return {"error": f"Неизвестная ошибка при запросе к {category}", "details": str(e), "category": category}, 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной эндпоинт для чата."""
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "Сообщение не может быть пустым"}), 400
    
    STATS['total_requests'] += 1
    logger.info(f"Вы: {message[:50]}...")
    
    # 1. Категоризация сообщения
    category = categorize_message(message)
    logger.info(f"📊 Категоризация: '{message[:50]}...' → {category}")
    
    # 2. Перенаправление запроса
    response_data, status_code = forward_request(category, message)
    
    # Добавляем информацию о категоризации в ответ
    if isinstance(response_data, dict):
        response_data['category'] = category
        response_data['server_description'] = SERVERS.get(category, {}).get('description', 'Неизвестный сервер')
    
    return jsonify(response_data), status_code

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья диспетчера."""
    uptime = datetime.now() - STATS['start_time']
    success_rate = (STATS['successful_requests'] / STATS['total_requests'] * 100) if STATS['total_requests'] > 0 else 0
    
    return jsonify({
        'status': 'healthy',
        'message': 'Simple Dispatcher работает стабильно',
        'uptime': str(uptime),
        'total_requests': STATS['total_requests'],
        'successful_requests': STATS['successful_requests'],
        'failed_requests': STATS['failed_requests'],
        'success_rate': f"{success_rate:.2f}%",
        'servers_count': len(SERVERS)
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Получение статистики работы."""
    uptime = datetime.now() - STATS['start_time']
    
    return jsonify({
        'uptime': str(uptime),
        'total_requests': STATS['total_requests'],
        'successful_requests': STATS['successful_requests'],
        'failed_requests': STATS['failed_requests'],
        'server_stats': STATS['server_stats'],
        'servers': {name: config['description'] for name, config in SERVERS.items()}
    }), 200

@app.route('/api/servers', methods=['GET'])
def get_servers():
    """Получение списка доступных серверов."""
    return jsonify({
        'servers': {name: {
            'port': config['port'],
            'endpoint': config['endpoint'],
            'description': config['description'],
            'keywords': config['keywords'][:5]  # Показываем только первые 5 ключевых слов
        } for name, config in SERVERS.items()}
    }), 200

@app.route('/matrix/RubinDeveloper.html', methods=['GET'])
def serve_rubin_developer():
    """Обслуживание RubinDeveloper интерфейса."""
    return send_from_directory('matrix', 'RubinDeveloper.html')

@app.route('/', methods=['GET'])
def index():
    """Главная страница с информацией о диспетчере."""
    return jsonify({
        'message': 'Simple Smart Dispatcher для Rubin AI v2',
        'version': '2.0-simple',
        'status': 'running',
        'endpoints': {
            'POST /api/chat': 'Основной чат',
            'GET /api/health': 'Проверка здоровья',
            'GET /api/stats': 'Статистика',
            'GET /api/servers': 'Список серверов',
            'GET /matrix/RubinDeveloper.html': 'Интерфейс разработчика'
        },
        'servers_count': len(SERVERS)
    }), 200

if __name__ == '__main__':
    print("🚀 Simple Smart Dispatcher запущен")
    print("=" * 50)
    print(f"📊 Доступно серверов: {len(SERVERS)}")
    print(f"🌐 URL: http://localhost:8080")
    print(f"📱 RubinDeveloper: http://localhost:8080/matrix/RubinDeveloper.html")
    print("=" * 50)
    
    # Отключаем reloader для стабильности
    app.run(port=8080, debug=False, use_reloader=False)