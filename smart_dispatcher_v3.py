#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 SMART DISPATCHER V3.0 - С ПОДДЕРЖКОЙ МОДУЛЯ ОБУЧЕНИЯ
======================================================
Улучшенный диспетчер с контекстной памятью и приоритетной маршрутизацией
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import logging
import os
from datetime import datetime
from typing import Dict, List, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Конфигурация серверов с приоритетами
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
        'priority': 5,
        'fallback': 'general',
        'fallback_keywords': ['радио', 'сигнал', 'антенна']
    },
    'controllers': {
        'port': 9000,
        'endpoint': '/api/controllers/topic/general',
        'keywords': ['пид', 'регулятор', 'plc', 'контроллер', 'автоматизация', 'логика', 'события', 'прерывания', 'events', 'interrupts', 'ascii', 'команды', 'протокол', 'отправка', 'получение', 'ответы', 'чпу', 'cnc', 'числовое', 'программное', 'управление', 'передача', 'данные', 'g-коды', 'координаты', 'pmac', 'многоосевой', 'движение', 'позиционирование', 'траектория', 'ось', 'оси', 'серводвигатель', 'шаговый', 'энкодер', 'обратная связь', 'сервопривод', 'настроить', 'настройка', 'конфигурация', 'параметры'],
        'priority': 5,
        'fallback': 'programming',
        'fallback_keywords': ['plc', 'контроллер', 'автоматизация', 'программирование', 'логика', 'управление', 'ошибка', 'анализ', 'файл']
    },
    'mathematics': {
        'port': 8086,
        'endpoint': '/api/chat',
        'keywords': ['уравнение', 'квадратное', 'математика', 'алгебра', 'геометрия', 'арифметика', '+', '-', '*', '/', '=', 'вычислить', 'посчитать', 'сложить', 'вычесть', 'умножить', 'разделить', 'число', 'цифра', 'результат', 'ответ'],
        'priority': 5,
        'fallback': 'general',
        'fallback_keywords': ['математика', 'вычислить', 'посчитать']
    },
    'programming': {
        'port': 8088,
        'endpoint': '/api/programming/explain',
        'keywords': ['код', 'алгоритм', 'python', 'программирование', 'функция', 'класс', 'переменная', 'цикл', 'условие', 'массив', 'список', 'словарь', 'объект', 'метод', 'библиотека', 'модуль', 'импорт', 'синтаксис', 'ошибка', 'отладка', 'тестирование', 'разработка', 'архитектура', 'паттерн', 'singleton', 'observer', 'factory', 'strategy', 'command'],
        'priority': 5,
        'fallback': 'general',
        'fallback_keywords': ['код', 'алгоритм', 'программирование']
    },
    'pytorch': {
        'port': 8092,
        'endpoint': '/api/pytorch/chat',
        'keywords': ['pytorch', 'torch', 'нейросеть', 'нейронная сеть', 'машинное обучение', 'ml', 'ai', 'градиент', 'backward', 'optimizer', 'cuda', 'gpu', 'device', 'tensor', 'модель', 'обучение модели', 'сохранение модели', 'загрузка модели', 'train', 'eval', 'state_dict', 'zero_grad', 'step', 'loss', 'criterion', 'batch', 'epoch', 'dropout', 'batchnorm', 'cnn', 'rnn', 'lstm', 'transformer', 'gan', 'reinforcement learning'],
        'priority': 8,  # Высокий приоритет для PyTorch вопросов
        'fallback': 'programming',
        'fallback_keywords': ['нейросеть', 'машинное обучение', 'ai']
    },
    'general': {
        'port': 8085,
        'endpoint': '/api/chat',
        'keywords': ['общий', 'вопрос', 'помощь', 'справка', 'информация'],
        'priority': 1,  # Минимальный приоритет для общих вопросов
        'fallback': None,
        'fallback_keywords': []
    }
}

# Контекстная память
CONTEXT_MEMORY = {
    "session_start": datetime.now().isoformat(),
    "interaction_history": [],
    "learning_context": {
        "today_activities": [
            "Создали систему постоянного сканирования Rubin AI",
            "Исправили HTTP 500 ошибки через fallback механизм в Smart Dispatcher",
            "Модернизировали VMB630 с паттернами проектирования",
            "Проанализировали PLC файл и нашли ошибки",
            "Создали систему автоматического исправления PLC ошибок",
            "Обучили Rubin AI пониманию процессов диагностики и модернизации"
        ],
        "current_focus": "context_understanding"
    }
}

def categorize_message(message: str) -> str:
    """Улучшенная категоризация с приоритетами"""
    message_lower = message.lower()
    
    # Подсчитываем совпадения для каждого сервера с учетом приоритета
    server_scores = {}
    
    for server_name, config in SERVERS.items():
        score = 0
        keywords = config['keywords']
        priority = config.get('priority', 1)
        
        # Подсчитываем совпадения ключевых слов
        for keyword in keywords:
            if keyword in message_lower:
                score += 1
        
        # Применяем приоритет (чем выше приоритет, тем больше вес)
        weighted_score = score * priority
        server_scores[server_name] = weighted_score
    
    # Находим сервер с максимальным счетом
    if server_scores:
        best_server = max(server_scores, key=server_scores.get)
        best_score = server_scores[best_server]
        
        # Логируем категоризацию
        logger.info(f"📊 Категоризация: '{message[:50]}...' → {best_server} (score: {best_score})")
        
        return best_server
    
    return 'general'

def add_context_to_message(message: str) -> str:
    """Добавляем контекст к сообщению"""
    context_hint = f"""
КОНТЕКСТ СЕССИИ:
- Сегодня мы работали над диагностикой ошибок
- Исправляли HTTP 500 через fallback механизмы  
- Модернизировали VMB630 с паттернами проектирования
- Анализировали PLC файлы
- Обучали Rubin AI пониманию процессов

ВАЖНО: Отвечай конкретно о нашем взаимодействии, НЕ давай шаблонные ответы!
"""
    
    return context_hint + "\n\n" + message

def forward_request(message: str, category: str) -> Dict[str, Any]:
    """Улучшенная пересылка запросов с контекстом"""
    if category not in SERVERS:
        logger.error(f"❌ Неизвестная категория: {category}")
        return {
            'success': False,
            'error': f'Неизвестная категория: {category}'
        }
    
    server_config = SERVERS[category]
    port = server_config['port']
    endpoint = server_config['endpoint']
    
    # Добавляем контекст к сообщению для модуля обучения
    if category == 'learning':
        contextual_message = add_context_to_message(message)
    else:
        contextual_message = message
    
    url = f'http://localhost:{port}{endpoint}'
    
    logger.info(f"🌐 Отправляем запрос к {category} на {url}")
    
    try:
        # Проверяем доступность сервера
        health_url = f'http://localhost:{port}/api/health'
        try:
            health_response = requests.get(health_url, timeout=2)
            if health_response.status_code != 200:
                raise requests.exceptions.RequestException("Сервер недоступен")
        except:
            # Если нет health endpoint, продолжаем
            pass
        
        # Отправляем основной запрос
        response = requests.post(url, 
                               json={'message': contextual_message}, 
                               timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ Сервер {category} ответил успешно")
            return response.json()
        else:
            logger.error(f"❌ Сервер {category} вернул статус {response.status_code}")
            return {
                'success': False,
                'error': f'Сервер {category} вернул статус {response.status_code}'
            }
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Ошибка соединения с {category}: {e}")
        
        # Применяем fallback механизм
        fallback = server_config.get('fallback')
        if fallback and fallback in SERVERS:
            logger.info(f"🔄 Fallback: отправляем запрос к {fallback}")
            return forward_request(message, fallback)
        else:
            return {
                'success': False,
                'error': f'Ошибка соединения с {category} и нет fallback'
            }
    
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при обращении к {category}: {e}")
        return {
            'success': False,
            'error': f'Неожиданная ошибка: {str(e)}'
        }

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной эндпоинт для чата"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Сообщение не может быть пустым'
            }), 400
        
        # Сохраняем взаимодействие в контекстную память
        CONTEXT_MEMORY["interaction_history"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "category": None  # Будет определено ниже
        })
        
        # Категоризируем сообщение
        category = categorize_message(message)
        
        # Обновляем категорию в истории
        CONTEXT_MEMORY["interaction_history"][-1]["category"] = category
        
        # Пересылаем запрос
        result = forward_request(message, category)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья диспетчера"""
    return jsonify({
        'status': 'healthy',
        'version': '3.0',
        'timestamp': datetime.now().isoformat(),
        'servers': {name: f"localhost:{config['port']}" for name, config in SERVERS.items()},
        'context_memory': len(CONTEXT_MEMORY["interaction_history"])
    })

@app.route('/api/context', methods=['GET'])
def get_context():
    """Получение контекстной памяти"""
    return jsonify({
        'success': True,
        'context': CONTEXT_MEMORY
    })

@app.route('/api/servers/status', methods=['GET'])
def servers_status():
    """Статус всех серверов"""
    status = {}
    
    for server_name, config in SERVERS.items():
        port = config['port']
        try:
            health_url = f'http://localhost:{port}/api/health'
            response = requests.get(health_url, timeout=2)
            status[server_name] = {
                'status': 'online' if response.status_code == 200 else 'error',
                'port': port,
                'priority': config.get('priority', 1)
            }
        except:
            status[server_name] = {
                'status': 'offline',
                'port': port,
                'priority': config.get('priority', 1)
            }
    
    return jsonify({
        'success': True,
        'servers': status
    })

if __name__ == '__main__':
    print("🚀 Smart Dispatcher v3.0 запущен")
    print("URL: http://localhost:8080")
    print("Доступные серверы:")
    for name, config in SERVERS.items():
        priority = config.get('priority', 1)
        print(f"  - {name}: localhost:{config['port']} (приоритет: {priority})")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
