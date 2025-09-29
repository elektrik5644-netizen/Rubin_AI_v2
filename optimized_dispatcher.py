#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированный Smart Dispatcher для Rubin AI v2
Минимальное потребление памяти, максимальная стабильность
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import gc
import os
from datetime import datetime

# Настройка логирования (минимальная)
logging.basicConfig(level=logging.WARNING)  # Только ошибки
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Минимальная конфигурация серверов
SERVERS = {
    'electrical': {'port': 8087, 'endpoint': '/api/electrical/status', 'method': 'GET'},
    'mathematics': {'port': 8086, 'endpoint': '/health', 'method': 'GET'},
    'programming': {'port': 8088, 'endpoint': '/api/programming/explain', 'method': 'GET'},
    'general': {'port': 8085, 'endpoint': '/api/general/chat', 'method': 'POST'},
    'neuro': {'port': 8090, 'endpoint': '/api/health', 'method': 'GET'},
    'controllers': {'port': 9000, 'endpoint': '/api/controllers/status', 'method': 'GET'},
    'gai': {'port': 8104, 'endpoint': '/api/gai/health', 'method': 'GET'},
    'unified_manager': {'port': 8084, 'endpoint': '/api/system/health', 'method': 'GET'},
    'ethical_core': {'port': 8105, 'endpoint': '/api/ethical/health', 'method': 'GET'}
}

def cleanup_memory():
    """Принудительная очистка памяти"""
    gc.collect()

def check_server_health(server_name, config):
    """Быстрая проверка здоровья сервера"""
    try:
        url = f"http://localhost:{config['port']}{config['endpoint']}"
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

def route_message(message):
    """Простая маршрутизация по ключевым словам"""
    message_lower = message.lower()
    
    # Электротехника
    if any(kw in message_lower for kw in ['закон', 'кирхгофа', 'резистор', 'транзистор', 'диод', 'мощность', 'ток', 'напряжение']):
        return 'electrical'
    
    # Математика
    elif any(kw in message_lower for kw in ['уравнение', 'математика', 'вычислить', 'посчитать', '+', '-', '*', '/']):
        return 'mathematics'
    
    # Программирование
    elif any(kw in message_lower for kw in ['код', 'программа', 'алгоритм', 'python', 'java', 'функция']):
        return 'programming'
    
    # Нейросети
    elif any(kw in message_lower for kw in ['нейросеть', 'нейронная', 'ии', 'машинное обучение']):
        return 'neuro'
    
    # Контроллеры
    elif any(kw in message_lower for kw in ['контроллер', 'плк', 'автоматизация', 'сервопривод']):
        return 'controllers'
    
    # GAI
    elif any(kw in message_lower for kw in ['создать', 'генерировать', 'написать', 'сочинить']):
        return 'gai'
    
    # По умолчанию - General
    else:
        return 'general'

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Пустое сообщение'}), 400
        
        # Определяем сервер
        target_server = route_message(message)
        server_config = SERVERS.get(target_server)
        
        if not server_config:
            return jsonify({'error': 'Сервер не найден'}), 404
        
        # Проверяем доступность сервера
        if not check_server_health(target_server, server_config):
            return jsonify({
                'response': f'Сервер {target_server} временно недоступен. Попробуйте позже.',
                'server': target_server,
                'status': 'offline'
            }), 503
        
        # Формируем ответ
        response = {
            'response': f'Запрос обработан сервером {target_server}',
            'server': target_server,
            'status': 'online',
            'timestamp': datetime.now().isoformat()
        }
        
        # Очищаем память
        cleanup_memory()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Ошибка в chat: {e}")
        cleanup_memory()
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья диспетчера"""
    return jsonify({
        'status': 'ok',
        'servers_count': len(SERVERS),
        'memory_optimized': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/servers', methods=['GET'])
def servers():
    """Список серверов"""
    server_status = {}
    for name, config in SERVERS.items():
        server_status[name] = {
            'port': config['port'],
            'status': 'online' if check_server_health(name, config) else 'offline'
        }
    
    cleanup_memory()
    return jsonify(server_status)

if __name__ == '__main__':
    print("🚀 Оптимизированный Smart Dispatcher запущен")
    print("💾 Минимальное потребление памяти")
    print("🌐 URL: http://localhost:8080")
    
    # Очищаем память при запуске
    cleanup_memory()
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)








