#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified System Manager
Единая система управления всеми функциями Rubin AI v2
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import logging
from datetime import datetime
import os
import threading
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Конфигурация всех серверов системы
SYSTEM_SERVERS = {
    # Основные серверы
    'smart_dispatcher': {
        'port': 8080,
        'endpoint': '/api/health',
        'name': 'Smart Dispatcher',
        'description': 'Главный диспетчер запросов',
        'status': 'unknown'
    },
    'general_api': {
        'port': 8085,
        'endpoint': '/api/chat',
        'name': 'General API',
        'description': 'Общие вопросы и справка',
        'status': 'unknown'
    },
    'mathematics': {
        'port': 8086,
        'endpoint': '/api/chat',
        'name': 'Mathematics Server',
        'description': 'Математические вычисления',
        'status': 'unknown'
    },
    'electrical': {
        'port': 8087,
        'endpoint': '/api/electrical/health',
        'name': 'Electrical API',
        'description': 'Электротехнические расчеты',
        'status': 'unknown'
    },
    'programming': {
        'port': 8088,
        'endpoint': '/api/chat',
        'name': 'Programming API',
        'description': 'Программирование и алгоритмы',
        'status': 'unknown'
    },
    'radiomechanics': {
        'port': 8089,
        'endpoint': '/api/radiomechanics/health',
        'name': 'Radiomechanics API',
        'description': 'Радиомеханические расчеты',
        'status': 'unknown'
    },
    'neuro': {
        'port': 8090,
        'endpoint': '/api/neuro/health',
        'name': 'Neural Network API',
        'description': 'Нейронные сети и машинное обучение',
        'status': 'unknown'
    },
    'controllers': {
        'port': 9000,
        'endpoint': '/api/controllers/health',
        'name': 'Controllers API',
        'description': 'ПЛК, ЧПУ, автоматизация',
        'status': 'unknown'
    },
    
    # Новые приоритетные серверы
    'plc_analysis': {
        'port': 8099,
        'endpoint': '/api/plc/health',
        'name': 'PLC Analysis API',
        'description': 'Анализ и диагностика PLC программ',
        'status': 'unknown'
    },
    'advanced_math': {
        'port': 8100,
        'endpoint': '/api/math/health',
        'name': 'Advanced Mathematics API',
        'description': 'Продвинутые математические вычисления',
        'status': 'unknown'
    },
    'data_processing': {
        'port': 8101,
        'endpoint': '/api/data/health',
        'name': 'Data Processing API',
        'description': 'Обработка и анализ данных',
        'status': 'unknown'
    },
    'search_engine': {
        'port': 8102,
        'endpoint': '/api/search/health',
        'name': 'Search Engine API',
        'description': 'Гибридный поиск и индексация',
        'status': 'unknown'
    },
    'system_utils': {
        'port': 8103,
        'endpoint': '/api/system/health',
        'name': 'System Utils API',
        'description': 'Системные утилиты и диагностика',
        'status': 'unknown'
    }
}

# Статистика системы
SYSTEM_STATS = {
    'total_servers': len(SYSTEM_SERVERS),
    'online_servers': 0,
    'offline_servers': 0,
    'last_check': None,
    'uptime': None,
    'requests_processed': 0
}

def check_server_status(server_name, config):
    """Проверка статуса отдельного сервера"""
    try:
        url = f"http://localhost:{config['port']}{config['endpoint']}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            config['status'] = 'online'
            config['last_response'] = response.json()
            config['response_time'] = response.elapsed.total_seconds()
            return True
        else:
            config['status'] = 'error'
            config['error'] = f"HTTP {response.status_code}"
            return False
            
    except requests.exceptions.ConnectionError:
        config['status'] = 'offline'
        config['error'] = 'Connection refused'
        return False
    except requests.exceptions.Timeout:
        config['status'] = 'timeout'
        config['error'] = 'Request timeout'
        return False
    except Exception as e:
        config['status'] = 'error'
        config['error'] = str(e)
        return False

def check_all_servers():
    """Проверка статуса всех серверов"""
    online_count = 0
    offline_count = 0
    
    for server_name, config in SYSTEM_SERVERS.items():
        if check_server_status(server_name, config):
            online_count += 1
        else:
            offline_count += 1
    
    SYSTEM_STATS['online_servers'] = online_count
    SYSTEM_STATS['offline_servers'] = offline_count
    SYSTEM_STATS['last_check'] = datetime.now().isoformat()
    
    return {
        'online': online_count,
        'offline': offline_count,
        'total': len(SYSTEM_SERVERS)
    }

def start_server(server_name):
    """Запуск сервера"""
    server_configs = {
        'smart_dispatcher': 'python smart_dispatcher.py',
        'general_api': 'python api/general_api.py',
        'mathematics': 'python math_server.py',
        'electrical': 'python api/electrical_api.py',
        'programming': 'python api/programming_api.py',
        'radiomechanics': 'python api/radiomechanics_api.py',
        'neuro': 'python neuro_server.py',
        'controllers': 'python api/controllers_api.py',
        'plc_analysis': 'python plc_analysis_api_server.py',
        'advanced_math': 'python advanced_math_api_server.py',
        'data_processing': 'python data_processing_api_server.py',
        'search_engine': 'python search_engine_api_server.py',
        'system_utils': 'python system_utils_api_server.py'
    }
    
    if server_name in server_configs:
        try:
            import subprocess
            command = server_configs[server_name]
            process = subprocess.Popen(command, shell=True)
            
            # Даем серверу время запуститься
            time.sleep(3)
            
            # Проверяем статус
            if check_server_status(server_name, SYSTEM_SERVERS[server_name]):
                return {'success': True, 'message': f'Сервер {server_name} успешно запущен'}
            else:
                return {'success': False, 'message': f'Сервер {server_name} не отвечает после запуска'}
                
        except Exception as e:
            return {'success': False, 'message': f'Ошибка запуска сервера {server_name}: {str(e)}'}
    else:
        return {'success': False, 'message': f'Неизвестный сервер: {server_name}'}

def get_system_health():
    """Получение общего здоровья системы"""
    stats = check_all_servers()
    
    health_score = (stats['online'] / stats['total']) * 100
    
    if health_score >= 80:
        health_status = 'excellent'
    elif health_score >= 60:
        health_status = 'good'
    elif health_score >= 40:
        health_status = 'warning'
    else:
        health_status = 'critical'
    
    return {
        'status': health_status,
        'score': round(health_score, 2),
        'stats': stats,
        'servers': SYSTEM_SERVERS,
        'timestamp': datetime.now().isoformat()
    }

@app.route('/')
def index():
    """Главная страница системы управления"""
    return jsonify({
        'service': 'Unified System Manager',
        'status': 'running',
        'version': '1.0',
        'endpoints': [
            '/api/system/status',
            '/api/system/check',
            '/api/system/start/<server>',
            '/api/system/restart_all',
            '/api/system/health',
            '/api/system/servers',
            '/api/system/capabilities'
        ]
    })

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Статус всей системы"""
    try:
        health = get_system_health()
        return jsonify({
            'success': True,
            'system_health': health,
            'stats': SYSTEM_STATS
        })
    except Exception as e:
        logger.error(f"Ошибка получения статуса системы: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/check', methods=['POST'])
def check_system():
    """Проверка всех серверов"""
    try:
        stats = check_all_servers()
        return jsonify({
            'success': True,
            'stats': stats,
            'servers': SYSTEM_SERVERS,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Ошибка проверки системы: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/start/<server_name>', methods=['POST'])
def start_server_endpoint(server_name):
    """Запуск сервера"""
    try:
        result = start_server(server_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ошибка запуска сервера {server_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/restart_all', methods=['POST'])
def restart_all_servers():
    """Перезапуск всех серверов"""
    try:
        results = {}
        
        for server_name in SYSTEM_SERVERS.keys():
            result = start_server(server_name)
            results[server_name] = result
        
        # Проверяем статус после перезапуска
        time.sleep(5)
        final_stats = check_all_servers()
        
        return jsonify({
            'success': True,
            'restart_results': results,
            'final_stats': final_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Ошибка перезапуска серверов: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья Unified System Manager"""
    return jsonify({
        'service': 'Unified System Manager',
        'status': 'healthy',
        'port': 8084,
        'version': '1.0',
        'capabilities': ['system_monitoring', 'server_management', 'health_check']
    })

@app.route('/api/system/health', methods=['GET'])
def system_health():
    """Здоровье системы"""
    try:
        health = get_system_health()
        return jsonify({
            'success': True,
            'health': health
        })
    except Exception as e:
        logger.error(f"Ошибка получения здоровья системы: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/servers', methods=['GET'])
def get_servers():
    """Получение списка всех серверов"""
    try:
        return jsonify({
            'success': True,
            'servers': SYSTEM_SERVERS,
            'total_count': len(SYSTEM_SERVERS),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Ошибка получения списка серверов: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/capabilities', methods=['GET'])
def get_capabilities():
    """Получение возможностей системы"""
    try:
        capabilities = {
            'mathematics': [
                'Решение уравнений',
                'Вычисления',
                'Графики функций',
                'Статистика'
            ],
            'electrical': [
                'Закон Ома',
                'Законы Кирхгофа',
                'Расчет цепей',
                'Электротехнические формулы'
            ],
            'programming': [
                'Анализ кода',
                'Алгоритмы',
                'Отладка',
                'Оптимизация'
            ],
            'plc_analysis': [
                'Анализ PLC программ',
                'Диагностика ошибок',
                'Статистика программы',
                'База знаний PLC'
            ],
            'data_processing': [
                'Предобработка данных',
                'Анализ временных рядов',
                'Нормализация',
                'Корреляционный анализ'
            ],
            'search_engine': [
                'Гибридный поиск',
                'Семантический поиск',
                'Индексация документов',
                'Поиск в базе знаний'
            ],
            'system_utils': [
                'Мониторинг системы',
                'Диагностика',
                'Очистка',
                'Резервное копирование'
            ]
        }
        
        return jsonify({
            'success': True,
            'capabilities': capabilities,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Ошибка получения возможностей: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def background_monitor():
    """Фоновый мониторинг системы"""
    while True:
        try:
            check_all_servers()
            time.sleep(30)  # Проверка каждые 30 секунд
        except Exception as e:
            logger.error(f"Ошибка фонового мониторинга: {e}")
            time.sleep(60)  # При ошибке ждем минуту

if __name__ == '__main__':
    print("🎛️ Unified System Manager запущен")
    print("URL: http://localhost:8084")
    print("Endpoints:")
    print("  - GET /api/system/status - Статус системы")
    print("  - POST /api/system/check - Проверка серверов")
    print("  - POST /api/system/start/<server> - Запуск сервера")
    print("  - POST /api/system/restart_all - Перезапуск всех серверов")
    print("  - GET /api/system/health - Здоровье системы")
    print("  - GET /api/system/servers - Список серверов")
    print("  - GET /api/system/capabilities - Возможности системы")
    
    # Запускаем фоновый мониторинг
    monitor_thread = threading.Thread(target=background_monitor, daemon=True)
    monitor_thread.start()
    
    app.run(host='0.0.0.0', port=8084, debug=False)
