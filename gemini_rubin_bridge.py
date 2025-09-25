#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мост Gemini-Rubin для обучения Rubin AI
Позволяет Gemini взаимодействовать с Rubin для обучения и анализа
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import json
import time
from datetime import datetime
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app)

# Конфигурация Rubin AI
RUBIN_SMART_DISPATCHER_URL = "http://localhost:8080"
RUBIN_MODULES = {
    'general': 'http://localhost:8085',
    'mathematics': 'http://localhost:8086', 
    'electrical': 'http://localhost:8087',
    'programming': 'http://localhost:8088',
    'neuro': 'http://localhost:8090',
    'controllers': 'http://localhost:9000',
    'gai': 'http://localhost:8104'
}

# Хранилище сессий обучения
LEARNING_SESSIONS = {}
LEARNING_STATS = {
    'total_sessions': 0,
    'total_interactions': 0,
    'successful_teachings': 0,
    'failed_teachings': 0,
    'categories_taught': {}
}

def create_session_id():
    """Создает уникальный ID сессии"""
    return f"gemini_session_{int(time.time())}"

def log_interaction(session_id, interaction_type, data, result):
    """Логирует взаимодействие Gemini с Rubin"""
    if session_id not in LEARNING_SESSIONS:
        LEARNING_SESSIONS[session_id] = {
            'created_at': datetime.now().isoformat(),
            'interactions': [],
            'status': 'active'
        }
    
    interaction = {
        'timestamp': datetime.now().isoformat(),
        'type': interaction_type,
        'data': data,
        'result': result
    }
    
    LEARNING_SESSIONS[session_id]['interactions'].append(interaction)
    LEARNING_STATS['total_interactions'] += 1

def check_rubin_health():
    """Проверяет доступность Rubin AI"""
    try:
        response = requests.get(f"{RUBIN_SMART_DISPATCHER_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_to_rubin(message, category='general'):
    """Отправляет сообщение в Rubin AI"""
    try:
        payload = {
            'message': message,
            'user_id': 'gemini_bridge'
        }
        
        response = requests.post(
            f"{RUBIN_SMART_DISPATCHER_URL}/api/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'HTTP {response.status_code}'}
            
    except Exception as e:
        return {'error': str(e)}

@app.route('/api/gemini/status', methods=['GET'])
def gemini_status():
    """Статус моста Gemini-Rubin"""
    rubin_healthy = check_rubin_health()
    
    return jsonify({
        'status': 'success',
        'bridge': 'Gemini-Rubin Bridge',
        'version': '1.0',
        'rubin_ai_status': 'healthy' if rubin_healthy else 'unreachable',
        'rubin_url': RUBIN_SMART_DISPATCHER_URL,
        'active_sessions': len(LEARNING_SESSIONS),
        'total_interactions': LEARNING_STATS['total_interactions']
    })

@app.route('/api/gemini/teach', methods=['POST'])
def teach_rubin():
    """Обучение Rubin от Gemini"""
    try:
        data = request.get_json()
        instruction = data.get('instruction', '')
        context = data.get('context', 'general')
        
        if not instruction:
            return jsonify({
                'status': 'error',
                'message': 'Инструкция не может быть пустой'
            }), 400
        
        session_id = create_session_id()
        
        # Формируем обучающее сообщение для Rubin
        teaching_message = f"""
[ОБУЧЕНИЕ ОТ GEMINI]
Контекст: {context}
Инструкция: {instruction}

Пожалуйста, примени эту инструкцию к своим будущим ответам.
"""
        
        # Отправляем в Rubin
        rubin_response = send_to_rubin(teaching_message, context)
        
        if 'error' in rubin_response:
            LEARNING_STATS['failed_teachings'] += 1
            log_interaction(session_id, 'teach_failed', {
                'instruction': instruction,
                'context': context
            }, rubin_response)
            
            return jsonify({
                'status': 'error',
                'message': f'Ошибка обучения Rubin: {rubin_response["error"]}'
            }), 500
        
        LEARNING_STATS['successful_teachings'] += 1
        LEARNING_STATS['total_sessions'] += 1
        
        if context not in LEARNING_STATS['categories_taught']:
            LEARNING_STATS['categories_taught'][context] = 0
        LEARNING_STATS['categories_taught'][context] += 1
        
        log_interaction(session_id, 'teach_success', {
            'instruction': instruction,
            'context': context
        }, rubin_response)
        
        return jsonify({
            'status': 'success',
            'message': 'Обучение успешно передано Rubin',
            'session_id': session_id,
            'rubin_ai_response': rubin_response.get('response', 'OK'),
            'context': context
        })
        
    except Exception as e:
        logger.error(f"Ошибка в teach_rubin: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gemini/analyze', methods=['POST'])
def analyze_rubin():
    """Анализ Rubin AI от Gemini"""
    try:
        data = request.get_json()
        analysis_type = data.get('type', 'general')
        query = data.get('query', '')
        
        session_id = create_session_id()
        
        # Формируем аналитический запрос
        analysis_message = f"""
[АНАЛИЗ ОТ GEMINI]
Тип анализа: {analysis_type}
Запрос: {query}

Пожалуйста, предоставь детальный анализ по этому запросу.
"""
        
        # Отправляем в Rubin
        rubin_response = send_to_rubin(analysis_message)
        
        if 'error' in rubin_response:
            log_interaction(session_id, 'analyze_failed', {
                'type': analysis_type,
                'query': query
            }, rubin_response)
            
            return jsonify({
                'status': 'error',
                'message': f'Ошибка анализа Rubin: {rubin_response["error"]}'
            }), 500
        
        log_interaction(session_id, 'analyze_success', {
            'type': analysis_type,
            'query': query
        }, rubin_response)
        
        return jsonify({
            'status': 'success',
            'analysis_type': analysis_type,
            'rubin_ai_response': rubin_response.get('response', ''),
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Ошибка в analyze_rubin: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gemini/feedback', methods=['POST'])
def provide_feedback():
    """Предоставление обратной связи от Gemini"""
    try:
        data = request.get_json()
        feedback_type = data.get('type', 'improvement')
        feedback_content = data.get('content', '')
        target_category = data.get('category', 'general')
        
        session_id = create_session_id()
        
        # Формируем сообщение обратной связи
        feedback_message = f"""
[ОБРАТНАЯ СВЯЗЬ ОТ GEMINI]
Тип: {feedback_type}
Категория: {target_category}
Содержание: {feedback_content}

Пожалуйста, учти эту обратную связь для улучшения своих ответов.
"""
        
        # Отправляем в Rubin
        rubin_response = send_to_rubin(feedback_message, target_category)
        
        if 'error' in rubin_response:
            log_interaction(session_id, 'feedback_failed', {
                'type': feedback_type,
                'content': feedback_content,
                'category': target_category
            }, rubin_response)
            
            return jsonify({
                'status': 'error',
                'message': f'Ошибка обратной связи: {rubin_response["error"]}'
            }), 500
        
        log_interaction(session_id, 'feedback_success', {
            'type': feedback_type,
            'content': feedback_content,
            'category': target_category
        }, rubin_response)
        
        return jsonify({
            'status': 'success',
            'message': 'Обратная связь успешно передана Rubin',
            'session_id': session_id,
            'rubin_ai_response': rubin_response.get('response', 'OK')
        })
        
    except Exception as e:
        logger.error(f"Ошибка в provide_feedback: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gemini/sessions', methods=['GET'])
def get_sessions():
    """Получение информации о сессиях обучения"""
    return jsonify({
        'status': 'success',
        'total_sessions': LEARNING_STATS['total_sessions'],
        'active_sessions': len(LEARNING_SESSIONS),
        'learning_sessions': LEARNING_SESSIONS,
        'statistics': LEARNING_STATS
    })

@app.route('/api/gemini/health', methods=['GET'])
def health():
    """Проверка здоровья моста"""
    rubin_healthy = check_rubin_health()
    
    return jsonify({
        'service': 'Gemini-Rubin Bridge',
        'status': 'healthy' if rubin_healthy else 'degraded',
        'port': 8082,
        'version': '1.0',
        'rubin_ai_accessible': rubin_healthy,
        'active_sessions': len(LEARNING_SESSIONS),
        'total_interactions': LEARNING_STATS['total_interactions']
    })

@app.route('/api/gemini/test', methods=['POST'])
def test_connection():
    """Тестирование соединения с Rubin AI"""
    try:
        test_message = "Тестовое сообщение от моста Gemini-Rubin"
        rubin_response = send_to_rubin(test_message)
        
        if 'error' in rubin_response:
            return jsonify({
                'status': 'error',
                'message': f'Ошибка соединения: {rubin_response["error"]}'
            }), 500
        
        return jsonify({
            'status': 'success',
            'message': 'Соединение с Rubin AI работает',
            'rubin_response': rubin_response.get('response', 'OK')
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("🌉 Запуск моста Gemini-Rubin...")
    print("📡 Порт: 8082")
    print("🔗 URL: http://localhost:8082")
    print("📋 Эндпоинты:")
    print("  - POST /api/gemini/teach - Обучение Rubin")
    print("  - POST /api/gemini/analyze - Анализ Rubin")
    print("  - POST /api/gemini/feedback - Обратная связь")
    print("  - GET /api/gemini/sessions - Сессии обучения")
    print("  - GET /api/gemini/status - Статус моста")
    print("  - GET /api/gemini/health - Проверка здоровья")
    print("  - POST /api/gemini/test - Тест соединения")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8082, debug=False)

