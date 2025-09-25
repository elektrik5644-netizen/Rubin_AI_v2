#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер с интегрированной аналитикой ответов
Предоставляет чат с автоматическим анализом и улучшением качества ответов
"""

from flask import Flask, request, jsonify
import logging
import json
from datetime import datetime
from intelligent_chat_analytics import get_intelligent_chat
from response_analytics import get_quality_controller

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Инициализация компонентов
intelligent_chat = get_intelligent_chat()
quality_controller = get_quality_controller()

@app.route('/api/intelligent/chat', methods=['POST'])
def intelligent_chat_endpoint():
    """Основной endpoint для интеллектуального чата с аналитикой"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'default')
        enable_analytics = data.get('enable_analytics', True)
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        logger.info(f"💬 Получено сообщение от {user_id}: {message[:50]}...")
        
        # Обрабатываем через интеллектуальный чат
        result = intelligent_chat.process_question(message, user_id)
        
        # Формируем ответ
        response_data = {
            'response': result['response'],
            'user_id': user_id,
            'timestamp': result['timestamp'],
            'analytics': result['analytics'] if enable_analytics else None,
            'metadata': result['metadata']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в intelligent chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intelligent/analyze', methods=['POST'])
def analyze_response_endpoint():
    """Анализирует качество ответа"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        response = data.get('response', '')
        server_type = data.get('server_type', 'general')
        
        if not question or not response:
            return jsonify({'error': 'Вопрос и ответ обязательны'}), 400
        
        logger.info(f"🔍 Анализирую ответ: {response[:50]}...")
        
        # Анализируем качество
        analysis_result = quality_controller.process_response(question, response, server_type)
        
        return jsonify({
            'question': question,
            'original_response': response,
            'analysis': analysis_result['analysis'],
            'correction_applied': analysis_result['correction_applied'],
            'quality_status': analysis_result['quality_status'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в analyze endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intelligent/history', methods=['GET'])
def get_chat_history():
    """Получает историю чата"""
    try:
        user_id = request.args.get('user_id', 'default')
        limit = int(request.args.get('limit', 10))
        
        history = intelligent_chat.get_chat_history(user_id, limit)
        
        return jsonify({
            'user_id': user_id,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в history endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intelligent/analytics', methods=['GET'])
def get_analytics_summary():
    """Получает сводку аналитики"""
    try:
        summary = intelligent_chat.get_analytics_summary()
        
        return jsonify({
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в analytics endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intelligent/configure', methods=['POST'])
def configure_settings():
    """Настраивает параметры системы"""
    try:
        data = request.get_json()
        
        # Настройки чата
        auto_improvement = data.get('auto_improvement')
        quality_threshold = data.get('quality_threshold')
        
        if auto_improvement is not None or quality_threshold is not None:
            intelligent_chat.configure_settings(auto_improvement, quality_threshold)
        
        # Настройки контроллера качества
        auto_correction = data.get('auto_correction')
        threshold = data.get('threshold')
        
        if auto_correction is not None or threshold is not None:
            quality_controller.configure_quality_settings(threshold, auto_correction)
        
        return jsonify({
            'status': 'success',
            'message': 'Настройки обновлены',
            'settings': {
                'auto_improvement': intelligent_chat.auto_improvement_enabled,
                'quality_threshold': intelligent_chat.quality_threshold,
                'auto_correction': quality_controller.auto_correction_enabled,
                'correction_threshold': quality_controller.quality_threshold
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в configure endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intelligent/health', methods=['GET'])
def health_check():
    """Проверка здоровья системы"""
    try:
        # Проверяем компоненты
        chat_working = intelligent_chat is not None
        controller_working = quality_controller is not None
        
        # Получаем статистику
        analytics_summary = intelligent_chat.get_analytics_summary()
        
        # Определяем общий статус
        if chat_working and controller_working:
            status = 'healthy'
            message = 'Все системы работают нормально'
        else:
            status = 'degraded'
            message = 'Некоторые компоненты недоступны'
        
        return jsonify({
            'status': status,
            'message': message,
            'components': {
                'intelligent_chat': 'healthy' if chat_working else 'unavailable',
                'quality_controller': 'healthy' if controller_working else 'unavailable'
            },
            'statistics': analytics_summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/intelligent/demo', methods=['GET'])
def demo_intelligent_chat():
    """Демонстрация возможностей интеллектуального чата"""
    try:
        demo_questions = [
            "Что такое закон Ома?",
            "Реши уравнение x^2 + 5x + 6 = 0",
            "Как работает транзистор?",
            "Напиши программу на Python для сортировки",
            "Объясни принцип работы ПЛК"
        ]
        
        demo_results = []
        
        for question in demo_questions:
            result = intelligent_chat.process_question(question)
            demo_results.append({
                'question': question,
                'quality_score': result['analytics']['quality_score'],
                'quality_status': result['analytics']['quality_status'],
                'correction_applied': result['analytics']['correction_applied'],
                'issues_found': result['analytics']['issues_found'],
                'response_length': result['metadata']['response_length'],
                'preview': result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
            })
        
        # Общая статистика
        total_questions = len(demo_results)
        avg_quality = sum(r['quality_score'] for r in demo_results) / total_questions
        corrections_count = sum(1 for r in demo_results if r['correction_applied'])
        avg_length = sum(r['response_length'] for r in demo_results) / total_questions
        
        return jsonify({
            'demo_results': demo_results,
            'summary': {
                'total_questions': total_questions,
                'average_quality_score': round(avg_quality, 2),
                'corrections_applied': corrections_count,
                'correction_rate': round(corrections_count / total_questions * 100, 1),
                'average_response_length': round(avg_length),
                'quality_distribution': {
                    'excellent': sum(1 for r in demo_results if r['quality_status'] == 'excellent'),
                    'good': sum(1 for r in demo_results if r['quality_status'] == 'good'),
                    'fair': sum(1 for r in demo_results if r['quality_status'] == 'fair'),
                    'poor': sum(1 for r in demo_results if r['quality_status'] == 'poor')
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в demo endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intelligent/feedback', methods=['POST'])
def submit_feedback():
    """Принимает обратную связь о качестве ответов"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        response = data.get('response', '')
        feedback_score = data.get('feedback_score', 0)  # 1-5
        feedback_text = data.get('feedback_text', '')
        user_id = data.get('user_id', 'default')
        
        if not question or not response:
            return jsonify({'error': 'Вопрос и ответ обязательны'}), 400
        
        # Сохраняем обратную связь (в реальной системе - в базу данных)
        feedback_entry = {
            'question': question,
            'response': response,
            'feedback_score': feedback_score,
            'feedback_text': feedback_text,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Здесь можно добавить логику обучения на основе обратной связи
        logger.info(f"📝 Получена обратная связь: {feedback_score}/5 от {user_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'Обратная связь принята',
            'feedback_id': f"fb_{int(datetime.now().timestamp())}",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в feedback endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint не найден'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == "__main__":
    print("🚀 Запуск API сервера с интегрированной аналитикой ответов")
    print("📡 Доступные endpoints:")
    print("  POST /api/intelligent/chat - Интеллектуальный чат с аналитикой")
    print("  POST /api/intelligent/analyze - Анализ качества ответа")
    print("  GET  /api/intelligent/history - История чата")
    print("  GET  /api/intelligent/analytics - Сводка аналитики")
    print("  POST /api/intelligent/configure - Настройка параметров")
    print("  GET  /api/intelligent/health - Проверка здоровья")
    print("  GET  /api/intelligent/demo - Демонстрация возможностей")
    print("  POST /api/intelligent/feedback - Обратная связь")
    
    # Запуск сервера
    app.run(host='0.0.0.0', port=8095, debug=True)





