#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для Центральной Базы Знаний Rubin AI
Интегрируется с существующей системой и предоставляет REST API
"""

from flask import Flask, request, jsonify
import logging
import json
from datetime import datetime
from intelligent_knowledge_chat import get_rubin_chat
from central_knowledge_base import get_knowledge_base

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Инициализация компонентов
chat_system = get_rubin_chat()
knowledge_base = get_knowledge_base()

@app.route('/api/knowledge/chat', methods=['POST'])
def chat_with_knowledge():
    """Основной endpoint для чата с предложениями знаний"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'default')
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        logger.info(f"💬 Получено сообщение от {user_id}: {message[:50]}...")
        
        # Обрабатываем сообщение
        response = chat_system.process_message(message, user_id)
        
        # Получаем активные предложения
        active_suggestions = chat_system.knowledge_manager.get_active_suggestions(user_id)
        
        return jsonify({
            'response': response,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'active_suggestions': len(active_suggestions),
            'suggestions': active_suggestions
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/search', methods=['GET'])
def search_knowledge():
    """Поиск в базе знаний"""
    try:
        query = request.args.get('q', '')
        category = request.args.get('category', None)
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Параметр q (запрос) обязателен'}), 400
        
        logger.info(f"🔍 Поиск знаний: {query}")
        
        results = knowledge_base.search_knowledge(query, category, limit)
        
        return jsonify({
            'query': query,
            'category': category,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в search endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/suggestions', methods=['GET'])
def get_suggestions():
    """Получение ожидающих подтверждения предложений"""
    try:
        user_id = request.args.get('user_id', 'default')
        
        suggestions = knowledge_base.get_pending_suggestions()
        active_suggestions = chat_system.knowledge_manager.get_active_suggestions(user_id)
        
        return jsonify({
            'pending_suggestions': suggestions,
            'active_suggestions': active_suggestions,
            'count': len(suggestions)
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в suggestions endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/approve', methods=['POST'])
def approve_suggestion():
    """Подтверждение предложения"""
    try:
        data = request.get_json()
        suggestion_id = data.get('suggestion_id')
        user_id = data.get('user_id', 'default')
        feedback = data.get('feedback', '')
        
        if not suggestion_id:
            return jsonify({'error': 'suggestion_id обязателен'}), 400
        
        logger.info(f"✅ Подтверждение предложения {suggestion_id} от {user_id}")
        
        result = chat_system.knowledge_manager.handle_user_feedback(
            f"approve {suggestion_id}", user_id
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в approve endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/reject', methods=['POST'])
def reject_suggestion():
    """Отклонение предложения"""
    try:
        data = request.get_json()
        suggestion_id = data.get('suggestion_id')
        user_id = data.get('user_id', 'default')
        feedback = data.get('feedback', '')
        
        if not suggestion_id:
            return jsonify({'error': 'suggestion_id обязателен'}), 400
        
        logger.info(f"❌ Отклонение предложения {suggestion_id} от {user_id}")
        
        result = chat_system.knowledge_manager.handle_user_feedback(
            f"reject {suggestion_id}", user_id
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в reject endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/add', methods=['POST'])
def add_knowledge():
    """Добавление нового знания"""
    try:
        data = request.get_json()
        
        required_fields = ['category', 'title', 'content']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Поле {field} обязательно'}), 400
        
        logger.info(f"➕ Добавление знания: {data['title']}")
        
        success = knowledge_base.add_knowledge(
            category=data['category'],
            title=data['title'],
            content=data['content'],
            keywords=data.get('keywords', ''),
            formulas=data.get('formulas', ''),
            examples=data.get('examples', ''),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', 'manual')
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Знание "{data["title"]}" добавлено'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Не удалось добавить знание (возможно, уже существует)'
            }), 400
        
    except Exception as e:
        logger.error(f"❌ Ошибка в add endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/stats', methods=['GET'])
def get_stats():
    """Получение статистики базы знаний"""
    try:
        stats = knowledge_base.get_knowledge_stats()
        
        return jsonify({
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в stats endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
@app.route('/api/knowledge/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        stats = knowledge_base.get_knowledge_stats()
        
        return jsonify({
            'status': 'healthy',
            'total_facts': stats['total_facts'],
            'pending_suggestions': stats['pending_suggestions'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/knowledge/configure', methods=['POST'])
def configure_preferences():
    """Настройка предпочтений пользователя"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        preferences = data.get('preferences', {})
        
        chat_system.knowledge_manager.configure_preferences(preferences)
        
        return jsonify({
            'status': 'success',
            'message': 'Настройки обновлены',
            'preferences': preferences
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в configure endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge/history', methods=['GET'])
def get_chat_history():
    """Получение истории чата"""
    try:
        user_id = request.args.get('user_id', 'default')
        limit = int(request.args.get('limit', 50))
        
        history = chat_system.get_chat_history(user_id)
        
        return jsonify({
            'user_id': user_id,
            'history': history[-limit:],  # Последние N сообщений
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в history endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint не найден'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == "__main__":
    print("🚀 Запуск API сервера Центральной Базы Знаний Rubin AI")
    print("📡 Доступные endpoints:")
    print("  POST /api/knowledge/chat - Чат с предложениями знаний")
    print("  GET  /api/knowledge/search - Поиск в базе знаний")
    print("  GET  /api/knowledge/suggestions - Получение предложений")
    print("  POST /api/knowledge/approve - Подтверждение предложения")
    print("  POST /api/knowledge/reject - Отклонение предложения")
    print("  POST /api/knowledge/add - Добавление знания")
    print("  GET  /api/knowledge/stats - Статистика")
    print("  GET  /api/knowledge/health - Проверка здоровья")
    print("  POST /api/knowledge/configure - Настройка предпочтений")
    print("  GET  /api/knowledge/history - История чата")
    
    # Запуск сервера
    try:
        print("🚀 Запуск Knowledge API Server...")
        app.run(host='0.0.0.0', port=8093, debug=False)
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        import traceback
        traceback.print_exc()







