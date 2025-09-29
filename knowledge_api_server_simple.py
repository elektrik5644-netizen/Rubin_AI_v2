#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенная версия Knowledge API Server для тестирования
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/knowledge/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'service': 'knowledge_api',
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        'message': 'Knowledge API Server работает'
    })

@app.route('/api/knowledge/test', methods=['GET'])
def test_endpoint():
    """Тестовый эндпоинт"""
    return jsonify({
        'message': 'Knowledge API Server тестирование успешно',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/knowledge/chat', methods=['POST'])
def chat_with_knowledge():
    """Упрощенный чат эндпоинт"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'default')
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        logger.info(f"💬 Получено сообщение от {user_id}: {message[:50]}...")
        
        # Простой ответ
        response = f"Получено сообщение: '{message}'. Knowledge API Server работает!"
        
        return jsonify({
            'response': response,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Ошибка в chat_with_knowledge: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("🧠 Knowledge API Server (упрощенная версия) запущен")
    print("URL: http://localhost:8095")
    print("Доступные эндпоинты:")
    print("  - GET /api/knowledge/health - проверка здоровья")
    print("  - GET /api/knowledge/test - тестовый эндпоинт")
    print("  - POST /api/knowledge/chat - чат с базой знаний")
    print("=" * 60)
    
    try:
        app.run(host='127.0.0.1', port=8095, debug=False)
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        import traceback
        traceback.print_exc()
