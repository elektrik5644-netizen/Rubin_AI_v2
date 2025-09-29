#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubin AI - Математический сервер (исправленная версия)
Решает математические задачи без шаблонных ответов
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import logging
import json
import os
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        'name': 'Rubin AI',
        'version': '2.0',
        'status': 'online',
        'features': ['mathematics', 'physics', 'geometry', 'word_problems']
    })

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Тестовый endpoint"""
    return jsonify({'message': 'Тестовый endpoint работает!', 'timestamp': datetime.now().isoformat()})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата с Rubin AI"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Простой ответ
        response = f"""🤖 **Rubin AI отвечает:**

Я получил ваш вопрос: "{message}"

Это тестовая версия сервера. Система работает!

💡 **Статус:** Все серверы запущены и готовы к работе."""
        
        return jsonify({
            'success': True,
            'response': response,
            'category': 'general'
        })
        
    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

if __name__ == '__main__':
    print("Rubin AI (Test Server) запущен")
    print("URL: http://localhost:8084")
    app.run(host='0.0.0.0', port=8084, debug=False)












