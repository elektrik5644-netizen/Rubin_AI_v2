#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def handle_arduino_nano_minimal(query):
    """Минимальная обработка Arduino Nano"""
    return f"Arduino Nano: {query}"

@app.route('/api/chat', methods=['POST'])
def minimal_chat():
    """Минимальный чат для тестирования"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Простая проверка на Arduino
        if 'arduino' in message.lower() or 'nano' in message.lower():
            result = handle_arduino_nano_minimal(message)
            return jsonify({
                'success': True,
                'response': result,
                'category': 'arduino_nano',
                'server': 'localhost:8082 (минимальный тест)'
            })
        else:
            return jsonify({
                'success': True,
                'response': 'Минимальный тест работает',
                'category': 'general',
                'server': 'localhost:8082 (минимальный тест)'
            })
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return jsonify({'error': f'Ошибка: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Minimal Arduino Test'})

if __name__ == '__main__':
    logger.info("🚀 Запуск минимального тестового сервера Arduino Nano...")
    logger.info("📡 Порт: 8082")
    logger.info("🔗 URL: http://localhost:8082")
    app.run(host='0.0.0.0', port=8082, debug=False)