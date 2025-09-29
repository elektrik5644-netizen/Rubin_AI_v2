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

def handle_arduino_nano_debug(query):
    """Отладочная обработка Arduino Nano"""
    logger.info(f"🔧 Обработка Arduino Nano запроса: {query}")
    return f"Arduino Nano Debug: {query}"

@app.route('/api/chat', methods=['POST'])
def debug_chat():
    """Отладочный чат для тестирования Arduino Nano"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        logger.info(f"📝 Получен запрос: {message}")
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Простая проверка на Arduino
        if any(word in message.lower() for word in ['arduino', 'ардуино', 'nano', 'нано']):
            logger.info("🔧 Обнаружен Arduino Nano запрос")
            result = handle_arduino_nano_debug(message)
            logger.info(f"✅ Результат: {result}")
            return jsonify({
                'success': True,
                'response': result,
                'category': 'arduino_nano',
                'server': 'localhost:8084 (отладочный тест)'
            })
        else:
            logger.info("📝 Обычный запрос")
            return jsonify({
                'success': True,
                'response': 'Отладочный тест работает',
                'category': 'general',
                'server': 'localhost:8084 (отладочный тест)'
            })
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Ошибка: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Debug Arduino Test'})

if __name__ == '__main__':
    logger.info("🚀 Запуск отладочного тестового сервера Arduino Nano...")
    logger.info("📡 Порт: 8084")
    logger.info("🔗 URL: http://localhost:8084")
    app.run(host='0.0.0.0', port=8084, debug=False)





