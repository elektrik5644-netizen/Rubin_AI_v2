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

def handle_arduino_nano_simple(query):
    """Простая обработка Arduino Nano"""
    query_lower = query.lower()
    
    if 'pin' in query_lower or 'пин' in query_lower:
        return "**Пины Arduino Nano:**\n- Цифровые: D2-D13\n- Аналоговые: A0-A7\n- PWM: D3, D5, D6, D9, D10, D11"
    
    if 'function' in query_lower or 'функц' in query_lower:
        return "**Основные функции:**\n- digitalWrite(pin, value)\n- digitalRead(pin)\n- analogRead(pin)\n- analogWrite(pin, value)"
    
    return "**Arduino Nano** - микроконтроллер ATmega328P с 14 цифровыми и 8 аналоговыми пинами."

@app.route('/api/chat', methods=['POST'])
def simple_chat():
    """Упрощенный чат для тестирования Arduino Nano"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Простая проверка на Arduino
        if any(word in message.lower() for word in ['arduino', 'ардуино', 'nano', 'нано', 'pin', 'пин']):
            result = handle_arduino_nano_simple(message)
            return jsonify({
                'success': True,
                'response': result,
                'category': 'arduino_nano',
                'server': 'localhost:8083 (упрощенный тест)'
            })
        else:
            return jsonify({
                'success': True,
                'response': 'Упрощенный тест работает. Для Arduino Nano используйте ключевые слова: arduino, nano, pin, пин',
                'category': 'general',
                'server': 'localhost:8083 (упрощенный тест)'
            })
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return jsonify({'error': f'Ошибка: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Simplified Arduino Test'})

if __name__ == '__main__':
    logger.info("🚀 Запуск упрощенного тестового сервера Arduino Nano...")
    logger.info("📡 Порт: 8083")
    logger.info("🔗 URL: http://localhost:8083")
    app.run(host='0.0.0.0', port=8083, debug=False)





