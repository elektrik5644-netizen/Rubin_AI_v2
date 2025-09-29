#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radiomechanics Server для обработки радиочастотных вопросов
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'healthy',
        'service': 'Radiomechanics Server',
        'port': 8089,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    """Обработка радиочастотных вопросов"""
    try:
        if request.method == 'GET':
            message = request.args.get('message', '')
        else:
            data = request.get_json()
            message = data.get('message', '')
        
        logger.info(f"📡 Получен запрос радиомеханики: {message[:50]}...")
        
        # Простая логика ответов
        response = "📡 **Radiomechanics Server:**\n\n"
        response += f"**Запрос:** {message}\n\n"
        
        if any(word in message.lower() for word in ['антенна', 'antenna']):
            response += "**Антенны:**\n"
            response += "- Дипольная антенна: λ/2\n"
            response += "- Монопольная антенна: λ/4\n"
            response += "- Спиральная антенна: для круговой поляризации\n"
        
        elif any(word in message.lower() for word in ['сигнал', 'signal']):
            response += "**Обработка сигналов:**\n"
            response += "- Амплитудная модуляция (AM)\n"
            response += "- Частотная модуляция (FM)\n"
            response += "- Фазовая модуляция (PM)\n"
        
        elif any(word in message.lower() for word in ['частота', 'frequency']):
            response += "**Частотные характеристики:**\n"
            response += "- Низкие частоты: 30-300 кГц\n"
            response += "- Средние частоты: 300 кГц - 3 МГц\n"
            response += "- Высокие частоты: 3-30 МГц\n"
        
        else:
            response += "**Общая информация:**\n"
            response += "- Радиоволны: электромагнитные волны\n"
            response += "- Скорость света: c = 3×10⁸ м/с\n"
            response += "- Формула: λ = c/f\n"
        
        return jsonify({
            'status': 'success',
            'response': response,
            'service': 'radiomechanics',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("📡 Radiomechanics Server запущен")
    print("URL: http://localhost:8089")
    print("Доступные эндпоинты:")
    print("  - GET/POST /api/chat - обработка радиочастотных вопросов")
    print("  - GET /api/health - проверка здоровья")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8089, debug=False)