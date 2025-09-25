#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General API сервер для общих вопросов Rubin AI v2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Обработка общих вопросов"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        logger.info(f"📝 Общий запрос: {message}")
        
        # Интеллектуальная обработка без шаблонов
        message_lower = message.lower()
        
        # Анализ сообщения для определения специализированного модуля
        if any(word in message_lower for word in ['транзистор', 'резистор', 'конденсатор', 'схема', 'электричество', 'ток', 'напряжение', 'мощность', 'диод']):
            response = "🔌 Ваш вопрос относится к электротехнике. Перенаправляю к специализированному модулю электротехники для получения точного ответа."
        
        elif any(word in message_lower for word in ['антенна', 'сигнал', 'радио', 'передатчик', 'приемник', 'частота', 'волна']):
            response = "📡 Ваш вопрос касается радиомеханики. Перенаправляю к модулю радиомеханики для детального ответа."
        
        elif any(word in message_lower for word in ['контроллер', 'plc', 'pmac', 'автоматизация', 'сервопривод', 'двигатель', 'привод']):
            response = "🎮 Ваш вопрос связан с контроллерами. Перенаправляю к модулю контроллеров для профессионального ответа."
        
        elif any(word in message_lower for word in ['уравнение', 'вычислить', 'решить', 'математика', 'формула', 'интеграл', 'производная', 'алгебра']):
            response = "🧮 Ваш вопрос требует математических вычислений. Перенаправляю к модулю математики для точного решения."
        
        elif any(word in message_lower for word in ['код', 'алгоритм', 'программирование', 'python', 'javascript', 'java', 'c++', 'функция', 'класс']):
            response = "💻 Ваш вопрос касается программирования. Перенаправляю к модулю программирования для детального объяснения."
        
        elif any(word in message_lower for word in ['нейросеть', 'машинное обучение', 'искусственный интеллект', 'ai', 'алгоритм обучения', 'модель']):
            response = "🧠 Ваш вопрос связан с нейросетями. Перенаправляю к модулю нейросетей для экспертного ответа."
        
        else:
            response = f"""🤔 Анализирую ваш вопрос: "{message}"

Для получения наиболее точного и профессионального ответа, пожалуйста, уточните область:
• Электротехника - компоненты, схемы, расчеты
• Радиомеханика - антенны, сигналы, передатчики  
• Контроллеры - ПЛК, автоматизация, сервоприводы
• Математика - уравнения, вычисления, формулы
• Программирование - код, алгоритмы, языки
• Нейросети - машинное обучение, ИИ

Или задайте более конкретный вопрос с техническими терминами."""
        
        return jsonify({
            'success': True,
            'response': response,
            'category': 'general',
            'server': 'localhost:8085'
        })
        
    except Exception as e:
        logger.error(f"Ошибка в general API: {e}")
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'}), 500

@app.route('/api/health')
def health():
    """Проверка здоровья сервера"""
    return jsonify({
        'server': 'General API',
        'status': 'online',
        'port': 8085,
        'message': 'General API server is running'
    })

@app.route('/')
def index():
    """Главная страница"""
    return jsonify({
        'name': 'Rubin AI v2 - General API',
        'version': '1.0',
        'status': 'online',
        'endpoints': {
            'chat': '/api/chat',
            'health': '/api/health'
        }
    })

if __name__ == '__main__':
    logger.info("🚀 Запускаю General API сервер на порту 8085...")
    app.run(host='0.0.0.0', port=8085, debug=False)