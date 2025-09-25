#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой AI чат сервер для Rubin AI v2
Работает на порту 8084 без сложных зависимостей
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from datetime import datetime
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
        return response

@app.after_request
def after_request(response):
    if response.content_type == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

# Естественные ответы без шаблонов
AI_RESPONSES = {
    "привет": "Привет! Расскажите, что вас интересует.",
    "hello": "Hello! What would you like to know?",
    "помощь": "Я здесь, чтобы помочь. Что именно вас интересует?",
    "help": "I'm here to help. What specifically interests you?",
    "статус": "Система работает нормально.",
    "status": "System is running normally.",
    "что ты умеешь": "Я могу отвечать на вопросы и помогать с различными задачами.",
    "who are you": "I'm an AI assistant here to help with your questions."
}

def get_ai_response(message):
    """Получение ответа от AI"""
    message_lower = message.lower()
    
    # Поиск точного совпадения
    for keyword, response in AI_RESPONSES.items():
        if keyword in message_lower:
            return response
    
    # Естественные ответы на основе ключевых слов
    if any(word in message_lower for word in ["программирование", "код", "python", "javascript"]):
        return "Расскажите подробнее о вашей задаче программирования."
    
    if any(word in message_lower for word in ["электротехника", "электричество", "ток", "напряжение"]):
        return "Что именно вас интересует в электротехнике?"
    
    if any(word in message_lower for word in ["радио", "антенна", "передатчик", "приемник"]):
        return "Какой вопрос по радиомеханике у вас есть?"
    
    if any(word in message_lower for word in ["контроллер", "plc", "автоматизация"]):
        return "Что вы хотели бы узнать о контроллерах?"
    
    # Естественный ответ без шаблонов
    return f"Вы спросили: '{message}'. Расскажите больше о том, что вас интересует."

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "status": "healthy",
        "service": "Simple AI Chat",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "port": 8084
    })

@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    """Основной endpoint для чата"""
    try:
        if request.method == 'GET':
            message = request.args.get('message', '')
        else:
            data = request.get_json()
            message = data.get('message', '') if data else ''

        if not message:
            return jsonify({"success": False, "message": "Сообщение не может быть пустым"}), 400

        response_text = get_ai_response(message)

        return jsonify({
            "success": True,
            "message": response_text,
            "query": message,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Ошибка в чате: {str(e)}")
        return jsonify({"success": False, "message": f"Внутренняя ошибка сервера: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Статус AI чата"""
    return jsonify({
        "status": "online",
        "service": "Simple AI Chat",
        "port": 8084,
        "features": ["chat", "health_check", "status"],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def index():
    """Главная страница - перенаправление на веб-интерфейс"""
    return send_from_directory('../', 'RubinIDE.html')

@app.route('/RubinIDE.html', methods=['GET'])
def rubin_ide():
    """Веб-интерфейс Rubin IDE"""
    return send_from_directory('../', 'RubinIDE.html')

if __name__ == '__main__':
    logger.info("Запуск простого AI чата на порту 8084...")
    logger.info("Веб-интерфейс: http://localhost:8084")
    logger.info("API чат: http://localhost:8084/api/chat")
    logger.info("Статус: http://localhost:8084/health")
    
    app.run(host='0.0.0.0', port=8084, debug=True)
