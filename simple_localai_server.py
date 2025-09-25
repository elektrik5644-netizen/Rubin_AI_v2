#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой LocalAI сервер для Rubin AI v2
Имитирует OpenAI API для локального использования
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from datetime import datetime

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

# Простая база знаний для генерации ответов
KNOWLEDGE_BASE = {
    "программирование": {
        "keywords": ["код", "программа", "алгоритм", "функция", "класс", "переменная"],
        "responses": [
            "Программирование - это процесс создания компьютерных программ.",
            "Основы программирования включают изучение алгоритмов и структур данных.",
            "Популярные языки программирования: Python, JavaScript, Java, C++."
        ]
    },
    "электротехника": {
        "keywords": ["ток", "напряжение", "сопротивление", "мощность", "схема", "электричество"],
        "responses": [
            "Электротехника изучает электрические явления и их применение.",
            "Основные законы: закон Ома, законы Кирхгофа.",
            "Электрические схемы состоят из источников, нагрузок и соединительных проводов."
        ]
    },
    "математика": {
        "keywords": ["число", "уравнение", "функция", "интеграл", "производная", "алгебра"],
        "responses": [
            "Математика - наука о числах, структурах и изменениях.",
            "Основные разделы: алгебра, геометрия, анализ, статистика.",
            "Математические методы применяются во всех областях науки."
        ]
    },
    "общее": {
        "keywords": ["привет", "помощь", "спасибо", "как дела"],
        "responses": [
            "Привет! Я LocalAI сервер для Rubin AI v2.",
            "Я могу помочь с различными техническими вопросами.",
            "Задавайте вопросы по программированию, электротехнике, математике!"
        ]
    }
}

def generate_response(message):
    """Генерация ответа на основе сообщения"""
    message_lower = message.lower()
    
    # Персонализированные ответы для живого общения
    if any(word in message_lower for word in ["привет", "здравствуй", "hello", "hi"]):
        responses = [
            "Привет! Я Rubin, твой AI-помощник. Рад тебя видеть! 😊",
            "Здравствуй! Как дела? Чем могу помочь?",
            "Привет! Готов помочь с любыми техническими вопросами!",
            "Привет! Я здесь, чтобы помочь с программированием, электротехникой и не только!"
        ]
        import random
        return random.choice(responses)
    
    if any(word in message_lower for word in ["как дела", "как жизнь", "как поживаешь"]):
        responses = [
            "У меня все отлично! Готов помочь с любыми задачами! А у тебя как дела?",
            "Прекрасно! Система работает стабильно. Как твои дела?",
            "Все хорошо! Готов к работе. Расскажи, что тебя интересует?",
            "Отлично! Все модули онлайн. Чем займемся?"
        ]
        import random
        return random.choice(responses)
    
    if any(word in message_lower for word in ["какой день", "какая дата", "сегодня", "время", "дата"]):
        from datetime import datetime
        now = datetime.now()
        responses = [
            f"Сегодня {now.strftime('%d.%m.%Y')} ({now.strftime('%A')}). Время: {now.strftime('%H:%M')}",
            f"Сегодня {now.strftime('%d %B %Y')}. День недели: {now.strftime('%A')}",
            f"Текущая дата: {now.strftime('%d.%m.%Y')}, время: {now.strftime('%H:%M:%S')}",
            f"Сегодня {now.strftime('%d.%m.%Y')}. Как дела? Чем займемся?"
        ]
        import random
        return random.choice(responses)
    
    if any(word in message_lower for word in ["pmac", "многоосевой", "контроллер движения"]):
        responses = [
            "PMAC (Programmable Multi-Axis Controller) - это программируемый многоосевой контроллер движения. Может управлять до 32 осями одновременно с высокой точностью позиционирования. Используется в станках с ЧПУ, робототехнике и измерительных системах.",
            "PMAC контроллеры - это мощные системы управления движением. Основные возможности: управление множественными осями, высокая точность, работа в реальном времени, встроенная математика для траекторий.",
            "PMAC - это контроллер движения для промышленной автоматизации. Поддерживает различные типы двигателей, имеет встроенные алгоритмы интерполяции и может работать с энкодерами для обратной связи."
        ]
        import random
        return random.choice(responses)
    
    # Поиск подходящей категории
    for category, data in KNOWLEDGE_BASE.items():
        for keyword in data["keywords"]:
            if keyword in message_lower:
                import random
                return random.choice(data["responses"])
    
    # Специальные ответы для технических вопросов
    if any(word in message_lower for word in ["трансформатор", "электрический", "мощность", "ток", "напряжение"]):
        return "Трансформатор - это электромагнитное устройство для преобразования переменного напряжения. Основной принцип: переменный ток в первичной обмотке создает переменное магнитное поле, которое индуцирует напряжение во вторичной обмотке. Коэффициент трансформации k = U1/U2 = N1/N2, где N - количество витков."
    
    if any(word in message_lower for word in ["программирование", "код", "алгоритм", "функция"]):
        return "Программирование - это процесс создания компьютерных программ. Основы включают изучение алгоритмов, структур данных, синтаксиса языков программирования. Популярные языки: Python (простота), JavaScript (веб), Java (кроссплатформенность), C++ (производительность)."
    
    if any(word in message_lower for word in ["математика", "уравнение", "функция", "интеграл"]):
        return "Математика - наука о числах, структурах и изменениях. Основные разделы: алгебра (уравнения, функции), геометрия (пространство, формы), анализ (производные, интегралы), статистика (анализ данных). Математические методы применяются во всех областях науки и техники."
    
    # Если не найдена категория, возвращаем живой ответ
    responses = [
        "Интересный вопрос! Расскажи подробнее, что именно тебя интересует?",
        "Хм, это интересно! Можешь уточнить, в какой области тебе нужна помощь?",
        "Понял! Давай разберемся вместе. Что конкретно ты хочешь узнать?",
        "Отличный вопрос! Я готов помочь, просто уточни детали.",
        "Интересно! Расскажи больше, и я постараюсь помочь."
    ]
    import random
    return random.choice(responses)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Simple LocalAI Server",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/v1/models', methods=['GET'])
def get_models():
    """Получение списка доступных моделей (OpenAI API совместимость)"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "localai"
            },
            {
                "id": "gpt-4",
                "object": "model", 
                "created": 1677610602,
                "owned_by": "localai"
            }
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Chat completions endpoint (OpenAI API совместимость)"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Получаем последнее сообщение
        last_message = messages[-1]
        user_message = last_message.get('content', '')
        
        # Генерируем ответ
        response_text = generate_response(user_message)
        
        return jsonify({
            "id": "chatcmpl-localai-" + str(int(datetime.now().timestamp())),
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": data.get('model', 'gpt-3.5-turbo'),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split())
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка в chat completions: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Text completions endpoint (OpenAI API совместимость)"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Генерируем ответ
        response_text = generate_response(prompt)
        
        return jsonify({
            "id": "cmpl-localai-" + str(int(datetime.now().timestamp())),
            "object": "text_completion",
            "created": int(datetime.now().timestamp()),
            "model": data.get('model', 'gpt-3.5-turbo'),
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка в completions: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Запуск простого LocalAI сервера на порту 11434...")
    logger.info("OpenAI API совместимость: http://localhost:11434/v1/")
    logger.info("Модели: http://localhost:11434/v1/models")
    logger.info("Статус: http://localhost:11434/health")
    app.run(host='0.0.0.0', port=11434, debug=True)
