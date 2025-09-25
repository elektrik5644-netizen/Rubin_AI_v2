#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сервер общих вопросов для Rubin AI v2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Обработка CORS preflight запросов
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
        return response

# Установка правильных заголовков для всех ответов
@app.after_request
def after_request(response):
    if response.content_type == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'General Server',
        'port': 8085,
        'timestamp': time.time()
    })

# База знаний для общих вопросов
GENERAL_KNOWLEDGE = {
    "привет": {
        "title": "Приветствие",
        "description": "Добро пожаловать в Rubin AI v2",
        "explanation": """
**🤖 Привет! Я Rubin - профессиональная среда разработки.**

**Я могу помочь вам с:**
• **⚡ Электротехникой** - законы Кирхгофа, схемы, расчеты
• **📡 Радиомеханикой** - антенны, сигналы, модуляция
• **🎛️ Контроллерами** - ПЛК, ЧПУ, автоматизация
• **🧮 Математикой** - уравнения, алгоритмы, расчеты
• **💻 Программированием** - C++, Python, алгоритмы

**Как использовать:**
• Задавайте вопросы на русском языке
• Я автоматически направлю ваш запрос к нужному специалисту
• Получите подробные, профессиональные ответы

**Примеры вопросов:**
• "Как решить квадратное уравнение x² - 5x + 6 = 0?"
• "Объясни закон Кирхгофа"
• "Как передаются данные с ЧПУ в контроллер?"
• "Сравни C++ и Python для промышленной автоматизации"

**Готов помочь! Задавайте вопросы!** 🚀
        """
    },
    "помощь": {
        "title": "Справка",
        "description": "Как использовать Rubin AI v2",
        "explanation": """
**📚 Справка по Rubin AI v2:**

**Доступные модули:**
• **Электротехника (порт 8087)** - электрические схемы, законы, расчеты
• **Радиомеханика (порт 8089)** - радиочастоты, антенны, передатчики
• **Контроллеры (порт 9000)** - ПЛК, ЧПУ, промышленная автоматизация
• **Математика (порт 8086)** - уравнения, функции, алгоритмы
• **Программирование (порт 8088)** - языки программирования, алгоритмы

**Типы вопросов:**
• **Технические** - конкретные инженерные задачи
• **Образовательные** - объяснение концепций
• **Практические** - решение задач
• **Сравнительные** - анализ технологий

**Формат ответов:**
• Подробные объяснения с примерами
• Код и формулы где необходимо
• Практические рекомендации
• Ссылки на дополнительные ресурсы

**Для получения помощи просто задайте вопрос!**
        """
    },
    "статус": {
        "title": "Статус системы",
        "description": "Информация о работе системы",
        "explanation": """
**📊 Статус системы Rubin AI v2:**

**Активные серверы:**
• ✅ **Умный диспетчер** (порт 8080) - маршрутизация запросов
• ✅ **Электротехника** (порт 8087) - электрические расчеты
• ✅ **Радиомеханика** (порт 8089) - радиочастотные системы
• ✅ **Контроллеры** (порт 9000) - промышленная автоматизация
• ✅ **Математика** (порт 8086) - математические задачи
• ✅ **Программирование** (порт 8088) - разработка ПО

**Возможности:**
• Интеллектуальная маршрутизация запросов
• Специализированные базы знаний
• Подробные технические ответы
• Поддержка русского языка

**Система работает стабильно!** 🚀
        """
    }
}

def find_best_match(query):
    """Поиск наиболее подходящего ответа по запросу"""
    query_lower = query.lower()
    
    # Прямое совпадение
    for key, data in GENERAL_KNOWLEDGE.items():
        if key in query_lower:
            return data
    
    # Поиск по ключевым словам
    keywords = {
        "привет": "привет",
        "hello": "привет",
        "hi": "привет",
        "здравствуй": "привет",
        "помощь": "помощь",
        "help": "помощь",
        "справка": "помощь",
        "статус": "статус",
        "status": "статус",
        "работает": "статус",
        "онлайн": "статус"
    }
    
    for keyword, topic in keywords.items():
        if keyword in query_lower:
            return GENERAL_KNOWLEDGE[topic]
    
    return None

@app.route('/')
def index():
    return jsonify({
        'name': 'General Server',
        'version': '1.0',
        'status': 'online',
        'features': ['greetings', 'help', 'status']
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Обработка общих вопросов"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        # Поиск подходящего ответа
        knowledge = find_best_match(message)
        
        if knowledge:
            response = {
                "success": True,
                "category": "general",
                "response": {
                    "title": knowledge["title"],
                    "description": knowledge["description"],
                    "explanation": knowledge["explanation"],
                    "success": True
                }
            }
            
            return jsonify(response)
        else:
            return jsonify({
                "success": True,
                "category": "general",
                "response": {
                    "title": "Общий ответ",
                    "description": "Я получил ваш вопрос",
                    "explanation": f"Вы спросили: \"{message}\"\n\nЯ получил ваш вопрос, но не могу дать специализированный ответ. Попробуйте переформулировать вопрос или обратитесь к конкретному модулю:\n\n• Электротехника - для электрических вопросов\n• Радиомеханика - для радиочастотных вопросов\n• Контроллеры - для вопросов автоматизации\n• Математика - для математических задач\n• Программирование - для вопросов разработки",
                    "success": True
                }
            })
    
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return jsonify({
            "error": "Внутренняя ошибка сервера",
            "details": str(e)
        }), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': '2025-09-20'})

if __name__ == '__main__':
    logger.info("Запуск сервера общих вопросов на порту 8085...")
    app.run(host='0.0.0.0', port=8085, debug=True)
