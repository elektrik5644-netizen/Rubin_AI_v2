#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенный Neuro API сервер для Rubin AI v2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import time
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# База знаний нейросетевых алгоритмов
NEURAL_KNOWLEDGE = {
    "нейросеть": {
        "keywords": ["нейросеть", "нейронная сеть", "neural network", "машинное обучение"],
        "explanation": """
🧠 **Нейронная сеть** - это вычислительная модель, вдохновленная биологическими нейронными сетями.

**Основные компоненты:**
- **Нейроны** - базовые вычислительные единицы
- **Связи** - передают сигналы между нейронами
- **Веса** - определяют силу связей
- **Функции активации** - определяют выход нейрона

**Типы нейросетей:**
- **Перцептрон** - простейшая нейросеть
- **Многослойный перцептрон** - несколько слоев
- **Сверточные сети** - для обработки изображений
- **Рекуррентные сети** - для последовательностей
- **Трансформеры** - современная архитектура

**Применения:**
- Распознавание образов
- Обработка естественного языка
- Прогнозирование
- Классификация данных
        """,
        "examples": [
            "Как работает нейросеть?",
            "Объясни принцип работы нейронных сетей",
            "Что такое машинное обучение?"
        ]
    },
    "обучение": {
        "keywords": ["обучение", "тренировка", "training", "обучать"],
        "explanation": """
🎓 **Обучение нейросети** - процесс настройки весов для решения конкретной задачи.

**Типы обучения:**
- **С учителем** - есть правильные ответы
- **Без учителя** - поиск скрытых закономерностей
- **С подкреплением** - обучение через взаимодействие

**Процесс обучения:**
1. **Инициализация** - случайные веса
2. **Прямое распространение** - вычисление выхода
3. **Вычисление ошибки** - сравнение с правильным ответом
4. **Обратное распространение** - корректировка весов
5. **Повтор** - до достижения нужной точности

**Алгоритмы оптимизации:**
- Градиентный спуск
- Adam
- RMSprop
- AdaGrad
        """,
        "examples": [
            "Как обучается нейросеть?",
            "Что такое обратное распространение?",
            "Объясни градиентный спуск"
        ]
    },
    "глубокое обучение": {
        "keywords": ["глубокое обучение", "deep learning", "глубокая сеть"],
        "explanation": """
🔬 **Глубокое обучение** - подраздел машинного обучения с многослойными нейросетями.

**Особенности:**
- **Много слоев** - обычно более 3-х
- **Автоматическое извлечение признаков**
- **Большие объемы данных**
- **Высокая вычислительная сложность**

**Архитектуры:**
- **CNN** - сверточные сети для изображений
- **RNN** - рекуррентные сети для последовательностей
- **LSTM** - долгосрочная память
- **GRU** - упрощенная версия LSTM
- **Transformer** - внимание вместо рекурсии

**Применения:**
- Компьютерное зрение
- Обработка естественного языка
- Распознавание речи
- Автономные системы
        """,
        "examples": [
            "Что такое глубокое обучение?",
            "Объясни CNN и RNN",
            "Как работает Transformer?"
        ]
    }
}

def find_best_match(query):
    """Находит лучший ответ на основе запроса."""
    query_lower = query.lower()
    
    best_match = None
    best_score = 0
    
    for topic, data in NEURAL_KNOWLEDGE.items():
        score = 0
        for keyword in data['keywords']:
            if keyword.lower() in query_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = data
    
    if best_match and best_score > 0:
        return {
            'success': True,
            'response': best_match['explanation'],
            'topic': topic,
            'score': best_score
        }
    else:
        return {
            'success': True,
            'response': """
🧠 **Нейронные сети и машинное обучение**

Я могу помочь с вопросами о:
• **Нейросетях** - принципы работы, архитектуры
• **Обучении** - алгоритмы, оптимизация
• **Глубоком обучении** - CNN, RNN, Transformer
• **Применениях** - компьютерное зрение, NLP

Задайте конкретный вопрос, и я дам подробный ответ!
            """.strip(),
            'topic': 'general',
            'score': 0
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера."""
    return jsonify({
        'status': 'healthy',
        'message': 'Neuro API работает',
        'module': 'Neuro Repository API',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/neuro/status', methods=['GET'])
def neuro_status():
    """Статус нейросетевых проектов."""
    return jsonify({
        'status': 'active',
        'projects': len(NEURAL_KNOWLEDGE),
        'capabilities': [
            'Анализ нейросетевых алгоритмов',
            'Объяснение принципов работы',
            'Консультации по машинному обучению',
            'Помощь с архитектурами сетей'
        ],
        'knowledge_base': list(NEURAL_KNOWLEDGE.keys())
    }), 200

@app.route('/api/neuro/chat', methods=['POST'])
def neuro_chat():
    """Чат с нейросетевыми алгоритмами."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Сообщение не может быть пустым'
            }), 400
        
        logger.info(f"Neuro API получил запрос: {message[:50]}...")
        
        # Поиск лучшего ответа
        result = find_best_match(message)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'topic': result['topic'],
            'confidence': result['score'],
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка в neuro_chat: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/neuro/knowledge', methods=['GET'])
def get_knowledge():
    """Получение базы знаний."""
    return jsonify({
        'success': True,
        'knowledge': NEURAL_KNOWLEDGE,
        'total_topics': len(NEURAL_KNOWLEDGE)
    }), 200

@app.route('/api/neuro/projects', methods=['GET'])
def get_projects():
    """Получение списка проектов."""
    projects = []
    for topic, data in NEURAL_KNOWLEDGE.items():
        projects.append({
            'name': topic,
            'description': data['explanation'][:100] + '...',
            'keywords': data['keywords'],
            'examples': data['examples']
        })
    
    return jsonify({
        'success': True,
        'projects': projects,
        'total': len(projects)
    }), 200

if __name__ == '__main__':
    print("🧠 Neuro API Server запущен")
    print("=" * 50)
    print("🌐 URL: http://localhost:8090")
    print("📚 База знаний: Нейросетевые алгоритмы")
    print("Endpoints:")
    print("  - GET /api/health - Проверка здоровья")
    print("  - GET /api/neuro/status - Статус проектов")
    print("  - POST /api/neuro/chat - Чат с нейросетями")
    print("  - GET /api/neuro/knowledge - База знаний")
    print("  - GET /api/neuro/projects - Список проектов")
    print("=" * 50)
    
    app.run(port=8090, debug=False, use_reloader=False)








