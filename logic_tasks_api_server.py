#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для логических задач Rubin AI v2
Интеграция с базой данных LogiEval
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from logic_tasks_module import get_logic_task, solve_logic_task, get_logic_statistics

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/api/logic/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера логических задач."""
    return jsonify({
        'status': 'healthy',
        'message': 'Logic Tasks API работает',
        'module': 'Logic Tasks Server'
    }), 200

@app.route('/api/logic/task', methods=['GET'])
def get_task():
    """Получает логическую задачу."""
    try:
        task_type = request.args.get('type', None)
        task = get_logic_task(task_type)
        
        if not task or "Задача не найдена" in task:
            return jsonify({
                'success': False,
                'error': 'Не удалось получить задачу'
            }), 404
        
        return jsonify({
            'success': True,
            'task': task,
            'task_type': task_type or 'random'
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения задачи: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/logic/solve', methods=['POST'])
def solve_task():
    """Решает логическую задачу."""
    try:
        data = request.get_json()
        task_type = data.get('task_type', None)
        user_answer = data.get('answer', None)
        
        if not user_answer:
            return jsonify({
                'success': False,
                'error': 'Необходимо указать ответ'
            }), 400
        
        result = solve_logic_task(task_type, user_answer)
        
        return jsonify({
            'success': True,
            'result': result
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка решения задачи: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/logic/answer', methods=['POST'])
def get_answer():
    """Получает правильный ответ на задачу."""
    try:
        data = request.get_json()
        task_type = data.get('task_type', None)
        
        result = solve_logic_task(task_type, None)  # Без ответа пользователя
        
        return jsonify({
            'success': True,
            'answer': result.get('correct_answer', 'Ответ не найден'),
            'explanation': result.get('message', '')
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения ответа: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/logic/stats', methods=['GET'])
def get_stats():
    """Получает статистику логических задач."""
    try:
        stats = get_logic_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/logic/types', methods=['GET'])
def get_task_types():
    """Получает доступные типы задач."""
    try:
        stats = get_logic_statistics()
        task_types = []
        
        for name, info in stats.get('dataset_stats', {}).items():
            task_types.append({
                'name': name,
                'count': info['count'],
                'description': info['description']
            })
        
        return jsonify({
            'success': True,
            'task_types': task_types
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения типов задач: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/logic/chat', methods=['POST'])
def chat():
    """Основной эндпоинт для чата с логическими задачами."""
    try:
        data = request.get_json()
        message = data.get('message', '').lower()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Сообщение не может быть пустым'
            }), 400
        
        # Определяем тип запроса
        if any(word in message for word in ['задача', 'логика', 'логическая', 'решить']):
            # Получить задачу
            task_type = None
            if 'доказательство' in message:
                task_type = 'доказательства'
            elif 'правило' in message:
                task_type = 'правила'
            elif 'медицин' in message:
                task_type = 'медицина'
            elif 'математик' in message:
                task_type = 'математика'
            elif 'аргумент' in message:
                task_type = 'аргументы'
            
            task = get_logic_task(task_type)
            
            return jsonify({
                'success': True,
                'response': task,
                'task_type': task_type or 'random'
            }), 200
            
        elif any(word in message for word in ['ответ', 'правильный', 'решение']):
            # Получить ответ
            result = solve_logic_task(None, None)
            
            return jsonify({
                'success': True,
                'response': f"Правильный ответ: {result.get('correct_answer', 'Не найден')}"
            }), 200
            
        elif any(word in message for word in ['статистика', 'статистик', 'результат']):
            # Получить статистику
            stats = get_logic_statistics()
            
            response = f"""
📊 **Статистика логических задач**

🎯 Всего попыток: {stats['total_attempts']}
✅ Решено правильно: {stats['solved_tasks']}
❌ Неправильно: {stats['failed_tasks']}
📈 Успешность: {stats['success_rate']}

📚 **Доступные типы задач:**
"""
            for name, info in stats['dataset_stats'].items():
                response += f"🔹 {name}: {info['count']} задач - {info['description']}\n"
            
            return jsonify({
                'success': True,
                'response': response.strip()
            }), 200
            
        else:
            # Общая информация
            response = """
🧠 **Логические задачи Rubin AI**

Доступные команды:
• "дай задачу" - получить случайную логическую задачу
• "задача на доказательства" - задача на доказательства
• "задача на правила" - задача с правилами
• "медицинская задача" - медицинская логическая задача
• "математическая задача" - математическая логическая задача
• "аргументативная задача" - аргументативная задача LSAT
• "покажи ответ" - показать правильный ответ
• "статистика" - показать статистику решения задач

Всего доступно более 15,000 логических задач различных типов!
"""
            
            return jsonify({
                'success': True,
                'response': response.strip()
            }), 200
            
    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🧠 Logic Tasks API Server запущен")
    print("=" * 50)
    print("🌐 URL: http://localhost:8106")
    print("📚 База данных: LogiEval")
    print("Endpoints:")
    print("  - GET /api/logic/health - Проверка здоровья")
    print("  - GET /api/logic/task - Получить задачу")
    print("  - POST /api/logic/solve - Решить задачу")
    print("  - POST /api/logic/answer - Получить ответ")
    print("  - GET /api/logic/stats - Статистика")
    print("  - GET /api/logic/types - Типы задач")
    print("  - POST /api/logic/chat - Чат с задачами")
    print("=" * 50)
    
    app.run(port=8106, debug=False, use_reloader=False)



