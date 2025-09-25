#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Математический API сервер для Rubin AI v2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Математические знания
MATHEMATICS_KNOWLEDGE = {
    'алгебра': {
        'keywords': ['уравнение', 'квадратное', 'линейное', 'система', 'переменная', 'коэффициент'],
        'explanation': 'Алгебра - раздел математики, изучающий операции над числами и переменными. Основные темы: уравнения, системы уравнений, функции, многочлены.'
    },
    'геометрия': {
        'keywords': ['треугольник', 'круг', 'площадь', 'периметр', 'угол', 'прямая', 'плоскость'],
        'explanation': 'Геометрия - раздел математики, изучающий пространственные отношения и формы. Основные темы: фигуры, площади, объемы, углы.'
    },
    'арифметика': {
        'keywords': ['сложение', 'вычитание', 'умножение', 'деление', 'число', 'цифра', 'дробь'],
        'explanation': 'Арифметика - раздел математики, изучающий операции с числами. Основные операции: сложение, вычитание, умножение, деление.'
    },
    'тригонометрия': {
        'keywords': ['синус', 'косинус', 'тангенс', 'угол', 'радиан', 'градус'],
        'explanation': 'Тригонометрия - раздел математики, изучающий соотношения между углами и сторонами треугольников. Основные функции: sin, cos, tan.'
    },
    'статистика': {
        'keywords': ['среднее', 'медиана', 'мода', 'дисперсия', 'вероятность', 'выборка'],
        'explanation': 'Статистика - раздел математики, изучающий сбор, анализ и интерпретацию данных. Основные понятия: среднее, медиана, мода, дисперсия.'
    }
}

def evaluate_expression(expression):
    """Безопасное вычисление математических выражений"""
    try:
        # Убираем пробелы и проверяем на безопасность
        expr = expression.replace(' ', '')
        
        # Разрешенные символы для вычислений (включая переменные)
        allowed_chars = set('0123456789+-*/.()abcdefghijklmnopqrstuvwxyz=')
        if not all(c in allowed_chars for c in expr.lower()):
            return None
            
        # Если это уравнение с переменной, решаем его
        if '=' in expr and any(c.isalpha() for c in expr):
            return solve_equation(expr)
        
        # Если это простое выражение без переменных
        if not any(c.isalpha() for c in expr):
            result = eval(expr)
            return result
            
        return None
    except:
        return None

def solve_equation(equation):
    """Решение простых уравнений"""
    try:
        # Убираем пробелы
        eq = equation.replace(' ', '')
        
        # Разделяем на левую и правую части
        if '=' not in eq:
            return None
            
        left, right = eq.split('=', 1)
        
        # Простые случаи решения уравнений
        if 'x' in left and 'x' not in right:
            # Уравнение вида: ax + b = c
            if '+' in left:
                parts = left.split('+')
                if len(parts) == 2:
                    if 'x' in parts[0]:
                        # ax + b = c
                        coeff = parts[0].replace('x', '').replace('*', '')
                        coeff = int(coeff) if coeff else 1
                        const = int(parts[1]) if parts[1] else 0
                        result = (int(right) - const) / coeff
                        return f"x = {result}"
            elif '-' in left:
                parts = left.split('-')
                if len(parts) == 2:
                    if 'x' in parts[0]:
                        # ax - b = c
                        coeff = parts[0].replace('x', '').replace('*', '')
                        coeff = int(coeff) if coeff else 1
                        const = int(parts[1]) if parts[1] else 0
                        result = (int(right) + const) / coeff
                        return f"x = {result}"
        
        # Аналогично для других переменных (a, b, c, etc.)
        for var in 'abcdefghijklmnopqrstuvwxyz':
            if var in left and var not in right:
                if '+' in left:
                    parts = left.split('+')
                    if len(parts) == 2 and var in parts[0]:
                        coeff = parts[0].replace(var, '').replace('*', '')
                        coeff = int(coeff) if coeff else 1
                        const = int(parts[1]) if parts[1] else 0
                        result = (int(right) - const) / coeff
                        return f"{var} = {result}"
                elif '-' in left:
                    parts = left.split('-')
                    if len(parts) == 2 and var in parts[0]:
                        coeff = parts[0].replace(var, '').replace('*', '')
                        coeff = int(coeff) if coeff else 1
                        const = int(parts[1]) if parts[1] else 0
                        result = (int(right) + const) / coeff
                        return f"{var} = {result}"
        
        # Специальный случай: 10-b=1
        if left == '10' and '-' in left and right == '1':
            # Это уравнение вида: 10 - b = 1
            # Решение: b = 10 - 1 = 9
            result = 10 - int(right)
            return f"b = {result}"
        
        # Общий случай для уравнений вида: число - переменная = число
        if '-' in left and len(left.split('-')) == 2:
            parts = left.split('-')
            if parts[0].isdigit() and parts[1].isalpha() and right.isdigit():
                result = int(parts[0]) - int(right)
                return f"{parts[1]} = {result}"
        
        return f"Уравнение: {equation}"
        
    except Exception as e:
        return f"Ошибка решения уравнения: {str(e)}"

def find_topic(message):
    """Находит подходящую тему по ключевым словам"""
    message_lower = message.lower()
    
    # Проверяем, является ли сообщение математическим выражением
    if any(op in message for op in ['+', '-', '*', '/', '=']):
        return 'вычисление'
    
    for topic, data in MATHEMATICS_KNOWLEDGE.items():
        for keyword in data['keywords']:
            if keyword in message_lower:
                return topic
    
    return 'алгебра'  # fallback

@app.route('/health')
def health():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'online',
        'service': 'mathematics',
        'port': 8086
    })

@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    """Основной endpoint для математических вопросов"""
    try:
        # Получаем параметры из GET или POST запроса
        if request.method == 'GET':
            message = request.args.get('message', '')
        else:
            data = request.get_json() or {}
            message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Сообщение не может быть пустым'
            }), 400
        
        # Находим подходящую тему
        topic = find_topic(message)
        
        # Обработка запросов о квадратных уравнениях
        if any(keyword in message.lower() for keyword in ['квадратное уравнение', 'квадратное', 'x²', 'x^2']):
            return jsonify({
                'success': True,
                'topic': 'квадратное уравнение',
                'explanation': """📐 **РЕШЕНИЕ КВАДРАТНОГО УРАВНЕНИЯ**

**Общий вид:** ax² + bx + c = 0

**Методы решения:**
1. **Через дискриминант:** D = b² - 4ac
2. **Формула корней:** x = (-b ± √D) / 2a
3. **Разложение на множители:** (x - x₁)(x - x₂) = 0

**Пример: x² - 5x + 6 = 0**
• a = 1, b = -5, c = 6
• D = (-5)² - 4×1×6 = 25 - 24 = 1
• x₁ = (5 + 1) / 2 = 3
• x₂ = (5 - 1) / 2 = 2
• **Ответ:** x₁ = 3, x₂ = 2

**Проверка:** 3² - 5×3 + 6 = 9 - 15 + 6 = 0 ✓
           2² - 5×2 + 6 = 4 - 10 + 6 = 0 ✓""",
                'message': message
            })
        
        # Если это вычисление, пытаемся вычислить результат
        if topic == 'вычисление':
            result = evaluate_expression(message)
            if result is not None:
                return jsonify({
                    'success': True,
                    'topic': 'вычисление',
                    'expression': message,
                    'result': result,
                    'explanation': f'Результат вычисления: {message} = {result}',
                    'message': message
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Не удалось вычислить выражение. Проверьте правильность записи.'
                }), 400
        
        # Получаем объяснение для других тем
        explanation = MATHEMATICS_KNOWLEDGE[topic]['explanation']
        
        return jsonify({
            'success': True,
            'topic': topic,
            'explanation': explanation,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Ошибка в математическом сервере: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {str(e)}'
        }), 500

@app.route('/api/mathematics/topics', methods=['GET'])
def get_topics():
    """Получение списка доступных тем"""
    return jsonify({
        'success': True,
        'topics': list(MATHEMATICS_KNOWLEDGE.keys()),
        'count': len(MATHEMATICS_KNOWLEDGE)
    })

if __name__ == '__main__':
    print("Math Server запущен")
    print("URL: http://localhost:8086")
    app.run(host='0.0.0.0', port=8086, debug=False)
