#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расширенный математический сервер с поддержкой физических и химических формул
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import logging
import time
from mathematical_problem_solver import MathematicalProblemSolver

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Инициализация расширенного математического решателя
math_solver = MathematicalProblemSolver()

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
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Math Server',
        'port': 8086,
        'timestamp': time.time()
    })

def solve_quadratic(message):
    """Решает квадратные уравнения"""
    # Квадратные уравнения: x² - 5x + 6 = 0, x^2 + 3x - 4 = 0
    quadratic_pattern = r'x[²²^2]\s*([+\-]?)\s*(\d*)\s*x\s*([+\-]?)\s*(\d*)\s*=\s*(\d+)'
    quadratic_match = re.search(quadratic_pattern, message)
    
    if quadratic_match:
        try:
            # Извлекаем коэффициенты
            a_sign = quadratic_match.group(1) or '+'
            b_coeff = quadratic_match.group(2) or '1'
            c_sign = quadratic_match.group(3) or '+'
            c_coeff = quadratic_match.group(4) or '0'
            right_side = int(quadratic_match.group(5))
            
            # Преобразуем в стандартную форму ax² + bx + c = 0
            a = 1  # коэффициент при x² всегда 1 в нашем случае
            b = int(a_sign + b_coeff) if b_coeff else 0
            c = int(c_sign + c_coeff) if c_coeff else 0
            
            # Переносим правую часть влево
            c = c - right_side
            
            # Вычисляем дискриминант
            discriminant = b**2 - 4*a*c
            
            if discriminant > 0:
                # Два различных корня
                x1 = (-b + discriminant**0.5) / (2*a)
                x2 = (-b - discriminant**0.5) / (2*a)
                
                return f"""📐 **Решение квадратного уравнения:**

**Уравнение:** x² {f'+ {b}x' if b > 0 else f'- {abs(b)}x' if b < 0 else ''} {f'+ {c}' if c > 0 else f'- {abs(c)}' if c < 0 else ''} = 0
**Коэффициенты:** a = {a}, b = {b}, c = {c}
**Дискриминант:** D = b² - 4ac = {b}² - 4×{a}×{c} = {discriminant}

**Корни:**
• x₁ = (-b + √D) / 2a = (-{b} + √{discriminant}) / 2 = {x1:.2f}
• x₂ = (-b - √D) / 2a = (-{b} - √{discriminant}) / 2 = {x2:.2f}

✅ **Ответ: x₁ = {x1:.2f}, x₂ = {x2:.2f}**"""
            
            elif discriminant == 0:
                # Один корень
                x = -b / (2*a)
                
                return f"""📐 **Решение квадратного уравнения:**

**Уравнение:** x² {f'+ {b}x' if b > 0 else f'- {abs(b)}x' if b < 0 else ''} {f'+ {c}' if c > 0 else f'- {abs(c)}' if c < 0 else ''} = 0
**Коэффициенты:** a = {a}, b = {b}, c = {c}
**Дискриминант:** D = b² - 4ac = {b}² - 4×{a}×{c} = 0

**Корень:** x = -b / 2a = -{b} / 2 = {x:.2f}

✅ **Ответ: x = {x:.2f}**"""
            
            else:
                # Нет действительных корней
                return f"""📐 **Решение квадратного уравнения:**

**Уравнение:** x² {f'+ {b}x' if b > 0 else f'- {abs(b)}x' if b < 0 else ''} {f'+ {c}' if c > 0 else f'- {abs(c)}' if c < 0 else ''} = 0
**Коэффициенты:** a = {a}, b = {b}, c = {c}
**Дискриминант:** D = b² - 4ac = {b}² - 4×{a}×{c} = {discriminant} < 0

❌ **Ответ: Уравнение не имеет действительных корней**"""
                
        except Exception as e:
            logger.error(f"Ошибка решения квадратного уравнения: {e}")
            return f"❌ Ошибка при решении уравнения: {str(e)}"
    
    return None

@app.route('/')
def index():
    return jsonify({
        'name': 'Math Server',
        'version': '1.0',
        'status': 'online',
        'features': ['quadratic_equations']
    })

@app.route('/api/chat', methods=['GET', 'POST'])
def chat():
    """Основной endpoint для чата с расширенным математическим решателем"""
    try:
        if request.method == 'GET':
            message = request.args.get('message', '').strip()
        else:
            data = request.get_json()
            message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        logger.info(f"🧮 Обработка математической задачи: {message}")
        
        # Используем расширенный математический решатель
        result = math_solver.solve_problem(message)
        
        # Форматируем ответ
        response_text = f"""🧮 **Math Server с расширенными возможностями:**

**Задача:** {message}

**Тип задачи:** {result.problem_type.value}
**Ответ:** {result.final_answer}
**Уверенность:** {result.confidence:.1%}

**Пошаговое решение:**
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(result.solution_steps)])}

**Объяснение:** {result.explanation}

*Решение выполнено с помощью расширенного математического решателя Rubin AI*"""
        
        return jsonify({
            'success': True,
            'response': response_text,
            'category': result.problem_type.value,
            'confidence': result.confidence,
            'problem_type': result.problem_type.value,
            'final_answer': str(result.final_answer)
        })
        
    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/graph/analyze', methods=['POST'])
def analyze_graph():
    """Анализ графиков и изображений"""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        
        if not image_path:
            return jsonify({'error': 'Путь к изображению не указан'}), 400
        
        logger.info(f"📊 Анализ графика: {image_path}")
        
        # Проверяем существование файла
        import os
        if not os.path.exists(image_path):
            return jsonify({
                'error': 'Файл изображения не найден',
                'image_path': image_path,
                'solution': 'Укажите правильный путь к файлу изображения'
            }), 404
        
        # Простой анализ (заглушка)
        analysis_result = f"""📊 **Анализ графика:**

**Файл:** {image_path}
**Статус:** Файл найден

**Рекомендации:**
1. Убедитесь, что изображение содержит график или диаграмму
2. Для лучшего анализа используйте четкие изображения
3. Поддерживаемые форматы: PNG, JPG, JPEG

**Возможности анализа:**
- Извлечение данных с графиков
- Определение трендов
- Вычисление значений в точках
- Анализ функций

*Для полного анализа требуется установка Tesseract OCR*"""
        
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'image_path': image_path,
            'status': 'analyzed'
        })
        
    except Exception as e:
        logger.error(f"Ошибка анализа графика: {e}")
        return jsonify({'error': f'Ошибка анализа: {str(e)}'}), 500

if __name__ == '__main__':
    print("Math Server запущен")
    print("URL: http://localhost:8086")
    app.run(host='0.0.0.0', port=8086, debug=False)

