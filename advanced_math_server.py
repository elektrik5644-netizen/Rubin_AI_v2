#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Math Server для продвинутых математических вычислений
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime
import math

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'healthy',
        'service': 'Advanced Math Server',
        'port': 8100,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/math/advanced', methods=['GET', 'POST'])
def advanced_math():
    """Продвинутые математические вычисления"""
    try:
        if request.method == 'GET':
            equation = request.args.get('equation', '')
        else:
            data = request.get_json()
            equation = data.get('equation', '')
        
        logger.info(f"🧮 Получен запрос продвинутой математики: {equation[:50]}...")
        
        # Простая логика вычислений
        result = {
            'status': 'success',
            'equation': equation,
            'solution': '',
            'steps': [],
            'service': 'advanced_math',
            'timestamp': datetime.now().isoformat()
        }
        
        if equation:
            equation_lower = equation.lower()
            
            if 'квадратное' in equation_lower or 'quadratic' in equation_lower:
                result['solution'] = "Квадратное уравнение: ax² + bx + c = 0"
                result['steps'] = [
                    "1. Определить коэффициенты a, b, c",
                    "2. Вычислить дискриминант: D = b² - 4ac",
                    "3. Если D > 0: два корня",
                    "4. Если D = 0: один корень",
                    "5. Если D < 0: комплексные корни"
                ]
            
            elif 'интеграл' in equation_lower or 'integral' in equation_lower:
                result['solution'] = "Интегральное исчисление"
                result['steps'] = [
                    "1. Определить тип интеграла",
                    "2. Применить подходящий метод",
                    "3. Методы: замена переменной, по частям",
                    "4. Проверить результат дифференцированием"
                ]
            
            elif 'производная' in equation_lower or 'derivative' in equation_lower:
                result['solution'] = "Дифференциальное исчисление"
                result['steps'] = [
                    "1. Определить функцию f(x)",
                    "2. Применить правила дифференцирования",
                    "3. Правила: константа, степень, сумма",
                    "4. Проверить результат"
                ]
            
            else:
                result['solution'] = "Продвинутые математические вычисления"
                result['steps'] = [
                    "1. Анализ уравнения",
                    "2. Выбор метода решения",
                    "3. Применение алгоритма",
                    "4. Проверка результата"
                ]
        else:
            result['solution'] = "Отправьте уравнение для анализа"
            result['steps'] = ["Поддерживаемые типы: квадратные уравнения, интегралы, производные"]
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка вычислений: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🧮 Advanced Math Server запущен")
    print("URL: http://localhost:8100")
    print("Доступные эндпоинты:")
    print("  - GET/POST /api/math/advanced - продвинутые вычисления")
    print("  - GET /api/health - проверка здоровья")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8100, debug=False)