#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Mathematics API Server
Продвинутые математические вычисления и анализ
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import re
import logging
from datetime import datetime
import json

# Попытка импорта математических модулей
try:
    import numpy as np
    import sympy as sp
    ADVANCED_MATH_AVAILABLE = True
except ImportError:
    ADVANCED_MATH_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def solve_quadratic_equation(a, b, c):
    """Решение квадратного уравнения ax² + bx + c = 0"""
    discriminant = b**2 - 4*a*c
    
    if discriminant > 0:
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        return {
            "type": "two_real_roots",
            "roots": [x1, x2],
            "discriminant": discriminant
        }
    elif discriminant == 0:
        x = -b / (2*a)
        return {
            "type": "one_real_root",
            "roots": [x],
            "discriminant": discriminant
        }
    else:
        real_part = -b / (2*a)
        imag_part = math.sqrt(-discriminant) / (2*a)
        return {
            "type": "complex_roots",
            "roots": [complex(real_part, imag_part), complex(real_part, -imag_part)],
            "discriminant": discriminant
        }

def solve_system_of_equations(equations):
    """Решение системы линейных уравнений"""
    if not ADVANCED_MATH_AVAILABLE:
        return {"error": "NumPy не доступен для решения систем уравнений"}
    
    try:
        # Простое решение для систем вида ax + by = c
        if len(equations) == 2:
            eq1, eq2 = equations
            # Извлекаем коэффициенты (упрощенная версия)
            # В реальной реализации нужен парсер уравнений
            return {
                "type": "system_solution",
                "equations": equations,
                "solution": "Решение системы уравнений (требует парсер)"
            }
    except Exception as e:
        return {"error": f"Ошибка решения системы: {str(e)}"}

def calculate_integral(expression, variable='x'):
    """Вычисление интеграла"""
    if not ADVANCED_MATH_AVAILABLE:
        return {"error": "SymPy не доступен для вычисления интегралов"}
    
    try:
        x = sp.Symbol(variable)
        expr = sp.sympify(expression)
        integral = sp.integrate(expr, x)
        
        return {
            "expression": str(expr),
            "integral": str(integral),
            "variable": variable
        }
    except Exception as e:
        return {"error": f"Ошибка вычисления интеграла: {str(e)}"}

def calculate_derivative(expression, variable='x'):
    """Вычисление производной"""
    if not ADVANCED_MATH_AVAILABLE:
        return {"error": "SymPy не доступен для вычисления производных"}
    
    try:
        x = sp.Symbol(variable)
        expr = sp.sympify(expression)
        derivative = sp.diff(expr, x)
        
        return {
            "expression": str(expr),
            "derivative": str(derivative),
            "variable": variable
        }
    except Exception as e:
        return {"error": f"Ошибка вычисления производной: {str(e)}"}

def analyze_function(expression, variable='x'):
    """Анализ функции"""
    if not ADVANCED_MATH_AVAILABLE:
        return {"error": "SymPy не доступен для анализа функций"}
    
    try:
        x = sp.Symbol(variable)
        expr = sp.sympify(expression)
        
        # Находим критические точки
        derivative = sp.diff(expr, x)
        critical_points = sp.solve(derivative, x)
        
        # Находим точки перегиба
        second_derivative = sp.diff(derivative, x)
        inflection_points = sp.solve(second_derivative, x)
        
        return {
            "expression": str(expr),
            "derivative": str(derivative),
            "second_derivative": str(second_derivative),
            "critical_points": [str(cp) for cp in critical_points],
            "inflection_points": [str(ip) for ip in inflection_points],
            "variable": variable
        }
    except Exception as e:
        return {"error": f"Ошибка анализа функции: {str(e)}"}

def parse_mathematical_expression(text):
    """Парсинг математического выражения из текста"""
    # Простые паттерны для извлечения уравнений
    patterns = {
        'quadratic': r'(\d*\.?\d*)\s*x\s*\*\*\s*2\s*\+\s*(\d*\.?\d*)\s*x\s*\+\s*(\d*\.?\d*)\s*=\s*0',
        'linear': r'(\d*\.?\d*)\s*x\s*\+\s*(\d*\.?\d*)\s*=\s*(\d*\.?\d*)',
        'integral': r'∫\s*([^d]+)\s*d([a-zA-Z])',
        'derivative': r'd\/(d[a-zA-Z])\s*\(([^)]+)\)'
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            results[pattern_name] = matches
    
    return results

@app.route('/api/math/advanced', methods=['POST'])
def advanced_math():
    """Продвинутые математические вычисления"""
    try:
        data = request.get_json()
        equation = data.get('equation', '')
        math_type = data.get('type', 'auto')
        
        if not equation:
            return jsonify({
                "success": False,
                "error": "Уравнение не может быть пустым"
            }), 400
        
        # Парсинг выражения
        parsed = parse_mathematical_expression(equation)
        
        result = {
            "equation": equation,
            "type": math_type,
            "parsed": parsed,
            "calculations": {}
        }
        
        # Определяем тип задачи и решаем
        if 'quadratic' in parsed:
            coeffs = parsed['quadratic'][0]
            a, b, c = float(coeffs[0]) if coeffs[0] else 1, float(coeffs[1]), float(coeffs[2])
            result["calculations"]["quadratic"] = solve_quadratic_equation(a, b, c)
        
        if 'integral' in parsed:
            integral_data = parsed['integral'][0]
            expression, variable = integral_data[0], integral_data[1]
            result["calculations"]["integral"] = calculate_integral(expression, variable)
        
        if 'derivative' in parsed:
            derivative_data = parsed['derivative'][0]
            variable, expression = derivative_data[0][1:], derivative_data[1]  # убираем 'd' из переменной
            result["calculations"]["derivative"] = calculate_derivative(expression, variable)
        
        # Если ничего не распознано, пробуем анализ функции
        if not result["calculations"] and ADVANCED_MATH_AVAILABLE:
            try:
                result["calculations"]["function_analysis"] = analyze_function(equation)
            except:
                pass
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка математических вычислений: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка вычислений: {str(e)}"
        }), 500

@app.route('/api/math/solve', methods=['POST'])
def solve_equation():
    """Решение уравнений"""
    try:
        data = request.get_json()
        equation = data.get('equation', '')
        
        if not equation:
            return jsonify({
                "success": False,
                "error": "Уравнение не может быть пустым"
            }), 400
        
        # Простое решение квадратных уравнений
        if 'x²' in equation or 'x^2' in equation:
            # Извлекаем коэффициенты из текста
            coeffs = re.findall(r'([+-]?\d*\.?\d*)', equation)
            try:
                a = float(coeffs[0]) if coeffs[0] else 1
                b = float(coeffs[1]) if len(coeffs) > 1 else 0
                c = float(coeffs[2]) if len(coeffs) > 2 else 0
                
                solution = solve_quadratic_equation(a, b, c)
                
                return jsonify({
                    "success": True,
                    "equation": equation,
                    "coefficients": {"a": a, "b": b, "c": c},
                    "solution": solution,
                    "timestamp": datetime.now().isoformat()
                })
            except:
                pass
        
        return jsonify({
            "success": False,
            "error": "Не удалось распознать тип уравнения"
        }), 400
        
    except Exception as e:
        logger.error(f"Ошибка решения уравнения: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка решения: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья Advanced Math API"""
    return jsonify({
        'service': 'Advanced Math API',
        'status': 'healthy',
        'port': 8100,
        'version': '1.0',
        'capabilities': ['calculus', 'linear_algebra', 'statistics', 'optimization']
    })

@app.route('/api/math/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "Advanced Mathematics API",
        "status": "online",
        "version": "1.0",
        "advanced_math_available": ADVANCED_MATH_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/math/status', methods=['GET'])
def get_status():
    """Статус сервера"""
    return jsonify({
        "service": "Advanced Mathematics API",
        "status": "running",
        "port": 8100,
        "endpoints": [
            "/api/math/advanced",
            "/api/math/solve",
            "/api/math/health",
            "/api/math/status"
        ],
        "capabilities": [
            "Решение квадратных уравнений",
            "Вычисление интегралов",
            "Вычисление производных",
            "Анализ функций",
            "Решение систем уравнений"
        ],
        "dependencies": {
            "numpy": ADVANCED_MATH_AVAILABLE,
            "sympy": ADVANCED_MATH_AVAILABLE
        }
    })

if __name__ == '__main__':
    print("🧮 Advanced Mathematics API Server запущен")
    print("URL: http://localhost:8100")
    print("Endpoints:")
    print("  - POST /api/math/advanced - Продвинутые вычисления")
    print("  - POST /api/math/solve - Решение уравнений")
    print("  - GET /api/math/health - Проверка здоровья")
    print("  - GET /api/math/status - Статус сервера")
    app.run(host='0.0.0.0', port=8100, debug=False)


