#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 API для математического решателя Rubin
=========================================

REST API для интеграции математического решателя с системой Rubin AI.
Предоставляет endpoints для решения различных типов математических задач.

Автор: Rubin AI System
Версия: 2.0
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
from typing import Dict, Any
import traceback

from mathematical_problem_solver import MathematicalProblemSolver, ProblemType

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание Flask приложения
app = Flask(__name__)
CORS(app)

# Инициализация решателя
solver = MathematicalProblemSolver()

@app.route('/api/mathematics/status', methods=['GET'])
def get_status():
    """Получение статуса математического модуля"""
    try:
        return jsonify({
            "success": True,
            "module": "Mathematical Problem Solver",
            "version": "2.0",
            "status": "active",
            "capabilities": [
                "arithmetic",
                "linear_equations", 
                "quadratic_equations",
                "system_equations",
                "geometry",
                "trigonometry",
                "calculus",
                "statistics",
                "percentage"
            ],
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/mathematics/solve', methods=['POST'])
def solve_problem():
    """Решение математической задачи"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        problem_text = data.get('problem', '')
        if not problem_text:
            return jsonify({
                "success": False,
                "error": "Problem text is required"
            }), 400
        
        logger.info(f"Solving problem: {problem_text}")
        
        # Решение задачи
        start_time = time.time()
        solution = solver.solve_problem(problem_text)
        processing_time = time.time() - start_time
        
        # Форматирование ответа
        response = {
            "success": True,
            "data": {
                "problem": problem_text,
                "problem_type": solution.problem_type.value,
                "solution_steps": solution.solution_steps,
                "final_answer": solution.final_answer,
                "verification": solution.verification,
                "confidence": solution.confidence,
                "explanation": solution.explanation,
                "processing_time": processing_time
            },
            "timestamp": time.time()
        }
        
        logger.info(f"Problem solved in {processing_time:.3f}s with confidence {solution.confidence:.2f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Solve error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/mathematics/calculate', methods=['POST'])
def calculate_expression():
    """Вычисление математического выражения"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        expression = data.get('expression', '')
        if not expression:
            return jsonify({
                "success": False,
                "error": "Expression is required"
            }), 400
        
        logger.info(f"Calculating expression: {expression}")
        
        # Решение как арифметической задачи
        start_time = time.time()
        solution = solver.solve_problem(f"Вычисли {expression}")
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "data": {
                "expression": expression,
                "result": solution.final_answer,
                "steps": solution.solution_steps,
                "processing_time": processing_time
            },
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Calculate error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/mathematics/equation', methods=['POST'])
def solve_equation():
    """Решение уравнения"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        equation = data.get('equation', '')
        if not equation:
            return jsonify({
                "success": False,
                "error": "Equation is required"
            }), 400
        
        logger.info(f"Solving equation: {equation}")
        
        # Решение уравнения
        start_time = time.time()
        solution = solver.solve_problem(f"Реши уравнение {equation}")
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "data": {
                "equation": equation,
                "solution": solution.final_answer,
                "steps": solution.solution_steps,
                "equation_type": solution.problem_type.value,
                "verification": solution.verification,
                "processing_time": processing_time
            },
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Equation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/mathematics/geometry', methods=['POST'])
def solve_geometry():
    """Решение геометрических задач"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        problem = data.get('problem', '')
        shape = data.get('shape', '')
        dimensions = data.get('dimensions', {})
        
        if not problem and not shape:
            return jsonify({
                "success": False,
                "error": "Problem description or shape is required"
            }), 400
        
        # Формирование текста задачи
        if not problem:
            problem = f"Найди площадь {shape}"
            if dimensions:
                dim_str = ", ".join([f"{k} {v}" for k, v in dimensions.items()])
                problem += f" с {dim_str}"
        
        logger.info(f"Solving geometry problem: {problem}")
        
        # Решение геометрической задачи
        start_time = time.time()
        solution = solver.solve_problem(problem)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "data": {
                "problem": problem,
                "shape": shape,
                "dimensions": dimensions,
                "result": solution.final_answer,
                "steps": solution.solution_steps,
                "verification": solution.verification,
                "processing_time": processing_time
            },
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Geometry error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/mathematics/trigonometry', methods=['POST'])
def solve_trigonometry():
    """Решение тригонометрических задач"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        problem = data.get('problem', '')
        angle = data.get('angle', 0)
        unit = data.get('unit', 'degrees')
        function = data.get('function', '')
        
        if not problem and not function:
            return jsonify({
                "success": False,
                "error": "Problem description or function is required"
            }), 400
        
        # Формирование текста задачи
        if not problem:
            problem = f"Вычисли {function}({angle}°)"
        
        logger.info(f"Solving trigonometry problem: {problem}")
        
        # Решение тригонометрической задачи
        start_time = time.time()
        solution = solver.solve_problem(problem)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "data": {
                "problem": problem,
                "angle": angle,
                "unit": unit,
                "function": function,
                "result": solution.final_answer,
                "steps": solution.solution_steps,
                "processing_time": processing_time
            },
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Trigonometry error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/mathematics/statistics', methods=['POST'])
def solve_statistics():
    """Решение статистических задач"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        problem = data.get('problem', '')
        numbers = data.get('data', [])
        
        if not problem and not numbers:
            return jsonify({
                "success": False,
                "error": "Problem description or data is required"
            }), 400
        
        # Формирование текста задачи
        if not problem:
            problem = f"Найди статистические характеристики чисел {numbers}"
        
        logger.info(f"Solving statistics problem: {problem}")
        
        # Решение статистической задачи
        start_time = time.time()
        solution = solver.solve_problem(problem)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "data": {
                "problem": problem,
                "input_data": numbers,
                "result": solution.final_answer,
                "steps": solution.solution_steps,
                "processing_time": processing_time
            },
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/mathematics/percentage', methods=['POST'])
def solve_percentage():
    """Решение задач на проценты"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        problem = data.get('problem', '')
        number = data.get('number', 0)
        percent = data.get('percent', 0)
        
        if not problem and (not number or not percent):
            return jsonify({
                "success": False,
                "error": "Problem description or number and percent are required"
            }), 400
        
        # Формирование текста задачи
        if not problem:
            problem = f"Найди {percent}% от {number}"
        
        logger.info(f"Solving percentage problem: {problem}")
        
        # Решение задачи на проценты
        start_time = time.time()
        solution = solver.solve_problem(problem)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "data": {
                "problem": problem,
                "number": number,
                "percent": percent,
                "result": solution.final_answer,
                "steps": solution.solution_steps,
                "processing_time": processing_time
            },
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Percentage error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/mathematics/help', methods=['GET'])
def get_help():
    """Получение справки по API"""
    help_info = {
        "success": True,
        "data": {
            "title": "Mathematical Problem Solver API",
            "version": "2.0",
            "description": "API для решения различных типов математических задач",
            "endpoints": {
                "GET /api/mathematics/status": "Статус модуля",
                "GET /api/mathematics/help": "Справка по API",
                "POST /api/mathematics/solve": "Решение любой математической задачи",
                "POST /api/mathematics/calculate": "Вычисление математического выражения",
                "POST /api/mathematics/equation": "Решение уравнений",
                "POST /api/mathematics/geometry": "Решение геометрических задач",
                "POST /api/mathematics/trigonometry": "Решение тригонометрических задач",
                "POST /api/mathematics/statistics": "Решение статистических задач",
                "POST /api/mathematics/percentage": "Решение задач на проценты"
            },
            "supported_problem_types": [
                "arithmetic",
                "linear_equations",
                "quadratic_equations", 
                "system_equations",
                "geometry",
                "trigonometry",
                "calculus",
                "statistics",
                "percentage"
            ],
            "examples": {
                "solve": {
                    "url": "/api/mathematics/solve",
                    "method": "POST",
                    "body": {"problem": "Реши уравнение 2x + 5 = 13"}
                },
                "calculate": {
                    "url": "/api/mathematics/calculate", 
                    "method": "POST",
                    "body": {"expression": "2 + 3 * 4"}
                },
                "geometry": {
                    "url": "/api/mathematics/geometry",
                    "method": "POST", 
                    "body": {
                        "shape": "треугольник",
                        "dimensions": {"base": 5, "height": 3}
                    }
                }
            }
        },
        "timestamp": time.time()
    }
    
    return jsonify(help_info)

@app.route('/api/mathematics/test', methods=['POST'])
def run_tests():
    """Запуск тестов математического решателя"""
    try:
        test_problems = [
            "Вычисли 2 + 3 * 4",
            "Реши уравнение 2x + 5 = 13", 
            "Найди площадь треугольника с основанием 5 и высотой 3",
            "Реши квадратное уравнение x² - 5x + 6 = 0",
            "Найди 15% от 200",
            "Вычисли sin(30°)",
            "Найди среднее значение чисел 1, 2, 3, 4, 5"
        ]
        
        results = []
        total_time = 0
        
        for i, problem in enumerate(test_problems):
            start_time = time.time()
            solution = solver.solve_problem(problem)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            results.append({
                "test_number": i + 1,
                "problem": problem,
                "problem_type": solution.problem_type.value,
                "answer": solution.final_answer,
                "confidence": solution.confidence,
                "verification": solution.verification,
                "processing_time": processing_time
            })
        
        response = {
            "success": True,
            "data": {
                "total_tests": len(test_problems),
                "total_time": total_time,
                "average_time": total_time / len(test_problems),
                "results": results
            },
            "timestamp": time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Обработка 404 ошибок"""
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": [
            "/api/mathematics/status",
            "/api/mathematics/help",
            "/api/mathematics/solve",
            "/api/mathematics/calculate",
            "/api/mathematics/equation",
            "/api/mathematics/geometry",
            "/api/mathematics/trigonometry",
            "/api/mathematics/statistics",
            "/api/mathematics/percentage",
            "/api/mathematics/test"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Обработка 500 ошибок"""
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "Произошла внутренняя ошибка сервера"
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Запуск Mathematical Problem Solver API")
    logger.info("📊 Доступные endpoints:")
    logger.info("   GET  /api/mathematics/status - Статус модуля")
    logger.info("   GET  /api/mathematics/help - Справка")
    logger.info("   POST /api/mathematics/solve - Решение задач")
    logger.info("   POST /api/mathematics/calculate - Вычисления")
    logger.info("   POST /api/mathematics/equation - Уравнения")
    logger.info("   POST /api/mathematics/geometry - Геометрия")
    logger.info("   POST /api/mathematics/trigonometry - Тригонометрия")
    logger.info("   POST /api/mathematics/statistics - Статистика")
    logger.info("   POST /api/mathematics/percentage - Проценты")
    logger.info("   POST /api/mathematics/test - Тесты")
    
    app.run(
        host='0.0.0.0',
        port=8089,
        debug=True
    )

