#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
⚡ ELECTRICAL SERVER
===================
Сервер для электротехнических задач
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/electrical/status', methods=['GET'])
def electrical_status():
    """Статус модуля электротехники"""
    return jsonify({
        "module": "electrical",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "capabilities": [
            "Расчет электрических цепей",
            "Анализ схем",
            "Решение задач по электротехнике"
        ]
    })

@app.route('/api/electrical/solve', methods=['POST'])
def solve_electrical():
    """Решение электротехнических задач"""
    try:
        data = request.get_json()
        problem = data.get('problem', '')
        
        logger.info(f"⚡ Получена электротехническая задача: {problem[:50]}...")
        
        # Простая логика решения
        if "закон ома" in problem.lower() or "напряжение" in problem.lower():
            result = "U = I × R (Закон Ома)"
        elif "мощность" in problem.lower():
            result = "P = U × I (Мощность)"
        elif "сопротивление" in problem.lower():
            result = "R = U / I (Сопротивление)"
        else:
            result = "Электротехническая задача требует дополнительного анализа"
        
        return jsonify({
            "module": "electrical",
            "problem": problem,
            "solution": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в electrical: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "electrical",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

if __name__ == "__main__":
    print("⚡ Electrical Server запущен")
    print("URL: http://localhost:8087")
    print("Доступные эндпоинты:")
    print("  - GET /api/electrical/status - статус модуля")
    print("  - POST /api/electrical/solve - решение задач")
    print("  - GET /api/health - проверка здоровья")
    app.run(host='0.0.0.0', port=8087, debug=True)





