#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arduino Nano API Server для Rubin AI v2
API сервер для работы с базой данных Arduino Nano
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from datetime import datetime
from arduino_nano_integration import ArduinoNanoIntegration

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание Flask приложения
app = Flask(__name__)
CORS(app)

# Инициализация интеграции Arduino Nano
arduino_integration = None

def get_arduino_integration():
    """Получение экземпляра интеграции Arduino Nano"""
    global arduino_integration
    if arduino_integration is None:
        arduino_integration = ArduinoNanoIntegration()
    return arduino_integration

@app.route('/api/arduino/query', methods=['POST'])
def arduino_query():
    """Обработка запросов по Arduino Nano"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Запрос не может быть пустым'}), 400
        
        logger.info(f"🔧 Arduino Nano запрос: {query}")
        
        # Получаем ответ от интеграции
        integration = get_arduino_integration()
        response = integration.get_arduino_response(query)
        
        return jsonify({
            "module": "arduino_nano",
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса Arduino Nano: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/functions', methods=['GET'])
def get_functions():
    """Получение списка функций Arduino"""
    try:
        # Получаем список функций из базы данных
        integration = get_arduino_integration()
        functions = []
        cursor = integration.arduino_db.connection.cursor()
        cursor.execute("SELECT function_name, description, syntax FROM arduino_functions ORDER BY function_name")
        
        for row in cursor.fetchall():
            functions.append({
                "name": row[0],
                "description": row[1],
                "syntax": row[2]
            })
        
        return jsonify({
            "module": "arduino_nano",
            "functions": functions,
            "count": len(functions)
        })
        
    except Exception as e:
        logger.error(f"Ошибка при получении функций: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/libraries', methods=['GET'])
def get_libraries():
    """Получение списка библиотек Arduino"""
    try:
        # Получаем список библиотек из базы данных
        integration = get_arduino_integration()
        libraries = []
        cursor = integration.arduino_db.connection.cursor()
        cursor.execute("SELECT library_name, description, category FROM arduino_libraries ORDER BY library_name")
        
        for row in cursor.fetchall():
            libraries.append({
                "name": row[0],
                "description": row[1],
                "category": row[2]
            })
        
        return jsonify({
            "module": "arduino_nano",
            "libraries": libraries,
            "count": len(libraries)
        })
        
    except Exception as e:
        logger.error(f"Ошибка при получении библиотек: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/projects', methods=['GET'])
def get_projects():
    """Получение списка проектов Arduino"""
    try:
        # Получаем список проектов из базы данных
        integration = get_arduino_integration()
        projects = []
        cursor = integration.arduino_db.connection.cursor()
        cursor.execute("SELECT project_name, description, difficulty_level, category FROM arduino_projects ORDER BY difficulty_level")
        
        for row in cursor.fetchall():
            projects.append({
                "name": row[0],
                "description": row[1],
                "difficulty": row[2],
                "category": row[3]
            })
        
        return jsonify({
            "module": "arduino_nano",
            "projects": projects,
            "count": len(projects)
        })
        
    except Exception as e:
        logger.error(f"Ошибка при получении проектов: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/pins', methods=['GET'])
def get_pins():
    """Получение информации о пинах Arduino Nano"""
    try:
        pin_number = request.args.get('pin', type=int)
        
        if pin_number is not None:
            # Получаем информацию о конкретном пине
            integration = get_arduino_integration()
            pin_info = integration.arduino_db.get_pin_info(pin_number)
            if pin_info:
                return jsonify({
                    "module": "arduino_nano",
                    "pin": dict(pin_info)
                })
            else:
                return jsonify({'error': f'Пин {pin_number} не найден'}), 404
        else:
            # Получаем список всех пинов
            integration = get_arduino_integration()
            pins = []
            cursor = integration.arduino_db.connection.cursor()
            cursor.execute("SELECT pin_number, pin_type, description, special_functions FROM arduino_pins ORDER BY pin_number")
            
            for row in cursor.fetchall():
                pins.append({
                    "number": row[0],
                    "type": row[1],
                    "description": row[2],
                    "special_functions": row[3]
                })
            
            return jsonify({
                "module": "arduino_nano",
                "pins": pins,
                "count": len(pins)
            })
        
    except Exception as e:
        logger.error(f"Ошибка при получении информации о пинах: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/categories', methods=['GET'])
def get_categories():
    """Получение списка категорий Arduino"""
    try:
        integration = get_arduino_integration()
        categories = integration.arduino_db.get_categories()
        
        return jsonify({
            "module": "arduino_nano",
            "categories": categories,
            "count": len(categories)
        })
        
    except Exception as e:
        logger.error(f"Ошибка при получении категорий: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/troubleshooting', methods=['POST'])
def get_troubleshooting():
    """Получение решений проблем Arduino"""
    try:
        data = request.get_json()
        error_keywords = data.get('error', '')
        
        if not error_keywords:
            return jsonify({'error': 'Описание проблемы не может быть пустым'}), 400
        
        integration = get_arduino_integration()
        solutions = integration.arduino_db.get_troubleshooting(error_keywords)
        
        return jsonify({
            "module": "arduino_nano",
            "error": error_keywords,
            "solutions": solutions,
            "count": len(solutions)
        })
        
    except Exception as e:
        logger.error(f"Ошибка при получении решений проблем: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/search', methods=['POST'])
def search_knowledge():
    """Поиск в базе знаний Arduino"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        category = data.get('category', None)
        
        if not query:
            return jsonify({'error': 'Поисковый запрос не может быть пустым'}), 400
        
        integration = get_arduino_integration()
        results = integration.arduino_db.search_knowledge(query, category)
        
        return jsonify({
            "module": "arduino_nano",
            "query": query,
            "category": category,
            "results": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Ошибка при поиске в базе знаний: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/arduino/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера Arduino Nano"""
    try:
        # Проверяем соединение с базой данных
        integration = get_arduino_integration()
        cursor = integration.arduino_db.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM arduino_categories")
        categories_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM arduino_knowledge")
        knowledge_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM arduino_functions")
        functions_count = cursor.fetchone()[0]
        
        return jsonify({
            "module": "arduino_nano",
            "status": "healthy",
            "database": "connected",
            "categories": categories_count,
            "knowledge_items": knowledge_count,
            "functions": functions_count,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка при проверке здоровья: {e}")
        return jsonify({
            "module": "arduino_nano",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/arduino/status', methods=['GET'])
def get_status():
    """Получение статуса модуля Arduino Nano"""
    return jsonify({
        "module": "arduino_nano",
        "name": "Arduino Nano Programming Module",
        "version": "1.0.0",
        "description": "Модуль для работы с Arduino Nano - программирование, функции, библиотеки, проекты",
        "endpoints": [
            "POST /api/arduino/query - Обработка запросов",
            "GET /api/arduino/functions - Список функций",
            "GET /api/arduino/libraries - Список библиотек",
            "GET /api/arduino/projects - Список проектов",
            "GET /api/arduino/pins - Информация о пинах",
            "GET /api/arduino/categories - Список категорий",
            "POST /api/arduino/troubleshooting - Решение проблем",
            "POST /api/arduino/search - Поиск в базе знаний",
            "GET /api/arduino/health - Проверка здоровья",
            "GET /api/arduino/status - Статус модуля"
        ],
        "features": [
            "База знаний по Arduino Nano",
            "Функции и библиотеки",
            "Готовые проекты",
            "Информация о пинах",
            "Решение проблем",
            "Поиск по базе знаний"
        ]
    })

if __name__ == '__main__':
    print("🔧 Arduino Nano API Server запущен")
    print("URL: http://localhost:8110")
    print("Endpoints:")
    print("  - POST /api/arduino/query - Обработка запросов")
    print("  - GET /api/arduino/functions - Список функций")
    print("  - GET /api/arduino/libraries - Список библиотек")
    print("  - GET /api/arduino/projects - Список проектов")
    print("  - GET /api/arduino/pins - Информация о пинах")
    print("  - GET /api/arduino/categories - Список категорий")
    print("  - POST /api/arduino/troubleshooting - Решение проблем")
    print("  - POST /api/arduino/search - Поиск в базе знаний")
    print("  - GET /api/arduino/health - Проверка здоровья")
    print("  - GET /api/arduino/status - Статус модуля")
    
    app.run(host='0.0.0.0', port=8110, debug=True)
