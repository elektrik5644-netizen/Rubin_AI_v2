#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubin AI v2.0 - Исправленный основной API сервер
С интеграцией математического решателя
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import pickle

# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'providers'))

# Импорты с обработкой ошибок
try:
    from config import Config
except ImportError:
    # Базовые настройки если config.py недоступен
    class Config:
        ALLOWED_ORIGINS = ["*"]
        LOG_LEVEL = "INFO"
        LOG_FILE = "rubin_ai.log"
        DOCUMENTS_STORAGE = "documents.pkl"

# Безопасный импорт провайдеров
provider_selector = None
documents_storage = []

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app, 
     origins=Config.ALLOWED_ORIGINS,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     supports_credentials=True)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rubin_ai_v2")

# Инициализация математического решателя
mathematical_solver = None

def initialize_mathematical_solver():
    """Инициализация математического решателя"""
    global mathematical_solver
    
    try:
        from mathematical_solver.integrated_solver import IntegratedMathematicalSolver, MathIntegrationConfig
        
        config = MathIntegrationConfig(
            enabled=True,
            confidence_threshold=0.7,
            fallback_to_general=False,
            log_requests=True,
            response_format="structured"
        )
        
        mathematical_solver = IntegratedMathematicalSolver(config)
        logger.info("✅ Математический решатель инициализирован")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Не удалось импортировать математический решатель: {e}")
        try:
            # Fallback к простому решателю
            from mathematical_problem_solver import MathematicalProblemSolver
            mathematical_solver = MathematicalProblemSolver()
            logger.info("✅ Простой математический решатель инициализирован")
            return True
        except ImportError as e2:
            logger.error(f"❌ Не удалось инициализировать математический решатель: {e2}")
            mathematical_solver = None
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации математического решателя: {e}")
        mathematical_solver = None
        return False

def is_mathematical_request(message: str) -> bool:
    """Проверка, является ли запрос математическим"""
    if not message or not isinstance(message, str):
        return False
    
    message_lower = message.lower().strip()
    
    # Простые математические операции
    import re
    math_patterns = [
        r'^\d+\s*[+\-*/]\s*\d+.*[=?]?$',  # 2+4, 3-1, 5*2, 8/2
        r'^\d+\s*[+\-*/]\s*\d+$',          # 2+4, 3-1 (без знака равенства)
        r'\d+\s*[+\-*/]\s*\d+',            # в тексте
        r'сколько.*\d+',                    # сколько яблок, сколько деревьев
        r'вычисли\s+\d+',                   # вычисли 2+3
        r'реши\s+\d+',                      # реши 5-2
        r'найди\s+\d+',                     # найди 3*4
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, message_lower):
            return True
    
    # Ключевые слова
    math_keywords = [
        'сколько', 'вычисли', 'найди', 'реши', 'задача',
        'скорость', 'время', 'расстояние', 'путь',
        'угол', 'градус', 'смежные', 'сумма',
        'деревьев', 'яблон', 'груш', 'слив',
        'м/с', 'км/ч', '°', '+', '-', '*', '/', '='
    ]
    
    for keyword in math_keywords:
        if keyword in message_lower:
            return True
    
    return False

def solve_mathematical_problem(message: str) -> dict:
    """Решение математической задачи"""
    global mathematical_solver
    
    if not mathematical_solver:
        return {
            "success": False,
            "error": "Математический решатель недоступен",
            "answer": None
        }
    
    try:
        # Используем интегрированный решатель если доступен
        if hasattr(mathematical_solver, 'process_request'):
            result = mathematical_solver.process_request(message)
            
            if result.get("success"):
                solution_data = result.get("solution_data", {})
                return {
                    "success": True,
                    "answer": solution_data.get("final_answer", "Не удалось получить ответ"),
                    "confidence": solution_data.get("confidence", 0.0),
                    "problem_type": solution_data.get("problem_type", "unknown"),
                    "explanation": solution_data.get("explanation", ""),
                    "provider": "Mathematical Solver (Integrated)"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error_message", "Неизвестная ошибка"),
                    "answer": None
                }
        
        # Fallback к простому решателю
        elif hasattr(mathematical_solver, 'solve_problem'):
            solution = mathematical_solver.solve_problem(message)
            
            if solution:
                return {
                    "success": True,
                    "answer": solution.final_answer,
                    "confidence": solution.confidence,
                    "problem_type": solution.problem_type.value if hasattr(solution.problem_type, 'value') else str(solution.problem_type),
                    "explanation": solution.explanation,
                    "provider": "Mathematical Solver (Simple)"
                }
            else:
                return {
                    "success": False,
                    "error": "Не удалось решить задачу",
                    "answer": None
                }
        
        else:
            return {
                "success": False,
                "error": "Неизвестный тип математического решателя",
                "answer": None
            }
            
    except Exception as e:
        logger.error(f"Ошибка решения математической задачи: {e}")
        return {
            "success": False,
            "error": f"Ошибка обработки: {str(e)}",
            "answer": None
        }

def initialize_system():
    """Инициализация системы Rubin AI v2.0"""
    global provider_selector, documents_storage
    
    logger.info("🚀 Инициализация Rubin AI v2.0 (исправленная версия)...")
    
    # Инициализируем математический решатель первым
    math_success = initialize_mathematical_solver()
    if math_success:
        logger.info("✅ Математический решатель готов к работе")
    else:
        logger.warning("⚠️ Математический решатель недоступен")
    
    # Попытка инициализации провайдеров (без критических ошибок)
    try:
        from providers.smart_provider_selector import SmartProviderSelector
        provider_selector = SmartProviderSelector()
        logger.info("✅ Провайдер селектор инициализирован")
    except Exception as e:
        logger.warning(f"⚠️ Провайдер селектор недоступен: {e}")
        provider_selector = None
    
    # Безопасная инициализация провайдеров
    if provider_selector:
        try:
            from providers.huggingface_provider import HuggingFaceProvider
            hf_provider = HuggingFaceProvider()
            if hf_provider.initialize():
                provider_selector.register_provider(hf_provider)
                logger.info("✅ Hugging Face провайдер инициализирован")
            else:
                logger.warning("⚠️ Hugging Face провайдер недоступен")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка инициализации Hugging Face: {e}")
        
        try:
            from providers.google_cloud_provider import GoogleCloudProvider
            gc_provider = GoogleCloudProvider()
            if gc_provider.initialize():
                provider_selector.register_provider(gc_provider)
                logger.info("✅ Google Cloud провайдер инициализирован")
            else:
                logger.warning("⚠️ Google Cloud провайдер недоступен")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка инициализации Google Cloud: {e}")
    
    # Загружаем документы
    load_documents()
    
    logger.info("🎉 Rubin AI v2.0 успешно инициализирован!")

def load_documents():
    """Загрузка документов из хранилища"""
    global documents_storage
    
    try:
        if os.path.exists(Config.DOCUMENTS_STORAGE):
            with open(Config.DOCUMENTS_STORAGE, 'rb') as f:
                documents_storage = pickle.load(f)
            logger.info(f"Загружено {len(documents_storage)} документов")
        else:
            documents_storage = []
            logger.info("Документы не найдены, создаем новое хранилище")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки документов: {e}")
        documents_storage = []

# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0-fixed",
        "mathematical_solver": mathematical_solver is not None,
        "provider_selector": provider_selector is not None,
        "documents_count": len(documents_storage)
    })

@app.route('/api/status', methods=['GET'])
def system_status():
    """Статус системы"""
    return jsonify({
        "system": "Rubin AI v2.0 (Fixed)",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "mathematical_solver": "operational" if mathematical_solver else "offline",
            "provider_selector": "operational" if provider_selector else "offline",
            "documents": "operational" if documents_storage else "empty"
        }
    })

@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """Основной endpoint для AI чата с математической поддержкой"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Отсутствует поле 'message'"
            }), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({
                "success": False,
                "error": "Пустое сообщение"
            }), 400
        
        logger.info(f"🔍 Анализирую вопрос: \"{message}\"")
        
        # ПРОВЕРЯЕМ МАТЕМАТИЧЕСКИЙ ЗАПРОС
        if is_mathematical_request(message):
            logger.info("🧮 Обнаружен математический запрос")
            
            # Решаем математическую задачу
            math_result = solve_mathematical_problem(message)
            
            if math_result["success"]:
                logger.info(f"✅ Математическая задача решена: {math_result['answer']}")
                
                return jsonify({
                    "success": True,
                    "response": math_result["answer"],
                    "provider": math_result["provider"],
                    "category": "mathematics",
                    "confidence": math_result["confidence"],
                    "explanation": math_result.get("explanation", ""),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                logger.warning(f"⚠️ Не удалось решить математическую задачу: {math_result['error']}")
                
                return jsonify({
                    "success": False,
                    "error": f"Ошибка решения математической задачи: {math_result['error']}",
                    "category": "mathematics",
                    "timestamp": datetime.now().isoformat()
                })
        
        # НЕ МАТЕМАТИЧЕСКИЙ ЗАПРОС - обрабатываем как обычно
        logger.info("📡 Направляю к модулю: AI Чат (общий)")
        
        # Здесь можно добавить логику для других типов запросов
        # Пока возвращаем базовый ответ
        return jsonify({
            "success": True,
            "response": f"Я понимаю ваш вопрос: \"{message}\"\n\nДля получения точного ответа мне нужен доступ к соответствующей технической документации.\n\n**Что я могу предложить:**\n• Поиск в базе знаний Rubin AI\n• Анализ технической документации\n• Помощь с программированием и автоматизацией\n\nПопробуйте переформулировать вопрос или уточнить область применения для более точного ответа.",
            "provider": "Rubin AI Chat (General)",
            "category": "general",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса: {e}")
        return jsonify({
            "success": False,
            "error": f"Внутренняя ошибка сервера: {str(e)}"
        }), 500

@app.route('/api/mathematics/solve', methods=['POST'])
def mathematics_solve():
    """Специальный endpoint для математических задач"""
    try:
        data = request.get_json()
        if not data or 'problem' not in data:
            return jsonify({
                "success": False,
                "error": "Отсутствует поле 'problem'"
            }), 400
        
        problem = data['problem'].strip()
        if not problem:
            return jsonify({
                "success": False,
                "error": "Пустая задача"
            }), 400
        
        logger.info(f"🧮 Решение математической задачи: \"{problem}\"")
        
        result = solve_mathematical_problem(problem)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "data": {
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "problem_type": result.get("problem_type", "unknown"),
                    "explanation": result.get("explanation", ""),
                    "provider": result["provider"]
                },
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"],
                "timestamp": datetime.now().isoformat()
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Ошибка решения математической задачи: {e}")
        return jsonify({
            "success": False,
            "error": f"Внутренняя ошибка: {str(e)}"
        }), 500

@app.route('/api/mathematics/status', methods=['GET'])
def mathematics_status():
    """Статус математического решателя"""
    if mathematical_solver:
        try:
            if hasattr(mathematical_solver, 'get_solver_status'):
                status = mathematical_solver.get_solver_status()
                return jsonify(status)
            else:
                return jsonify({
                    "status": "operational",
                    "solver_type": "Mathematical Solver",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    else:
        return jsonify({
            "status": "offline",
            "error": "Математический решатель не инициализирован",
            "timestamp": datetime.now().isoformat()
        })

@app.route('/api/stats', methods=['GET'])
def system_stats():
    """Статистика системы"""
    return jsonify({
        "system": "Rubin AI v2.0 (Fixed)",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "mathematical_solver": "operational" if mathematical_solver else "offline",
            "provider_selector": "operational" if provider_selector else "offline",
            "documents": len(documents_storage)
        },
        "endpoints": [
            "GET /health",
            "GET /api/status", 
            "POST /api/ai/chat",
            "POST /api/mathematics/solve",
            "GET /api/mathematics/status",
            "GET /api/stats"
        ]
    })

@app.route('/api/documents/stats', methods=['GET'])
def documents_stats():
    """Статистика документов"""
    return jsonify({
        "total_documents": len(documents_storage),
        "storage_file": Config.DOCUMENTS_STORAGE,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("🚀 Запуск Rubin AI v2.0 (исправленная версия)")
    logger.info("📊 Доступные endpoints:")
    logger.info("   GET  /health - Проверка здоровья")
    logger.info("   GET  /api/status - Статус системы")
    logger.info("   POST /api/ai/chat - AI чат с математической поддержкой")
    logger.info("   POST /api/mathematics/solve - Решение математических задач")
    logger.info("   GET  /api/mathematics/status - Статус математического решателя")
    logger.info("   GET  /api/stats - Статистика системы")
    
    # Инициализация системы
    initialize_system()
    
    # Запуск сервера
    app.run(host='0.0.0.0', port=8083, debug=True)













