#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing API Server
Обработка и анализ данных
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime
import statistics
import re

# Попытка импорта библиотек для обработки данных
try:
    import pandas as pd
    import numpy as np
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def preprocess_text(text):
    """Предобработка текста"""
    if not text:
        return {"error": "Текст не может быть пустым"}
    
    # Очистка текста
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
    words = cleaned_text.split()
    
    # Статистика
    stats = {
        "original_length": len(text),
        "cleaned_length": len(cleaned_text),
        "word_count": len(words),
        "unique_words": len(set(words)),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    
    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "words": words,
        "unique_words": list(set(words)),
        "statistics": stats
    }

def analyze_time_series(data):
    """Анализ временных рядов"""
    if not isinstance(data, list):
        return {"error": "Данные должны быть списком"}
    
    if len(data) < 2:
        return {"error": "Недостаточно данных для анализа"}
    
    try:
        # Простой анализ временного ряда
        values = [float(x) for x in data if isinstance(x, (int, float, str)) and str(x).replace('.', '').isdigit()]
        
        if len(values) < 2:
            return {"error": "Недостаточно числовых данных"}
        
        analysis = {
            "data_points": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
        }
        
        # Простой тренд
        if len(values) > 1:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            analysis["trend_strength"] = abs(statistics.mean(second_half) - statistics.mean(first_half))
        
        return analysis
        
    except Exception as e:
        return {"error": f"Ошибка анализа временного ряда: {str(e)}"}

def calculate_correlation(data1, data2):
    """Вычисление корреляции между двумя наборами данных"""
    if not isinstance(data1, list) or not isinstance(data2, list):
        return {"error": "Данные должны быть списками"}
    
    if len(data1) != len(data2):
        return {"error": "Наборы данных должны иметь одинаковую длину"}
    
    try:
        # Преобразуем в числа
        values1 = [float(x) for x in data1 if isinstance(x, (int, float, str)) and str(x).replace('.', '').isdigit()]
        values2 = [float(x) for x in data2 if isinstance(x, (int, float, str)) and str(x).replace('.', '').isdigit()]
        
        if len(values1) != len(values2) or len(values1) < 2:
            return {"error": "Недостаточно данных для корреляции"}
        
        # Простая корреляция Пирсона
        n = len(values1)
        sum1 = sum(values1)
        sum2 = sum(values2)
        sum1_sq = sum(x**2 for x in values1)
        sum2_sq = sum(x**2 for x in values2)
        sum12 = sum(values1[i] * values2[i] for i in range(n))
        
        numerator = n * sum12 - sum1 * sum2
        denominator = ((n * sum1_sq - sum1**2) * (n * sum2_sq - sum2**2))**0.5
        
        if denominator == 0:
            correlation = 0
        else:
            correlation = numerator / denominator
        
        return {
            "correlation": correlation,
            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak",
            "direction": "positive" if correlation > 0 else "negative" if correlation < 0 else "none",
            "data_points": n
        }
        
    except Exception as e:
        return {"error": f"Ошибка вычисления корреляции: {str(e)}"}

def filter_data(data, condition):
    """Фильтрация данных по условию"""
    if not isinstance(data, list):
        return {"error": "Данные должны быть списком"}
    
    try:
        filtered = []
        for item in data:
            # Простые условия фильтрации
            if condition == "numeric":
                if isinstance(item, (int, float)) or (isinstance(item, str) and item.replace('.', '').isdigit()):
                    filtered.append(item)
            elif condition == "text":
                if isinstance(item, str) and not item.replace('.', '').isdigit():
                    filtered.append(item)
            elif condition == "positive":
                if isinstance(item, (int, float)) and item > 0:
                    filtered.append(item)
            elif condition == "negative":
                if isinstance(item, (int, float)) and item < 0:
                    filtered.append(item)
            else:
                filtered.append(item)
        
        return {
            "original_count": len(data),
            "filtered_count": len(filtered),
            "filtered_data": filtered,
            "condition": condition
        }
        
    except Exception as e:
        return {"error": f"Ошибка фильтрации: {str(e)}"}

def normalize_data(data):
    """Нормализация данных"""
    if not isinstance(data, list):
        return {"error": "Данные должны быть списком"}
    
    try:
        # Преобразуем в числа
        values = [float(x) for x in data if isinstance(x, (int, float, str)) and str(x).replace('.', '').isdigit()]
        
        if len(values) < 2:
            return {"error": "Недостаточно числовых данных для нормализации"}
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 1
        
        normalized = [(x - mean_val) / std_val for x in values]
        
        return {
            "original_data": values,
            "normalized_data": normalized,
            "mean": mean_val,
            "std_dev": std_val,
            "method": "z-score"
        }
        
    except Exception as e:
        return {"error": f"Ошибка нормализации: {str(e)}"}

@app.route('/api/data/process', methods=['POST'])
def process_data():
    """Обработка данных"""
    try:
        data = request.get_json()
        input_data = data.get('data', '')
        operation = data.get('operation', 'analyze')
        
        if not input_data:
            return jsonify({
                "success": False,
                "error": "Данные не могут быть пустыми"
            }), 400
        
        result = {
            "operation": operation,
            "input_data": input_data,
            "processing_result": {}
        }
        
        # Определяем тип операции
        if operation == 'preprocess_text':
            result["processing_result"] = preprocess_text(input_data)
        
        elif operation == 'analyze_time_series':
            if isinstance(input_data, str):
                # Пытаемся парсить как JSON или разделить по запятым
                try:
                    parsed_data = json.loads(input_data)
                except:
                    parsed_data = [x.strip() for x in input_data.split(',')]
            else:
                parsed_data = input_data
            
            result["processing_result"] = analyze_time_series(parsed_data)
        
        elif operation == 'normalize':
            if isinstance(input_data, str):
                try:
                    parsed_data = json.loads(input_data)
                except:
                    parsed_data = [x.strip() for x in input_data.split(',')]
            else:
                parsed_data = input_data
            
            result["processing_result"] = normalize_data(parsed_data)
        
        elif operation == 'filter':
            condition = data.get('condition', 'numeric')
            if isinstance(input_data, str):
                try:
                    parsed_data = json.loads(input_data)
                except:
                    parsed_data = [x.strip() for x in input_data.split(',')]
            else:
                parsed_data = input_data
            
            result["processing_result"] = filter_data(parsed_data, condition)
        
        else:
            return jsonify({
                "success": False,
                "error": f"Неизвестная операция: {operation}"
            }), 400
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка обработки данных: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка обработки: {str(e)}"
        }), 500

@app.route('/api/data/correlation', methods=['POST'])
def calculate_correlation_endpoint():
    """Вычисление корреляции"""
    try:
        data = request.get_json()
        data1 = data.get('data1', [])
        data2 = data.get('data2', [])
        
        if not data1 or not data2:
            return jsonify({
                "success": False,
                "error": "Оба набора данных должны быть предоставлены"
            }), 400
        
        correlation_result = calculate_correlation(data1, data2)
        
        return jsonify({
            "success": True,
            "data1": data1,
            "data2": data2,
            "correlation": correlation_result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка вычисления корреляции: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка корреляции: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья Data Processing API"""
    return jsonify({
        'service': 'Data Processing API',
        'status': 'healthy',
        'port': 8101,
        'version': '1.0',
        'capabilities': ['data_analysis', 'visualization', 'statistics', 'ml_preprocessing']
    })

@app.route('/api/data/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "Data Processing API",
        "status": "online",
        "version": "1.0",
        "data_processing_available": DATA_PROCESSING_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/data/status', methods=['GET'])
def get_status():
    """Статус сервера"""
    return jsonify({
        "service": "Data Processing API",
        "status": "running",
        "port": 8101,
        "endpoints": [
            "/api/data/process",
            "/api/data/correlation",
            "/api/data/health",
            "/api/data/status"
        ],
        "capabilities": [
            "Предобработка текста",
            "Анализ временных рядов",
            "Нормализация данных",
            "Фильтрация данных",
            "Вычисление корреляции",
            "Статистический анализ"
        ],
        "dependencies": {
            "pandas": DATA_PROCESSING_AVAILABLE,
            "numpy": DATA_PROCESSING_AVAILABLE
        }
    })

if __name__ == '__main__':
    print("📊 Data Processing API Server запущен")
    print("URL: http://localhost:8101")
    print("Endpoints:")
    print("  - POST /api/data/process - Обработка данных")
    print("  - POST /api/data/correlation - Вычисление корреляции")
    print("  - GET /api/data/health - Проверка здоровья")
    print("  - GET /api/data/status - Статус сервера")
    app.run(host='0.0.0.0', port=8101, debug=False)


