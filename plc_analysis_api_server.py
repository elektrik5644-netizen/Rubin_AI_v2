#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLC Analysis API Server
Анализ и диагностика PLC программ
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import logging
from datetime import datetime

# Импортируем функции анализа PLC
try:
    from plc_analyzer import analyze_plc_file
    PLC_ANALYZER_AVAILABLE = True
except ImportError:
    PLC_ANALYZER_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# База знаний по PLC
PLC_KNOWLEDGE = {
    "ladder_logic": {
        "description": "Релейная логика - графический язык программирования PLC",
        "elements": ["контакты", "катушки", "таймеры", "счетчики", "блоки функций"],
        "keywords": ["ladder", "релейная", "логика", "контакты", "катушки"]
    },
    "function_blocks": {
        "description": "Функциональные блоки - модульные элементы программы",
        "types": ["AND", "OR", "NOT", "TIMER", "COUNTER", "PID"],
        "keywords": ["функциональные блоки", "function blocks", "AND", "OR", "NOT"]
    },
    "data_types": {
        "description": "Типы данных в PLC программах",
        "types": ["BOOL", "INT", "REAL", "STRING", "ARRAY", "STRUCT"],
        "keywords": ["типы данных", "BOOL", "INT", "REAL", "STRING"]
    },
    "communication": {
        "description": "Коммуникационные протоколы PLC",
        "protocols": ["Modbus", "Profibus", "Ethernet/IP", "OPC UA"],
        "keywords": ["коммуникация", "протокол", "Modbus", "Profibus", "Ethernet"]
    }
}

def analyze_plc_content(content):
    """Анализ содержимого PLC программы"""
    analysis = {
        "syntax_errors": [],
        "warnings": [],
        "functions": {},
        "variables": {},
        "statistics": {}
    }
    
    lines = content.split('\n')
    
    # Поиск синтаксических ошибок
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
            
        # Проверка на незакрытые скобки
        if line.count('(') != line.count(')'):
            analysis["syntax_errors"].append(f"Строка {i}: Незакрытые скобки")
        
        # Проверка на неопределенные переменные
        if 'VAR_' in line and not any(keyword in line for keyword in ['VAR_INPUT', 'VAR_OUTPUT', 'VAR_IN_OUT']):
            analysis["warnings"].append(f"Строка {i}: Возможно неопределенная переменная")
    
    # Подсчет статистики
    analysis["statistics"] = {
        "total_lines": len([l for l in lines if l.strip() and not l.strip().startswith('//')]),
        "comment_lines": len([l for l in lines if l.strip().startswith('//')]),
        "function_blocks": len([l for l in lines if 'FUNCTION_BLOCK' in l]),
        "variables": len([l for l in lines if 'VAR_' in l])
    }
    
    return analysis

def find_plc_knowledge(query):
    """Поиск знаний по PLC"""
    query_lower = query.lower()
    results = []
    
    for topic, info in PLC_KNOWLEDGE.items():
        score = 0
        for keyword in info["keywords"]:
            if keyword.lower() in query_lower:
                score += 1
        
        if score > 0:
            results.append({
                "topic": topic,
                "description": info["description"],
                "score": score,
                "details": info
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

@app.route('/api/plc/analyze', methods=['POST'])
def analyze_plc():
    """Анализ PLC программы"""
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        action = data.get('action', 'analyze')
        
        if action == 'analyze':
            if file_path and os.path.exists(file_path):
                # Анализ файла
                if PLC_ANALYZER_AVAILABLE:
                    result = analyze_plc_file(file_path)
                else:
                    # Fallback анализ
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    result = analyze_plc_content(content)
                
                return jsonify({
                    "success": True,
                    "file": file_path,
                    "analysis": result,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Файл не найден или путь не указан"
                }), 400
        
        return jsonify({
            "success": False,
            "error": "Неизвестное действие"
        }), 400
        
    except Exception as e:
        logger.error(f"Ошибка анализа PLC: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка анализа: {str(e)}"
        }), 500

@app.route('/api/plc/knowledge', methods=['POST'])
def get_plc_knowledge():
    """Получение знаний по PLC"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({
                "success": False,
                "error": "Запрос не может быть пустым"
            }), 400
        
        knowledge = find_plc_knowledge(query)
        
        return jsonify({
            "success": True,
            "query": query,
            "knowledge": knowledge,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения знаний PLC: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения знаний: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья PLC Analysis API"""
    return jsonify({
        'service': 'PLC Analysis API',
        'status': 'healthy',
        'port': 8099,
        'version': '1.0',
        'capabilities': ['plc_analysis', 'code_review', 'syntax_check']
    })

@app.route('/api/plc/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "PLC Analysis API",
        "status": "online",
        "version": "1.0",
        "plc_analyzer_available": PLC_ANALYZER_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/plc/status', methods=['GET'])
def get_status():
    """Статус сервера"""
    return jsonify({
        "service": "PLC Analysis API",
        "status": "running",
        "port": 8099,
        "endpoints": [
            "/api/plc/analyze",
            "/api/plc/knowledge", 
            "/api/plc/health",
            "/api/plc/status"
        ],
        "capabilities": [
            "Анализ PLC программ",
            "Поиск синтаксических ошибок",
            "Статистика программы",
            "База знаний по PLC"
        ]
    })

if __name__ == '__main__':
    print("🔧 PLC Analysis API Server запущен")
    print("URL: http://localhost:8099")
    print("Endpoints:")
    print("  - POST /api/plc/analyze - Анализ PLC программы")
    print("  - POST /api/plc/knowledge - Получение знаний по PLC")
    print("  - GET /api/plc/health - Проверка здоровья")
    print("  - GET /api/plc/status - Статус сервера")
    app.run(host='0.0.0.0', port=8099, debug=True)


