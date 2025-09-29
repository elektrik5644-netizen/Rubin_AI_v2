#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLC Analysis Server для анализа PLC программ
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'healthy',
        'service': 'PLC Analysis Server',
        'port': 8099,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/plc/analyze', methods=['GET', 'POST'])
def analyze_plc():
    """Анализ PLC программы"""
    try:
        if request.method == 'GET':
            plc_code = request.args.get('plc_code', '')
        else:
            data = request.get_json()
            plc_code = data.get('plc_code', '')
        
        logger.info(f"🔧 Получен запрос анализа PLC: {plc_code[:50]}...")
        
        # Простая логика анализа
        analysis = {
            'status': 'success',
            'analysis': {
                'code_length': len(plc_code),
                'lines_count': plc_code.count('\n') + 1 if plc_code else 0,
                'issues': [],
                'recommendations': []
            },
            'service': 'plc_analysis',
            'timestamp': datetime.now().isoformat()
        }
        
        if plc_code:
            # Простой анализ кода
            if 'LD' in plc_code.upper():
                analysis['analysis']['recommendations'].append("Используется Ladder Logic")
            
            if 'ST' in plc_code.upper():
                analysis['analysis']['recommendations'].append("Используется Structured Text")
            
            if 'FBD' in plc_code.upper():
                analysis['analysis']['recommendations'].append("Используется Function Block Diagram")
            
            # Проверка на потенциальные проблемы
            if plc_code.count('(') != plc_code.count(')'):
                analysis['analysis']['issues'].append("Несбалансированные скобки")
            
            if 'GOTO' in plc_code.upper():
                analysis['analysis']['issues'].append("Использование GOTO - не рекомендуется")
        else:
            analysis['analysis']['recommendations'].append("Отправьте PLC код для анализа")
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа PLC: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🔧 PLC Analysis Server запущен")
    print("URL: http://localhost:8099")
    print("Доступные эндпоинты:")
    print("  - GET/POST /api/plc/analyze - анализ PLC программ")
    print("  - GET /api/health - проверка здоровья")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8099, debug=False)