#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing Server для обработки и анализа данных
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
        'service': 'Data Processing Server',
        'port': 8101,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/data/process', methods=['GET', 'POST'])
def process_data():
    """Обработка данных"""
    try:
        if request.method == 'GET':
            data = request.args.get('data', '')
        else:
            request_data = request.get_json()
            data = request_data.get('data', '')
        
        logger.info(f"📊 Получен запрос обработки данных: {data[:50]}...")
        
        # Простая логика обработки
        result = {
            'status': 'success',
            'input_data': data,
            'processing_results': {
                'data_type': 'text',
                'length': len(data),
                'words_count': len(data.split()) if data else 0,
                'processing_methods': []
            },
            'service': 'data_processing',
            'timestamp': datetime.now().isoformat()
        }
        
        if data:
            # Анализ данных
            if data.isdigit():
                result['processing_results']['data_type'] = 'numeric'
                result['processing_results']['processing_methods'].append('Числовой анализ')
            
            elif ',' in data:
                result['processing_results']['data_type'] = 'csv'
                result['processing_results']['processing_methods'].append('CSV обработка')
            
            elif 'json' in data.lower():
                result['processing_results']['data_type'] = 'json'
                result['processing_results']['processing_methods'].append('JSON парсинг')
            
            else:
                result['processing_results']['processing_methods'].append('Текстовый анализ')
            
            # Добавляем общие методы
            result['processing_results']['processing_methods'].extend([
                'Предобработка',
                'Фильтрация',
                'Нормализация',
                'Анализ'
            ])
        else:
            result['processing_results']['processing_methods'] = [
                'Отправьте данные для обработки',
                'Поддерживаемые форматы: текст, числа, CSV, JSON'
            ]
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки данных: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("📊 Data Processing Server запущен")
    print("URL: http://localhost:8101")
    print("Доступные эндпоинты:")
    print("  - GET/POST /api/data/process - обработка данных")
    print("  - GET /api/health - проверка здоровья")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8101, debug=False)