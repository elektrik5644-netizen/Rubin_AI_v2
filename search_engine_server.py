#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search Engine Server для гибридного поиска
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
        'service': 'Search Engine Server',
        'port': 8102,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/search/hybrid', methods=['GET', 'POST'])
def hybrid_search():
    """Гибридный поиск"""
    try:
        if request.method == 'GET':
            query = request.args.get('query', '')
        else:
            data = request.get_json()
            query = data.get('query', '')
        
        logger.info(f"🔍 Получен запрос поиска: {query[:50]}...")
        
        # Простая логика поиска
        result = {
            'status': 'success',
            'query': query,
            'search_results': {
                'total_results': 0,
                'results': [],
                'search_methods': []
            },
            'service': 'search_engine',
            'timestamp': datetime.now().isoformat()
        }
        
        if query:
            # Имитация результатов поиска
            result['search_results']['total_results'] = len(query.split()) * 3
            result['search_results']['search_methods'] = [
                'Полнотекстовый поиск',
                'Векторный поиск',
                'Семантический поиск'
            ]
            
            # Добавляем примеры результатов
            for i, word in enumerate(query.split()[:3]):
                result['search_results']['results'].append({
                    'title': f"Результат {i+1}: {word}",
                    'content': f"Содержимое документа о {word}",
                    'relevance': 0.9 - i * 0.1,
                    'source': f"Документ {i+1}"
                })
        else:
            result['search_results']['search_methods'] = [
                'Отправьте запрос для поиска',
                'Поддерживаемые методы: полнотекстовый, векторный, семантический'
            ]
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка поиска: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🔍 Search Engine Server запущен")
    print("URL: http://localhost:8102")
    print("Доступные эндпоинты:")
    print("  - GET/POST /api/search/hybrid - гибридный поиск")
    print("  - GET /api/health - проверка здоровья")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8102, debug=False)