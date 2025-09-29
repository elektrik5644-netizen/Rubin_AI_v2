#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API для векторного поиска Rubin AI v2.0
Предоставляет эндпоинты для семантического поиска документов
"""

import os
import sys
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Добавляем корневую папку в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_search import VectorSearchEngine
from hybrid_search import HybridSearchEngine

app = Flask(__name__)
CORS(app)

# Инициализация движков поиска
vector_engine = None
hybrid_engine = None

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vector_search_api.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def initialize_engines():
    """Инициализация движков поиска"""
    global vector_engine, hybrid_engine
    
    try:
        logger.info("🔄 Инициализация движков поиска...")
        
        # Инициализация векторного движка
        vector_engine = VectorSearchEngine()
        
        # Инициализация гибридного движка
        hybrid_engine = HybridSearchEngine()
        
        logger.info("✅ Движки поиска инициализированы")
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации движков: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния API"""
    try:
        stats = {}
        
        if vector_engine:
            stats['vector_engine'] = vector_engine.get_stats()
        else:
            stats['vector_engine'] = {'status': 'not_initialized'}
            
        if hybrid_engine:
            stats['hybrid_engine'] = hybrid_engine.get_search_stats()
        else:
            stats['hybrid_engine'] = {'status': 'not_initialized'}
            
        return jsonify({
            'status': 'healthy',
            'service': 'Vector Search API',
            'version': '2.0',
            'engines': stats
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка health check: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/vector/search', methods=['POST'])
def vector_search():
    """Векторный поиск документов"""
    try:
        if not vector_engine:
            return jsonify({
                'error': 'Vector engine not initialized'
            }), 500
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query parameter is required'
            }), 400
            
        query = data['query']
        top_k = data.get('top_k', 5)
        threshold = data.get('threshold', 0.7)
        
        logger.info(f"🔍 Векторный поиск: '{query}' (top_k={top_k}, threshold={threshold})")
        
        # Выполнение поиска
        results = vector_engine.search_similar(query, top_k, threshold)
        
        # Форматирование результатов
        formatted_results = []
        for result in results:
            doc_id = result['document_id']
            content = vector_engine.get_document_content(doc_id)
            
            formatted_results.append({
                'document_id': doc_id,
                'similarity': result['similarity'],
                'content_preview': result['metadata'].get('preview', ''),
                'category': result['metadata'].get('category', ''),
                'full_content': content[:500] + "..." if content and len(content) > 500 else content
            })
            
        return jsonify({
            'query': query,
            'results': formatted_results,
            'total_found': len(formatted_results),
            'search_type': 'vector'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка векторного поиска: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/hybrid/search', methods=['POST'])
def hybrid_search():
    """Гибридный поиск документов"""
    try:
        if not hybrid_engine:
            return jsonify({
                'error': 'Hybrid engine not initialized'
            }), 500
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query parameter is required'
            }), 400
            
        query = data['query']
        limit = data.get('limit', 10)
        search_type = data.get('search_type', 'hybrid')
        
        logger.info(f"🔄 Гибридный поиск: '{query}' (limit={limit}, type={search_type})")
        
        # Выполнение поиска
        results = hybrid_engine.search(query, limit, search_type)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_found': len(results),
            'search_type': search_type
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка гибридного поиска: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/vector/index', methods=['POST'])
def index_document():
    """Индексация документа в векторном пространстве"""
    try:
        if not vector_engine:
            return jsonify({
                'error': 'Vector engine not initialized'
            }), 500
            
        data = request.get_json()
        if not data or 'document_id' not in data or 'content' not in data:
            return jsonify({
                'error': 'document_id and content are required'
            }), 400
            
        document_id = data['document_id']
        content = data['content']
        metadata = data.get('metadata', {})
        
        logger.info(f"📄 Индексация документа {document_id}")
        
        # Индексация документа
        success = vector_engine.index_document(document_id, content, metadata)
        
        if success:
            return jsonify({
                'message': f'Document {document_id} indexed successfully',
                'document_id': document_id
            }), 200
        else:
            return jsonify({
                'error': f'Failed to index document {document_id}'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка индексации документа: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/vector/build_index', methods=['POST'])
def build_faiss_index():
    """Построение FAISS индекса для быстрого поиска"""
    try:
        if not vector_engine:
            return jsonify({
                'error': 'Vector engine not initialized'
            }), 500
            
        logger.info("🔨 Построение FAISS индекса...")
        
        # Построение индекса
        success = vector_engine.build_faiss_index()
        
        if success:
            return jsonify({
                'message': 'FAISS index built successfully'
            }), 200
        else:
            return jsonify({
                'error': 'Failed to build FAISS index'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Ошибка построения FAISS индекса: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/vector/stats', methods=['GET'])
def get_vector_stats():
    """Получение статистики векторного поиска"""
    try:
        if not vector_engine:
            return jsonify({
                'error': 'Vector engine not initialized'
            }), 500
            
        stats = vector_engine.get_stats()
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/hybrid/stats', methods=['GET'])
def get_hybrid_stats():
    """Получение статистики гибридного поиска"""
    try:
        if not hybrid_engine:
            return jsonify({
                'error': 'Hybrid engine not initialized'
            }), 500
            
        stats = hybrid_engine.get_search_stats()
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/hybrid/weights', methods=['POST'])
def set_search_weights():
    """Установка весов для гибридного поиска"""
    try:
        if not hybrid_engine:
            return jsonify({
                'error': 'Hybrid engine not initialized'
            }), 500
            
        data = request.get_json()
        if not data or 'text_weight' not in data or 'vector_weight' not in data:
            return jsonify({
                'error': 'text_weight and vector_weight are required'
            }), 400
            
        text_weight = float(data['text_weight'])
        vector_weight = float(data['vector_weight'])
        
        # Установка весов
        hybrid_engine.set_search_weights(text_weight, vector_weight)
        
        return jsonify({
            'message': 'Search weights updated successfully',
            'text_weight': text_weight,
            'vector_weight': vector_weight
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка установки весов: {e}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Инициализация при запуске
    initialize_engines()
    
    # Запуск сервера
    port = 8091
    logger.info(f"🚀 Запуск Vector Search API на порту {port}")
    logger.info(f"📊 Health check: http://localhost:{port}/health")
    logger.info(f"🔍 Vector search: http://localhost:{port}/api/vector/search")
    logger.info(f"🔄 Hybrid search: http://localhost:{port}/api/hybrid/search")
    
    app.run(host='0.0.0.0', port=port, debug=True)






















