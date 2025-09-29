#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API для работы с документами Rubin AI v2.0
"""

import os
import sys
import sqlite3
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentsAPI:
    def __init__(self, db_path="rubin_ai_v2.db"):
        self.db_path = db_path
        
    def search_documents(self, query, category=None, limit=10):
        """Поиск документов"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if category:
                # Поиск в определенной категории
                cursor.execute('''
                    SELECT id, file_name, category, metadata, content
                    FROM documents
                    WHERE category = ? AND (content LIKE ? OR file_name LIKE ?)
                    ORDER BY file_name
                    LIMIT ?
                ''', (category, f'%{query}%', f'%{query}%', limit))
            else:
                # Поиск по всем документам
                cursor.execute('''
                    SELECT id, file_name, category, metadata, content
                    FROM documents
                    WHERE content LIKE ? OR file_name LIKE ?
                    ORDER BY file_name
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))
                
            results = cursor.fetchall()
            conn.close()
            
            # Форматирование результатов
            documents = []
            for row in results:
                doc_id, file_name, category, metadata, content = row
                metadata_dict = json.loads(metadata) if metadata else {}
                
                documents.append({
                    'id': doc_id,
                    'file_name': file_name,
                    'category': category,
                    'metadata': metadata_dict,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                })
                
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка поиска документов: {e}")
            return []
            
    def get_document_by_id(self, doc_id):
        """Получение документа по ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, file_name, category, metadata, content, file_path
                FROM documents
                WHERE id = ?
            ''', (doc_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                doc_id, file_name, category, metadata, content, file_path = result
                metadata_dict = json.loads(metadata) if metadata else {}
                
                return {
                    'id': doc_id,
                    'file_name': file_name,
                    'category': category,
                    'metadata': metadata_dict,
                    'content': content,
                    'file_path': file_path
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Ошибка получения документа: {e}")
            return None
            
    def get_categories(self):
        """Получение списка категорий"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT category, COUNT(*) as count
                FROM documents
                GROUP BY category
                ORDER BY count DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            return [{'name': category, 'count': count} for category, count in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения категорий: {e}")
            return []
            
    def get_document_stats(self):
        """Получение статистики документов"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            # Статистика по категориям
            cursor.execute('''
                SELECT category, COUNT(*) as count
                FROM documents
                GROUP BY category
                ORDER BY count DESC
            ''')
            categories = cursor.fetchall()
            
            # Статистика по типам файлов
            cursor.execute('''
                SELECT file_type, COUNT(*) as count
                FROM documents
                GROUP BY file_type
                ORDER BY count DESC
            ''')
            file_types = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_documents': total_docs,
                'categories': [{'name': cat, 'count': count} for cat, count in categories],
                'file_types': [{'type': ftype, 'count': count} for ftype, count in file_types]
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return None

# Создание экземпляра API
documents_api = DocumentsAPI()

@app.route('/api/documents/search', methods=['GET', 'POST'])
def search_documents():
    """Поиск документов"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            query = data.get('query', '')
            category = data.get('category')
            limit = data.get('limit', 10)
        else:
            query = request.args.get('query', '')
            category = request.args.get('category')
            limit = int(request.args.get('limit', 10))
            
        if not query:
            return jsonify({'error': 'Параметр query обязателен'}), 400
            
        results = documents_api.search_documents(query, category, limit)
        
        return jsonify({
            'success': True,
            'query': query,
            'category': category,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Получение документа по ID"""
    try:
        document = documents_api.get_document_by_id(doc_id)
        
        if document:
            return jsonify({
                'success': True,
                'document': document
            })
        else:
            return jsonify({'error': 'Документ не найден'}), 404
            
    except Exception as e:
        logger.error(f"Ошибка получения документа: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/categories', methods=['GET'])
def get_categories():
    """Получение списка категорий"""
    try:
        categories = documents_api.get_categories()
        
        return jsonify({
            'success': True,
            'categories': categories
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения категорий: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/stats', methods=['GET'])
def get_stats():
    """Получение статистики документов"""
    try:
        stats = documents_api.get_document_stats()
        
        if stats:
            return jsonify({
                'success': True,
                'stats': stats
            })
        else:
            return jsonify({'error': 'Не удалось получить статистику'}), 500
            
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/status', methods=['GET'])
def get_status():
    """Проверка статуса API"""
    return jsonify({
        'status': 'online',
        'service': 'Documents API',
        'version': '2.0',
        'database': documents_api.db_path
    })

@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья API"""
    return jsonify({
        'status': 'healthy',
        'service': 'Documents API',
        'version': '2.0'
    })

if __name__ == '__main__':
    print("📚 Запуск API документов Rubin AI v2.0...")
    print("📊 API доступен по адресу: http://localhost:8088")
    print("📋 Документация: http://localhost:8088/health")
    print("\nНажмите Ctrl+C для остановки")
    
    try:
        app.run(host='0.0.0.0', port=8088, debug=False)
    except KeyboardInterrupt:
        print("\nAPI документов остановлен")























