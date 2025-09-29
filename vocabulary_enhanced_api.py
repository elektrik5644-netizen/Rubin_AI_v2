#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API для поиска с использованием технического словаря
"""

import sqlite3
import json
import re
from typing import List, Dict, Tuple, Set
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

class VocabularyEnhancedAPI:
    """API класс для поиска с использованием словаря"""
    
    def __init__(self, db_path: str = "rubin_ai_v2.db"):
        self.db_path = db_path
        self.connection = None
        self.connect_database()
    
    def connect_database(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise
    
    def get_synonyms(self, term: str) -> List[str]:
        """Получение синонимов для термина"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT DISTINCT synonym FROM technical_synonyms 
                WHERE main_term = ? OR synonym = ?
                UNION
                SELECT DISTINCT synonym FROM synonyms 
                WHERE term = ? OR synonym = ?
            """, (term, term, term, term))
            
            synonyms = [row[0] for row in cursor.fetchall()]
            return synonyms
            
        except Exception as e:
            print(f"❌ Ошибка получения синонимов: {e}")
            return []
    
    def expand_query_with_synonyms(self, query: str) -> Dict[str, List[str]]:
        """Расширение запроса синонимами"""
        try:
            words = re.findall(r'\b\w+\b', query.lower())
            expanded_terms = {}
            
            for word in words:
                if len(word) > 2:
                    synonyms = self.get_synonyms(word)
                    if synonyms:
                        expanded_terms[word] = synonyms
            
            return expanded_terms
            
        except Exception as e:
            print(f"❌ Ошибка расширения запроса: {e}")
            return {}
    
    def search_documents_with_synonyms(self, query: str, limit: int = 10, category: str = None) -> List[Dict]:
        """Поиск документов с использованием синонимов"""
        try:
            cursor = self.connection.cursor()
            
            expanded_terms = self.expand_query_with_synonyms(query)
            search_terms = [query]
            
            for term, synonyms in expanded_terms.items():
                search_terms.extend(synonyms)
            
            search_terms = list(set(search_terms))
            
            # Базовый SQL запрос
            base_sql = """
                SELECT DISTINCT 
                    id, file_name, content, category, tags, difficulty_level,
                    CASE 
                        WHEN content LIKE ? THEN 3
                        WHEN content LIKE ? THEN 2
                        ELSE 1
                    END as relevance_score
                FROM documents 
                WHERE {placeholders}
            """
            
            # Добавляем фильтр по категории если указан
            if category:
                base_sql += " AND category = ?"
            
            base_sql += """
                ORDER BY relevance_score DESC, id DESC
                LIMIT ?
            """
            
            placeholders = ' OR '.join(['content LIKE ?' for _ in search_terms])
            sql_query = base_sql.format(placeholders=placeholders)
            
            params = []
            for term in search_terms:
                params.append(f'%{term}%')
            
            params.append(f'%{query}%')
            params.append(f'%{query}%')
            
            if category:
                params.append(category)
            
            params.append(limit)
            
            cursor.execute(sql_query, params)
            results = cursor.fetchall()
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'id': row[0],
                    'file_name': row[1],
                    'content': row[2][:500] + '...' if len(row[2]) > 500 else row[2],
                    'category': row[3],
                    'tags': row[4],
                    'difficulty_level': row[5],
                    'relevance_score': row[6],
                    'matched_terms': self.find_matched_terms(row[2], search_terms),
                    'synonyms_used': list(expanded_terms.keys())
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Ошибка поиска документов: {e}")
            return []
    
    def find_matched_terms(self, content: str, search_terms: List[str]) -> List[str]:
        """Поиск совпавших терминов в контенте"""
        matched = []
        content_lower = content.lower()
        
        for term in search_terms:
            if term.lower() in content_lower:
                matched.append(term)
        
        return matched
    
    def get_category_suggestions(self, query: str) -> List[str]:
        """Получение предложений по категориям"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT DISTINCT category, COUNT(*) as count
                FROM technical_synonyms 
                WHERE main_term LIKE ? OR synonym LIKE ?
                GROUP BY category
                ORDER BY count DESC
                LIMIT 5
            """, (f'%{query}%', f'%{query}%'))
            
            categories = [row[0] for row in cursor.fetchall()]
            return categories
            
        except Exception as e:
            print(f"❌ Ошибка получения категорий: {e}")
            return []
    
    def get_related_terms(self, term: str) -> List[str]:
        """Получение связанных терминов"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT DISTINCT main_term
                FROM technical_synonyms 
                WHERE category = (
                    SELECT category FROM technical_synonyms 
                    WHERE main_term = ? OR synonym = ? 
                    LIMIT 1
                )
                AND main_term != ?
                LIMIT 10
            """, (term, term, term))
            
            related = [row[0] for row in cursor.fetchall()]
            return related
            
        except Exception as e:
            print(f"❌ Ошибка получения связанных терминов: {e}")
            return []
    
    def get_vocabulary_stats(self) -> Dict:
        """Получение статистики словаря"""
        try:
            cursor = self.connection.cursor()
            
            # Общая статистика
            cursor.execute("SELECT COUNT(*) FROM technical_synonyms")
            total_synonyms = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT main_term) FROM technical_synonyms")
            unique_terms = cursor.fetchone()[0]
            
            # Статистика по категориям
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM technical_synonyms
                GROUP BY category
                ORDER BY count DESC
            """)
            categories = cursor.fetchall()
            
            return {
                'total_synonyms': total_synonyms,
                'unique_terms': unique_terms,
                'categories': [{'name': cat[0], 'count': cat[1]} for cat in categories],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def close_connection(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()

class VocabularyAPIHandler(BaseHTTPRequestHandler):
    """HTTP обработчик для API"""
    
    def __init__(self, *args, **kwargs):
        self.vocabulary_api = VocabularyEnhancedAPI()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Обработка GET запросов"""
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            query_params = urllib.parse.parse_qs(parsed_path.query)
            
            if path == '/api/vocabulary/search':
                self.handle_search(query_params)
            elif path == '/api/vocabulary/synonyms':
                self.handle_synonyms(query_params)
            elif path == '/api/vocabulary/categories':
                self.handle_categories(query_params)
            elif path == '/api/vocabulary/related':
                self.handle_related(query_params)
            elif path == '/api/vocabulary/stats':
                self.handle_stats()
            else:
                self.send_error(404, "Endpoint not found")
                
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Обработка POST запросов"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            
            if path == '/api/vocabulary/search':
                self.handle_search_post(data)
            else:
                self.send_error(404, "Endpoint not found")
                
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_search(self, params):
        """Обработка поискового запроса (GET)"""
        query = params.get('q', [''])[0]
        limit = int(params.get('limit', ['10'])[0])
        category = params.get('category', [None])[0]
        
        if not query:
            self.send_error(400, "Query parameter 'q' is required")
            return
        
        results = self.vocabulary_api.search_documents_with_synonyms(query, limit, category)
        suggestions = self.vocabulary_api.get_category_suggestions(query)
        
        response = {
            'query': query,
            'results': results,
            'suggestions': suggestions,
            'total_found': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_json_response(response)
    
    def handle_search_post(self, data):
        """Обработка поискового запроса (POST)"""
        query = data.get('query', '')
        limit = data.get('limit', 10)
        category = data.get('category')
        
        if not query:
            self.send_error(400, "Field 'query' is required")
            return
        
        results = self.vocabulary_api.search_documents_with_synonyms(query, limit, category)
        suggestions = self.vocabulary_api.get_category_suggestions(query)
        expanded_terms = self.vocabulary_api.expand_query_with_synonyms(query)
        
        response = {
            'query': query,
            'results': results,
            'suggestions': suggestions,
            'expanded_terms': expanded_terms,
            'total_found': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_json_response(response)
    
    def handle_synonyms(self, params):
        """Обработка запроса синонимов"""
        term = params.get('term', [''])[0]
        
        if not term:
            self.send_error(400, "Parameter 'term' is required")
            return
        
        synonyms = self.vocabulary_api.get_synonyms(term)
        
        response = {
            'term': term,
            'synonyms': synonyms,
            'count': len(synonyms),
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_json_response(response)
    
    def handle_categories(self, params):
        """Обработка запроса категорий"""
        query = params.get('q', [''])[0]
        
        if query:
            categories = self.vocabulary_api.get_category_suggestions(query)
        else:
            # Возвращаем все категории
            stats = self.vocabulary_api.get_vocabulary_stats()
            categories = [cat['name'] for cat in stats.get('categories', [])]
        
        response = {
            'categories': categories,
            'count': len(categories),
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_json_response(response)
    
    def handle_related(self, params):
        """Обработка запроса связанных терминов"""
        term = params.get('term', [''])[0]
        
        if not term:
            self.send_error(400, "Parameter 'term' is required")
            return
        
        related = self.vocabulary_api.get_related_terms(term)
        
        response = {
            'term': term,
            'related_terms': related,
            'count': len(related),
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_json_response(response)
    
    def handle_stats(self):
        """Обработка запроса статистики"""
        stats = self.vocabulary_api.get_vocabulary_stats()
        self.send_json_response(stats)
    
    def send_json_response(self, data):
        """Отправка JSON ответа"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(json_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Отключение стандартного логирования"""
        pass

def run_server(port=8085):
    """Запуск сервера"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, VocabularyAPIHandler)
    
    print(f"🚀 Vocabulary Enhanced API Server запущен на порту {port}")
    print(f"📡 Доступные endpoints:")
    print(f"  GET  /api/vocabulary/search?q=query&limit=10&category=category")
    print(f"  POST /api/vocabulary/search")
    print(f"  GET  /api/vocabulary/synonyms?term=term")
    print(f"  GET  /api/vocabulary/categories?q=query")
    print(f"  GET  /api/vocabulary/related?term=term")
    print(f"  GET  /api/vocabulary/stats")
    print(f"🌐 Откройте: http://localhost:{port}/api/vocabulary/stats")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
        httpd.server_close()

if __name__ == "__main__":
    run_server()






















