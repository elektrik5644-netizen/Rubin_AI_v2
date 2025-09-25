#!/usr/bin/env python3
"""
Улучшенный сервер Rubin AI с поддержкой базы данных
"""

import json
import time
import sqlite3
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
from datetime import datetime

class EnhancedRubinAIHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.db_path = "rubin_ai.db"
        self.init_database()
        super().__init__(*args, **kwargs)
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица сообщений
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'anonymous',
                    session_id TEXT
                )
            ''')
            
            # Таблица документов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content TEXT,
                    category TEXT,
                    tags TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    content_type TEXT
                )
            ''')
            
            # Таблица статистики
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"✅ База данных инициализирована: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации базы данных: {e}")
    
    def do_GET(self):
        """Обработка GET запросов"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.handle_health()
        elif parsed_path.path == '/api/documents/stats':
            self.handle_documents_stats()
        elif parsed_path.path == '/api/system/stats':
            self.handle_system_stats()
        elif parsed_path.path == '/':
            self.handle_root()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_POST(self):
        """Обработка POST запросов"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/chat':
            self.handle_chat()
        elif parsed_path.path == '/api/code/analyze':
            self.handle_code_analyze()
        elif parsed_path.path == '/api/documents/upload-content':
            self.handle_documents_upload()
        elif parsed_path.path == '/api/documents/search':
            self.handle_documents_search()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_OPTIONS(self):
        """Обработка CORS preflight запросов"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()
    
    def handle_health(self):
        """Обработка health check"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Получаем статистику из базы данных
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Подсчет сообщений
            cursor.execute("SELECT COUNT(*) FROM messages")
            message_count = cursor.fetchone()[0]
            
            # Подсчет документов
            cursor.execute("SELECT COUNT(*) FROM documents")
            document_count = cursor.fetchone()[0]
            
            conn.close()
            
            response = {
                "status": "healthy",
                "message": "Enhanced Rubin AI is running!",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "version": "2.1.0",
                "database": {
                    "status": "connected",
                    "messages": message_count,
                    "documents": document_count
                }
            }
        except Exception as e:
            response = {
                "status": "degraded",
                "message": f"Server running but database error: {e}",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "version": "2.1.0"
            }
        
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
    
    def handle_root(self):
        """Обработка корневого пути"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "message": "Enhanced Rubin AI v2.1",
            "status": "running",
            "endpoints": [
                "/health",
                "/api/chat",
                "/api/code/analyze",
                "/api/documents/upload-content",
                "/api/documents/search",
                "/api/documents/stats",
                "/api/system/stats"
            ],
            "features": [
                "Chat with AI",
                "Code analysis",
                "Document upload",
                "Database storage",
                "Statistics tracking"
            ]
        }
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
    
    def handle_chat(self):
        """Обработка чата"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            user_id = data.get('user_id', 'anonymous')
            session_id = data.get('session_id', f'session_{int(time.time())}')
            
            if not message.strip():
                self.send_error(400, "Empty message")
                return
            
            # Генерируем ответ
            response_text = self.generate_chat_response(message)
            
            # Сохраняем в базу данных
            self.save_message_to_db(message, response_text, user_id, session_id)
            
            # Отправляем ответ
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "response": response_text,
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "message_id": self.get_last_message_id()
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"Chat error: {e}")
    
    def handle_code_analyze(self):
        """Обработка анализа кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            code = data.get('code', '')
            language = data.get('language', 'unknown')
            
            if not code.strip():
                self.send_error(400, "Empty code")
                return
            
            # Анализируем код
            analysis_result = self.analyze_code(code, language)
            
            # Отправляем ответ
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(analysis_result, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"Code analysis error: {e}")
    
    def handle_documents_upload(self):
        """Обработка загрузки документов"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            filename = data.get('filename', 'unknown.txt')
            content = data.get('content', '')
            category = data.get('category', 'general')
            tags = data.get('tags', [])
            
            if not content.strip():
                self.send_error(400, "Empty content")
                return
            
            # Сохраняем документ в базу данных
            document_id = self.save_document_to_db(filename, content, category, tags)
            
            # Отправляем ответ
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "success",
                "message": "Документ загружен успешно",
                "document_id": document_id,
                "filename": filename,
                "size": len(content),
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"Document upload error: {e}")
    
    def handle_documents_search(self):
        """Обработка поиска документов"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query', '')
            limit = data.get('limit', 10)
            
            if not query.strip():
                self.send_error(400, "Empty search query")
                return
            
            # Поиск в базе данных
            results = self.search_documents(query, limit)
            
            # Отправляем ответ
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"Document search error: {e}")
    
    def handle_documents_stats(self):
        """Обработка статистики документов"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_documents = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(file_size) FROM documents WHERE file_size IS NOT NULL")
            total_size = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT category, COUNT(*) FROM documents GROUP BY category")
            categories = dict(cursor.fetchall())
            
            cursor.execute("SELECT DATE(timestamp) as date, COUNT(*) FROM documents GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 7")
            recent_uploads = dict(cursor.fetchall())
            
            conn.close()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "success",
                "statistics": {
                    "total_documents": total_documents,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "categories": categories,
                    "recent_uploads": recent_uploads
                },
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"Statistics error: {e}")
    
    def handle_system_stats(self):
        """Обработка системной статистики"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Статистика сообщений
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM messages WHERE timestamp >= datetime('now', '-1 hour')")
            messages_last_hour = cursor.fetchone()[0]
            
            cursor.execute("SELECT DATE(timestamp) as date, COUNT(*) FROM messages GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 7")
            messages_by_date = dict(cursor.fetchall())
            
            # Статистика документов
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_documents = cursor.fetchone()[0]
            
            conn.close()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "success",
                "system_stats": {
                    "messages": {
                        "total": total_messages,
                        "last_hour": messages_last_hour,
                        "by_date": messages_by_date
                    },
                    "documents": {
                        "total": total_documents
                    },
                    "uptime": time.time() - self.server_start_time if hasattr(self, 'server_start_time') else 0
                },
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            self.send_error(500, f"System stats error: {e}")
    
    def generate_chat_response(self, message):
        """Генерация ответа чата"""
        message_lower = message.lower()
        
        # Базовые ответы
        if "привет" in message_lower or "hello" in message_lower:
            return "Привет! Я Enhanced Rubin AI с поддержкой базы данных. Готов помочь с программированием и промышленной автоматизацией!"
        elif "как дела" in message_lower:
            return "Отлично! Система работает стабильно, база данных подключена. Чем могу помочь?"
        elif "python" in message_lower:
            return "Python - отличный язык программирования! Могу помочь с анализом кода, созданием скриптов или решением задач."
        elif "plc" in message_lower or "плц" in message_lower:
            return "PLC программирование - моя специализация! Помогу с Ladder Logic, Structured Text, диагностикой PMAC."
        elif "pmac" in message_lower:
            return "PMAC контроллеры - это моя область! Могу помочь с настройкой, программированием и диагностикой."
        elif "база данных" in message_lower or "database" in message_lower:
            return "База данных работает! Все сообщения и документы сохраняются. Могу показать статистику."
        elif "статистика" in message_lower or "stats" in message_lower:
            return "Запросите статистику через /api/documents/stats или /api/system/stats для получения подробной информации."
        elif "анализ" in message_lower or "анализ кода" in message_lower:
            return "Анализ кода - одна из моих основных функций! Загрузите код через /api/code/analyze для детального анализа."
        elif "помощь" in message_lower or "help" in message_lower:
            return """Доступные функции:
• Анализ кода (Python, PLC, PMAC)
• Генерация кода
• Диагностика промышленного оборудования
• Программирование PLC
• Работа с PMAC контроллерами
• Математические вычисления
• Загрузка и поиск документов
• Статистика системы"""
        else:
            return f"Понял ваш запрос: '{message}'. Я Enhanced Rubin AI с поддержкой базы данных. Специализируюсь на промышленной автоматизации, программировании PLC, PMAC и анализе кода. Чем конкретно могу помочь?"
    
    def analyze_code(self, code, language):
        """Анализ кода"""
        issues = []
        recommendations = []
        quality_score = 85.0
        
        if language.lower() == "python":
            if "import *" in code:
                issues.append({
                    "type": "warning",
                    "message": "Использование 'import *' не рекомендуется",
                    "severity": "medium",
                    "line": code.find("import *") + 1
                })
                recommendations.append("Используйте конкретные импорты")
                quality_score -= 5
            
            if "eval(" in code:
                issues.append({
                    "type": "security",
                    "message": "Использование eval() может быть небезопасно",
                    "severity": "high",
                    "line": code.find("eval(") + 1
                })
                recommendations.append("Избегайте использования eval()")
                quality_score -= 10
            
            if "print(" in code and "def " in code:
                recommendations.append("Рассмотрите использование логирования вместо print() в функциях")
            
            if len(code.split('\n')) < 3:
                recommendations.append("Код довольно короткий, возможно стоит добавить больше функциональности")
        
        elif language.lower() == "c":
            if "printf(" in code and "stdio.h" not in code:
                issues.append({
                    "type": "error",
                    "message": "Использование printf() без подключения stdio.h",
                    "severity": "high",
                    "line": code.find("printf(") + 1
                })
                recommendations.append("Добавьте #include <stdio.h>")
                quality_score -= 15
        
        elif language.lower() == "sql":
            if "SELECT *" in code.upper():
                issues.append({
                    "type": "warning",
                    "message": "Использование SELECT * может быть неэффективным",
                    "severity": "medium",
                    "line": 1
                })
                recommendations.append("Указывайте конкретные колонки вместо *")
                quality_score -= 5
        
        return {
            "issues": issues,
            "recommendations": recommendations,
            "quality_score": max(0, quality_score),
            "language": language,
            "lines_of_code": len(code.split('\n')),
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    
    def save_message_to_db(self, message, response, user_id, session_id):
        """Сохранение сообщения в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO messages (message, response, user_id, session_id)
                VALUES (?, ?, ?, ?)
            ''', (message, response, user_id, session_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка сохранения сообщения: {e}")
            return False
    
    def save_document_to_db(self, filename, content, category, tags):
        """Сохранение документа в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            tags_str = ','.join(tags) if isinstance(tags, list) else str(tags)
            
            cursor.execute('''
                INSERT INTO documents (filename, content, category, tags, file_size, content_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, content, category, tags_str, len(content), 'text/plain'))
            
            document_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return document_id
        except Exception as e:
            print(f"Ошибка сохранения документа: {e}")
            return None
    
    def search_documents(self, query, limit):
        """Поиск документов в базе данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Простой поиск по содержимому и названию
            cursor.execute('''
                SELECT id, filename, content, category, tags, timestamp
                FROM documents
                WHERE content LIKE ? OR filename LIKE ? OR tags LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "filename": row[1],
                    "content_preview": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                    "category": row[3],
                    "tags": row[4].split(',') if row[4] else [],
                    "timestamp": row[5]
                })
            
            conn.close()
            return results
        except Exception as e:
            print(f"Ошибка поиска документов: {e}")
            return []
    
    def get_last_message_id(self):
        """Получение ID последнего сообщения"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM messages ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            print(f"Ошибка получения ID сообщения: {e}")
            return None

def run_enhanced_server(port=8083):
    """Запуск улучшенного сервера"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, EnhancedRubinAIHandler)
    
    # Устанавливаем время запуска для статистики
    httpd.server_start_time = time.time()
    
    print(f"🚀 Enhanced Rubin AI Server запущен на порту {port}")
    print(f"📊 База данных: rubin_ai.db")
    print(f"🌐 Доступен по адресу: http://localhost:{port}")
    print(f"📋 Endpoints:")
    print(f"   • GET  /health - проверка состояния")
    print(f"   • POST /api/chat - чат с AI")
    print(f"   • POST /api/code/analyze - анализ кода")
    print(f"   • POST /api/documents/upload-content - загрузка документов")
    print(f"   • POST /api/documents/search - поиск документов")
    print(f"   • GET  /api/documents/stats - статистика документов")
    print(f"   • GET  /api/system/stats - системная статистика")
    print("="*60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️ Сервер остановлен пользователем")
        httpd.server_close()

if __name__ == "__main__":
    run_enhanced_server()
