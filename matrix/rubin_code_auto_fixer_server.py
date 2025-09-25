#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - API сервер для автоматического исправления кода
Предоставляет REST API для анализа, исправления и генерации кода
"""

import json
import time
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import os
import sys

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем модуль автоматического исправления
try:
    from rubin_code_auto_fixer import RubinCodeAutoFixer, CodeGenerationRequest, FixType, FixConfidence
    AUTO_FIXER_AVAILABLE = True
except ImportError:
    print("⚠️ Модуль автоматического исправления недоступен")
    AUTO_FIXER_AVAILABLE = False

class RubinCodeAutoFixerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Инициализируем систему автоматического исправления
        if AUTO_FIXER_AVAILABLE:
            self.auto_fixer = RubinCodeAutoFixer()
        else:
            self.auto_fixer = None
        
        super().__init__(*args, **kwargs)
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect('rubin_auto_fixer.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fix_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code_hash TEXT NOT NULL,
                    language TEXT,
                    filename TEXT,
                    fixes_count INTEGER,
                    applied_fixes INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_time REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generation_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    language TEXT,
                    context TEXT,
                    requirements TEXT,
                    generated_code TEXT,
                    confidence TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка инициализации БД: {e}")
            return False
    
    def save_fix_request(self, code_hash, language, filename, fixes_count, applied_fixes, processing_time):
        """Сохраняет запрос на исправление"""
        try:
            conn = sqlite3.connect('rubin_auto_fixer.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fix_requests 
                (code_hash, language, filename, fixes_count, applied_fixes, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (code_hash, language, filename, fixes_count, applied_fixes, processing_time))
            
            request_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return request_id
        except Exception as e:
            print(f"Ошибка сохранения запроса: {e}")
            return None
    
    def save_generation_request(self, description, language, context, requirements, generated_code, confidence):
        """Сохраняет запрос на генерацию кода"""
        try:
            conn = sqlite3.connect('rubin_auto_fixer.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO generation_requests 
                (description, language, context, requirements, generated_code, confidence, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (description, language, context, json.dumps(requirements), generated_code, confidence, True))
            
            request_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return request_id
        except Exception as e:
            print(f"Ошибка сохранения генерации: {e}")
            return None
    
    def do_POST(self):
        """Обработка POST запросов"""
        if self.path == '/api/analyze':
            self.handle_analyze_request()
        elif self.path == '/api/fix':
            self.handle_fix_request()
        elif self.path == '/api/generate':
            self.handle_generate_request()
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def handle_analyze_request(self):
        """Обработка запроса на анализ кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            code = data.get('code', '')
            filename = data.get('filename', '')
            language = data.get('language', 'python')
            
            if not code:
                self.send_error_response('Код не может быть пустым')
                return
            
            print(f"Получен запрос анализа: {filename} ({language})")
            
            # Выполняем анализ
            start_time = time.time()
            
            if self.auto_fixer:
                fixes = self.auto_fixer.analyze_and_fix_code(code, language, filename)
                processing_time = time.time() - start_time
                
                # Сохраняем запрос
                code_hash = str(hash(code))
                request_id = self.save_fix_request(
                    code_hash, language, filename, len(fixes), 0, processing_time
                )
                
                # Конвертируем исправления в JSON
                fixes_json = []
                for fix in fixes:
                    fixes_json.append({
                        'id': fix.id,
                        'type': fix.type.value,
                        'confidence': fix.confidence.value,
                        'description': fix.description,
                        'explanation': fix.explanation,
                        'line_start': fix.line_start,
                        'line_end': fix.line_end,
                        'column_start': fix.column_start,
                        'column_end': fix.column_end,
                        'original_code': fix.original_code,
                        'fixed_code': fix.fixed_code,
                        'before_snippet': fix.before_snippet,
                        'after_snippet': fix.after_snippet
                    })
                
                result = {
                    'language': language,
                    'filename': filename,
                    'fixes': fixes_json,
                    'fixes_count': len(fixes),
                    'request_id': request_id,
                    'processing_time': processing_time,
                    'code_hash': code_hash
                }
                
            else:
                result = {
                    'error': 'Система автоматического исправления недоступна',
                    'language': language,
                    'filename': filename,
                    'fixes': [],
                    'fixes_count': 0
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка обработки анализа: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def handle_fix_request(self):
        """Обработка запроса на исправление кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            code = data.get('code', '')
            fixes_data = data.get('fixes', [])
            language = data.get('language', 'python')
            filename = data.get('filename', '')
            
            if not code or not fixes_data:
                self.send_error_response('Код и исправления не могут быть пустыми')
                return
            
            print(f"Получен запрос исправления: {filename} ({len(fixes_data)} исправлений)")
            
            # Выполняем исправление
            start_time = time.time()
            
            if self.auto_fixer:
                # Конвертируем исправления обратно в объекты
                fixes = []
                for fix_data in fixes_data:
                    fix = type('CodeFix', (), {
                        'id': fix_data['id'],
                        'type': FixType(fix_data['type']),
                        'confidence': FixConfidence(fix_data['confidence']),
                        'original_code': fix_data['original_code'],
                        'fixed_code': fix_data['fixed_code'],
                        'line_start': fix_data['line_start'],
                        'line_end': fix_data['line_end'],
                        'column_start': fix_data['column_start'],
                        'column_end': fix_data['column_end'],
                        'description': fix_data['description'],
                        'explanation': fix_data['explanation'],
                        'before_snippet': fix_data['before_snippet'],
                        'after_snippet': fix_data['after_snippet']
                    })()
                    fixes.append(fix)
                
                fixed_code, applied_fixes = self.auto_fixer.apply_fixes(code, fixes)
                processing_time = time.time() - start_time
                
                # Сохраняем исправления
                for fix in applied_fixes:
                    self.auto_fixer.save_fix_to_database(fix)
                
                # Обновляем запрос
                code_hash = str(hash(code))
                self.save_fix_request(
                    code_hash, language, filename, len(fixes), len(applied_fixes), processing_time
                )
                
                result = {
                    'original_code': code,
                    'fixed_code': fixed_code,
                    'applied_fixes': len(applied_fixes),
                    'total_fixes': len(fixes),
                    'processing_time': processing_time,
                    'success': True
                }
                
            else:
                result = {
                    'error': 'Система автоматического исправления недоступна',
                    'success': False
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка обработки исправления: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def handle_generate_request(self):
        """Обработка запроса на генерацию кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            description = data.get('description', '')
            language = data.get('language', 'python')
            context = data.get('context', '')
            requirements = data.get('requirements', [])
            examples = data.get('examples', [])
            
            if not description:
                self.send_error_response('Описание не может быть пустым')
                return
            
            print(f"Получен запрос генерации: {description} ({language})")
            
            # Выполняем генерацию
            start_time = time.time()
            
            if self.auto_fixer:
                request = CodeGenerationRequest(
                    id=f"gen_{int(time.time())}",
                    description=description,
                    language=language,
                    context=context,
                    requirements=requirements,
                    examples=examples
                )
                
                generated_code = self.auto_fixer.generate_code(request)
                request.generated_code = generated_code
                
                processing_time = time.time() - start_time
                
                # Сохраняем генерацию
                request_id = self.save_generation_request(
                    description, language, context, requirements, generated_code, request.confidence.value
                )
                
                result = {
                    'description': description,
                    'language': language,
                    'context': context,
                    'requirements': requirements,
                    'generated_code': generated_code,
                    'confidence': request.confidence.value,
                    'request_id': request_id,
                    'processing_time': processing_time,
                    'success': True
                }
                
            else:
                result = {
                    'error': 'Система генерации кода недоступна',
                    'success': False
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка обработки генерации: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == '/health':
            self.send_json_response({
                'status': 'ok',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'auto_fixer_available': AUTO_FIXER_AVAILABLE,
                'pid': os.getpid()
            })
        elif self.path == '/api/stats':
            self.send_stats_response()
        elif self.path == '/api/history':
            self.send_history_response()
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def send_stats_response(self):
        """Отправляет статистику"""
        try:
            conn = sqlite3.connect('rubin_auto_fixer.db')
            cursor = conn.cursor()
            
            # Статистика исправлений
            cursor.execute('SELECT COUNT(*) FROM fix_requests')
            total_fix_requests = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(fixes_count) FROM fix_requests')
            total_fixes_found = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(applied_fixes) FROM fix_requests')
            total_fixes_applied = cursor.fetchone()[0] or 0
            
            # Статистика генерации
            cursor.execute('SELECT COUNT(*) FROM generation_requests')
            total_generation_requests = cursor.fetchone()[0]
            
            # Статистика по языкам
            cursor.execute('SELECT language, COUNT(*) FROM fix_requests GROUP BY language')
            language_stats = dict(cursor.fetchall())
            
            # Среднее время обработки
            cursor.execute('SELECT AVG(processing_time) FROM fix_requests')
            avg_processing_time = cursor.fetchone()[0] or 0
            
            conn.close()
            
            self.send_json_response({
                'total_fix_requests': total_fix_requests,
                'total_fixes_found': total_fixes_found,
                'total_fixes_applied': total_fixes_applied,
                'total_generation_requests': total_generation_requests,
                'language_stats': language_stats,
                'avg_processing_time': round(avg_processing_time, 3),
                'auto_fixer_available': AUTO_FIXER_AVAILABLE
            })
            
        except Exception as e:
            self.send_error_response(f'Ошибка получения статистики: {str(e)}')
    
    def send_history_response(self):
        """Отправляет историю запросов"""
        try:
            conn = sqlite3.connect('rubin_auto_fixer.db')
            cursor = conn.cursor()
            
            # Последние исправления
            cursor.execute('''
                SELECT id, code_hash, language, filename, fixes_count, applied_fixes, 
                       timestamp, processing_time
                FROM fix_requests 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            fix_requests = []
            for row in cursor.fetchall():
                fix_requests.append({
                    'id': row[0],
                    'code_hash': row[1],
                    'language': row[2],
                    'filename': row[3],
                    'fixes_count': row[4],
                    'applied_fixes': row[5],
                    'timestamp': row[6],
                    'processing_time': row[7]
                })
            
            # Последние генерации
            cursor.execute('''
                SELECT id, description, language, context, confidence, timestamp
                FROM generation_requests 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            generation_requests = []
            for row in cursor.fetchall():
                generation_requests.append({
                    'id': row[0],
                    'description': row[1],
                    'language': row[2],
                    'context': row[3],
                    'confidence': row[4],
                    'timestamp': row[5]
                })
            
            conn.close()
            
            self.send_json_response({
                'fix_requests': fix_requests,
                'generation_requests': generation_requests,
                'fix_requests_count': len(fix_requests),
                'generation_requests_count': len(generation_requests)
            })
            
        except Exception as e:
            self.send_error_response(f'Ошибка получения истории: {str(e)}')
    
    def send_json_response(self, data):
        """Отправляет JSON ответ"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def send_error_response(self, error_message):
        """Отправляет ответ с ошибкой"""
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = json.dumps({'error': error_message}, ensure_ascii=False)
        self.wfile.write(error_response.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Отключаем логирование в консоль"""
        pass

def run_server(port=8085):
    """Запуск сервера"""
    print("🔧 Запуск Smart Rubin AI Code Auto Fixer Server...")
    print(f"📡 Порт: {port}")
    print(f"🔧 Автоисправление: {'✅ Доступно' if AUTO_FIXER_AVAILABLE else '❌ Недоступно'}")
    print("=" * 60)
    
    # Инициализируем базу данных
    handler = RubinCodeAutoFixerHandler
    temp_handler = handler(None, None, None)
    if temp_handler.init_database():
        print("✅ База данных инициализирована")
    else:
        print("❌ Ошибка инициализации базы данных")
        return
    
    try:
        server = HTTPServer(('localhost', port), handler)
        print(f"🎉 Сервер запущен на http://localhost:{port}")
        print("📋 Доступные endpoints:")
        print(f"   POST http://localhost:{port}/api/analyze - анализ кода")
        print(f"   POST http://localhost:{port}/api/fix - исправление кода")
        print(f"   POST http://localhost:{port}/api/generate - генерация кода")
        print(f"   GET  http://localhost:{port}/health - статус сервера")
        print(f"   GET  http://localhost:{port}/api/stats - статистика")
        print(f"   GET  http://localhost:{port}/api/history - история запросов")
        print(f"   GET  http://localhost:{port}/rubin_code_auto_fixer_ui.html - веб-интерфейс")
        print("\n🛑 Для остановки нажмите Ctrl+C")
        print("=" * 60)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Остановка сервера...")
        server.shutdown()
        print("✅ Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка сервера: {e}")

if __name__ == "__main__":
    run_server()
