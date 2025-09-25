#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - API сервер для диагностики кода
Предоставляет REST API для анализа кода в реальном времени
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

# Импортируем модуль диагностики
try:
    from rubin_code_editor_diagnostic import RubinCodeEditorDiagnostic
    DIAGNOSTIC_AVAILABLE = True
except ImportError:
    print("⚠️ Модуль диагностики кода недоступен")
    DIAGNOSTIC_AVAILABLE = False

class RubinCodeDiagnosticHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Инициализируем систему диагностики
        if DIAGNOSTIC_AVAILABLE:
            self.diagnostic = RubinCodeEditorDiagnostic()
        else:
            self.diagnostic = None
        
        super().__init__(*args, **kwargs)
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect('rubin_code_diagnostics.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnostic_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code_hash TEXT NOT NULL,
                    language TEXT,
                    filename TEXT,
                    diagnostics_count INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_time REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnostic_results (
                    id TEXT PRIMARY KEY,
                    request_id INTEGER,
                    diagnostic_type TEXT,
                    severity TEXT,
                    message TEXT,
                    line INTEGER,
                    column INTEGER,
                    length INTEGER,
                    code_snippet TEXT,
                    suggestion TEXT,
                    quick_fix TEXT,
                    FOREIGN KEY (request_id) REFERENCES diagnostic_requests (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка инициализации БД: {e}")
            return False
    
    def save_diagnostic_request(self, code_hash, language, filename, diagnostics_count, processing_time):
        """Сохраняет запрос диагностики"""
        try:
            conn = sqlite3.connect('rubin_code_diagnostics.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO diagnostic_requests 
                (code_hash, language, filename, diagnostics_count, processing_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (code_hash, language, filename, diagnostics_count, processing_time))
            
            request_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return request_id
        except Exception as e:
            print(f"Ошибка сохранения запроса: {e}")
            return None
    
    def save_diagnostic_results(self, request_id, diagnostics):
        """Сохраняет результаты диагностики"""
        try:
            conn = sqlite3.connect('rubin_code_diagnostics.db')
            cursor = conn.cursor()
            
            for diagnostic in diagnostics:
                cursor.execute('''
                    INSERT OR REPLACE INTO diagnostic_results 
                    (id, request_id, diagnostic_type, severity, message, line, column, 
                     length, code_snippet, suggestion, quick_fix)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    diagnostic['id'],
                    request_id,
                    diagnostic['type'],
                    diagnostic['severity'],
                    diagnostic['message'],
                    diagnostic['line'],
                    diagnostic['column'],
                    diagnostic['length'],
                    diagnostic['code_snippet'],
                    diagnostic['suggestion'],
                    diagnostic.get('quick_fix')
                ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка сохранения результатов: {e}")
            return False
    
    def do_POST(self):
        """Обработка POST запросов"""
        if self.path == '/api/diagnose':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                code = data.get('code', '')
                filename = data.get('filename', '')
                language = data.get('language', 'auto')
                
                if not code:
                    self.send_error_response('Код не может быть пустым')
                    return
                
                print(f"Получен запрос диагностики: {filename} ({language})")
                
                # Выполняем диагностику
                start_time = time.time()
                
                if self.diagnostic:
                    result = self.diagnostic.diagnose_code(code, filename)
                    processing_time = time.time() - start_time
                    
                    # Сохраняем запрос и результаты
                    request_id = self.save_diagnostic_request(
                        result['code_hash'], 
                        result['language'], 
                        filename, 
                        len(result['diagnostics']), 
                        processing_time
                    )
                    
                    if request_id:
                        self.save_diagnostic_results(request_id, result['diagnostics'])
                    
                    # Добавляем метаданные
                    result['request_id'] = request_id
                    result['processing_time'] = processing_time
                    result['server_version'] = '1.0.0'
                    
                else:
                    result = {
                        'error': 'Система диагностики недоступна',
                        'language': 'unknown',
                        'diagnostics': [],
                        'recommendations': [],
                        'summary': 'Ошибка инициализации системы диагностики'
                    }
                
                # Отправляем ответ
                self.send_json_response(result)
                
            except Exception as e:
                print(f"Ошибка обработки POST: {e}")
                self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == '/health':
            self.send_json_response({
                'status': 'ok',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'diagnostic_available': DIAGNOSTIC_AVAILABLE,
                'pid': os.getpid()
            })
        elif self.path == '/api/stats':
            self.send_stats_response()
        elif self.path == '/api/history':
            self.send_history_response()
        elif self.path.startswith('/api/diagnostic/'):
            # Получение конкретного результата диагностики
            diagnostic_id = self.path.split('/')[-1]
            self.send_diagnostic_response(diagnostic_id)
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def send_stats_response(self):
        """Отправляет статистику"""
        try:
            conn = sqlite3.connect('rubin_code_diagnostics.db')
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute('SELECT COUNT(*) FROM diagnostic_requests')
            total_requests = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM diagnostic_results')
            total_diagnostics = cursor.fetchone()[0]
            
            # Статистика по языкам
            cursor.execute('SELECT language, COUNT(*) FROM diagnostic_requests GROUP BY language')
            language_stats = dict(cursor.fetchall())
            
            # Статистика по серьезности
            cursor.execute('SELECT severity, COUNT(*) FROM diagnostic_results GROUP BY severity')
            severity_stats = dict(cursor.fetchall())
            
            # Среднее время обработки
            cursor.execute('SELECT AVG(processing_time) FROM diagnostic_requests')
            avg_processing_time = cursor.fetchone()[0] or 0
            
            conn.close()
            
            self.send_json_response({
                'total_requests': total_requests,
                'total_diagnostics': total_diagnostics,
                'language_stats': language_stats,
                'severity_stats': severity_stats,
                'avg_processing_time': round(avg_processing_time, 3),
                'diagnostic_available': DIAGNOSTIC_AVAILABLE
            })
            
        except Exception as e:
            self.send_error_response(f'Ошибка получения статистики: {str(e)}')
    
    def send_history_response(self):
        """Отправляет историю диагностики"""
        try:
            conn = sqlite3.connect('rubin_code_diagnostics.db')
            cursor = conn.cursor()
            
            # Получаем последние 10 запросов
            cursor.execute('''
                SELECT id, code_hash, language, filename, diagnostics_count, 
                       timestamp, processing_time
                FROM diagnostic_requests 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            requests = []
            for row in cursor.fetchall():
                requests.append({
                    'id': row[0],
                    'code_hash': row[1],
                    'language': row[2],
                    'filename': row[3],
                    'diagnostics_count': row[4],
                    'timestamp': row[5],
                    'processing_time': row[6]
                })
            
            conn.close()
            
            self.send_json_response({
                'requests': requests,
                'count': len(requests)
            })
            
        except Exception as e:
            self.send_error_response(f'Ошибка получения истории: {str(e)}')
    
    def send_diagnostic_response(self, diagnostic_id):
        """Отправляет конкретный результат диагностики"""
        try:
            conn = sqlite3.connect('rubin_code_diagnostics.db')
            cursor = conn.cursor()
            
            # Получаем запрос
            cursor.execute('''
                SELECT id, code_hash, language, filename, diagnostics_count, 
                       timestamp, processing_time
                FROM diagnostic_requests 
                WHERE id = ?
            ''', (diagnostic_id,))
            
            request_row = cursor.fetchone()
            if not request_row:
                self.send_error_response('Запрос не найден')
                return
            
            # Получаем результаты диагностики
            cursor.execute('''
                SELECT id, diagnostic_type, severity, message, line, column, 
                       length, code_snippet, suggestion, quick_fix
                FROM diagnostic_results 
                WHERE request_id = ?
            ''', (diagnostic_id,))
            
            diagnostics = []
            for row in cursor.fetchall():
                diagnostics.append({
                    'id': row[0],
                    'type': row[1],
                    'severity': row[2],
                    'message': row[3],
                    'line': row[4],
                    'column': row[5],
                    'length': row[6],
                    'code_snippet': row[7],
                    'suggestion': row[8],
                    'quick_fix': row[9]
                })
            
            conn.close()
            
            result = {
                'request_id': request_row[0],
                'code_hash': request_row[1],
                'language': request_row[2],
                'filename': request_row[3],
                'diagnostics_count': request_row[4],
                'timestamp': request_row[5],
                'processing_time': request_row[6],
                'diagnostics': diagnostics
            }
            
            self.send_json_response(result)
            
        except Exception as e:
            self.send_error_response(f'Ошибка получения диагностики: {str(e)}')
    
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

def run_server(port=8084):
    """Запуск сервера"""
    print("🔍 Запуск Smart Rubin AI Code Diagnostic Server...")
    print(f"📡 Порт: {port}")
    print(f"🔍 Диагностика кода: {'✅ Доступна' if DIAGNOSTIC_AVAILABLE else '❌ Недоступна'}")
    print("=" * 60)
    
    # Инициализируем базу данных
    handler = RubinCodeDiagnosticHandler
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
        print(f"   POST http://localhost:{port}/api/diagnose - диагностика кода")
        print(f"   GET  http://localhost:{port}/health - статус сервера")
        print(f"   GET  http://localhost:{port}/api/stats - статистика")
        print(f"   GET  http://localhost:{port}/api/history - история запросов")
        print(f"   GET  http://localhost:{port}/api/diagnostic/{{id}} - конкретный результат")
        print(f"   GET  http://localhost:{port}/rubin_code_diagnostic_ui.html - веб-интерфейс")
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
