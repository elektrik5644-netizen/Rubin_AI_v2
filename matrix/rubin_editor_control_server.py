#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - API сервер для управления редактором
Предоставляет REST API для управления файлами, редактирования кода и создания документов
"""

import json
import time
import sqlite3
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import shutil
import datetime

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем модуль управления редактором
try:
    from rubin_editor_controller import RubinEditorController, EditorCommand, EditorAction, DocumentType
    EDITOR_CONTROLLER_AVAILABLE = True
except ImportError:
    print("⚠️ Модуль управления редактором недоступен")
    EDITOR_CONTROLLER_AVAILABLE = False

class RubinEditorControlHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Инициализируем контроллер редактора
        if EDITOR_CONTROLLER_AVAILABLE:
            self.editor_controller = RubinEditorController()
        else:
            self.editor_controller = None
        
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Обработка POST запросов"""
        if self.path == '/api/file/operation':
            self.handle_file_operation()
        elif self.path == '/api/code/save':
            self.handle_save_code()
        elif self.path == '/api/code/format':
            self.handle_format_code()
        elif self.path == '/api/code/fix':
            self.handle_fix_code()
        elif self.path == '/api/document/create':
            self.handle_create_document()
        elif self.path == '/api/code/generate':
            self.handle_generate_code()
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def handle_file_operation(self):
        """Обработка операций с файлами"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            action = data.get('action', '')
            file_path = data.get('file_path', '')
            content = data.get('content', '')
            
            if not action or not file_path:
                self.send_error_response('Действие и путь к файлу обязательны')
                return
            
            print(f"Получен запрос операции с файлом: {action} {file_path}")
            
            # Выполняем операцию
            start_time = time.time()
            
            if self.editor_controller:
                # Создаем команду
                editor_action = None
                if action == 'create':
                    editor_action = EditorAction.CREATE_FILE
                elif action == 'edit':
                    editor_action = EditorAction.EDIT_FILE
                elif action == 'delete':
                    editor_action = EditorAction.DELETE_FILE
                elif action == 'save':
                    editor_action = EditorAction.SAVE_FILE
                elif action == 'open':
                    editor_action = EditorAction.OPEN_FILE
                
                if editor_action:
                    command = EditorCommand(
                        id=f"file_op_{int(time.time())}",
                        action=editor_action,
                        target=file_path,
                        content=content
                    )
                    
                    success = self.editor_controller.execute_command(command)
                    processing_time = time.time() - start_time
                    
                    if action == 'open' and success:
                        # Для операции открытия возвращаем содержимое
                        file_content = self.editor_controller.open_file(file_path)
                        result = {
                            'success': success,
                            'file_path': file_path,
                            'content': file_content,
                            'processing_time': processing_time
                        }
                    else:
                        result = {
                            'success': success,
                            'file_path': file_path,
                            'processing_time': processing_time
                        }
                else:
                    result = {
                        'success': False,
                        'error': f'Неизвестное действие: {action}'
                    }
            else:
                result = {
                    'success': False,
                    'error': 'Контроллер редактора недоступен'
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка обработки операции с файлом: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def handle_save_code(self):
        """Обработка сохранения кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_path = data.get('file_path', '')
            code = data.get('code', '')
            
            if not file_path or not code:
                self.send_error_response('Путь к файлу и код обязательны')
                return
            
            print(f"Получен запрос сохранения кода: {file_path}")
            
            # Сохраняем код
            start_time = time.time()
            
            if self.editor_controller:
                command = EditorCommand(
                    id=f"save_code_{int(time.time())}",
                    action=EditorAction.SAVE_FILE,
                    target=file_path,
                    content=code
                )
                
                success = self.editor_controller.execute_command(command)
                processing_time = time.time() - start_time
                
                result = {
                    'success': success,
                    'file_path': file_path,
                    'processing_time': processing_time
                }
            else:
                result = {
                    'success': False,
                    'error': 'Контроллер редактора недоступен'
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка сохранения кода: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def handle_format_code(self):
        """Обработка форматирования кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_path = data.get('file_path', '')
            code = data.get('code', '')
            
            if not file_path:
                self.send_error_response('Путь к файлу обязателен')
                return
            
            print(f"Получен запрос форматирования кода: {file_path}")
            
            # Форматируем код
            start_time = time.time()
            
            if self.editor_controller:
                command = EditorCommand(
                    id=f"format_code_{int(time.time())}",
                    action=EditorAction.FORMAT_CODE,
                    target=file_path
                )
                
                success = self.editor_controller.execute_command(command)
                processing_time = time.time() - start_time
                
                # Получаем отформатированный код
                formatted_code = ""
                if success:
                    formatted_code = self.editor_controller.open_file(file_path) or code
                
                result = {
                    'success': success,
                    'file_path': file_path,
                    'formatted_code': formatted_code,
                    'processing_time': processing_time
                }
            else:
                result = {
                    'success': False,
                    'error': 'Контроллер редактора недоступен'
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка форматирования кода: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def handle_fix_code(self):
        """Обработка исправления кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_path = data.get('file_path', '')
            code = data.get('code', '')
            
            if not file_path:
                self.send_error_response('Путь к файлу обязателен')
                return
            
            print(f"Получен запрос исправления кода: {file_path}")
            
            # Исправляем код
            start_time = time.time()
            
            if self.editor_controller:
                command = EditorCommand(
                    id=f"fix_code_{int(time.time())}",
                    action=EditorAction.FIX_CODE,
                    target=file_path
                )
                
                success = self.editor_controller.execute_command(command)
                processing_time = time.time() - start_time
                
                # Получаем исправленный код
                fixed_code = ""
                if success:
                    fixed_code = self.editor_controller.open_file(file_path) or code
                
                result = {
                    'success': success,
                    'file_path': file_path,
                    'fixed_code': fixed_code,
                    'processing_time': processing_time
                }
            else:
                result = {
                    'success': False,
                    'error': 'Контроллер редактора недоступен'
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка исправления кода: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def handle_create_document(self):
        """Обработка создания документа"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_path = data.get('file_path', '')
            doc_type = data.get('doc_type', 'readme')
            parameters = data.get('parameters', {})
            
            if not file_path:
                self.send_error_response('Путь к файлу обязателен')
                return
            
            print(f"Получен запрос создания документа: {file_path} ({doc_type})")
            
            # Создаем документ
            start_time = time.time()
            
            if self.editor_controller:
                command = EditorCommand(
                    id=f"create_doc_{int(time.time())}",
                    action=EditorAction.CREATE_DOCUMENT,
                    target=file_path,
                    parameters={
                        'type': doc_type,
                        'parameters': parameters
                    }
                )
                
                success = self.editor_controller.execute_command(command)
                processing_time = time.time() - start_time
                
                result = {
                    'success': success,
                    'file_path': file_path,
                    'doc_type': doc_type,
                    'processing_time': processing_time
                }
            else:
                result = {
                    'success': False,
                    'error': 'Контроллер редактора недоступен'
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка создания документа: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def handle_generate_code(self):
        """Обработка генерации кода"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_path = data.get('file_path', '')
            language = data.get('language', 'python')
            code_type = data.get('code_type', 'function')
            parameters = data.get('parameters', {})
            
            if not file_path:
                self.send_error_response('Путь к файлу обязателен')
                return
            
            print(f"Получен запрос генерации кода: {file_path} ({language}.{code_type})")
            
            # Генерируем код
            start_time = time.time()
            
            if self.editor_controller:
                command = EditorCommand(
                    id=f"generate_code_{int(time.time())}",
                    action=EditorAction.GENERATE_CODE,
                    target=file_path,
                    parameters={
                        'language': language,
                        'type': code_type,
                        'parameters': parameters
                    }
                )
                
                success = self.editor_controller.execute_command(command)
                processing_time = time.time() - start_time
                
                result = {
                    'success': success,
                    'file_path': file_path,
                    'language': language,
                    'code_type': code_type,
                    'processing_time': processing_time
                }
            else:
                result = {
                    'success': False,
                    'error': 'Контроллер редактора недоступен'
                }
            
            # Отправляем ответ
            self.send_json_response(result)
            
        except Exception as e:
            print(f"Ошибка генерации кода: {e}")
            self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
    
    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == '/health':
            self.send_json_response({
                'status': 'ok',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'editor_controller_available': EDITOR_CONTROLLER_AVAILABLE,
                'pid': os.getpid()
            })
        elif self.path == '/api/workspace/status':
            self.send_workspace_status()
        elif self.path == '/api/files/list':
            self.send_files_list()
        elif self.path == '/api/stats':
            self.send_stats_response()
        elif self.path == '/api/history':
            self.send_history_response()
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def send_workspace_status(self):
        """Отправляет статус рабочего пространства"""
        try:
            if self.editor_controller:
                status = self.editor_controller.get_workspace_status()
                self.send_json_response(status)
            else:
                self.send_error_response('Контроллер редактора недоступен')
        except Exception as e:
            self.send_error_response(f'Ошибка получения статуса: {str(e)}')
    
    def send_files_list(self):
        """Отправляет список файлов"""
        try:
            if self.editor_controller:
                status = self.editor_controller.get_workspace_status()
                files = status.get('files', [])
                folders = status.get('folders', [])
                
                # Формируем список файлов с информацией
                file_list = []
                for file_path in files:
                    full_path = os.path.join(self.editor_controller.workspace_path, file_path)
                    if os.path.exists(full_path):
                        stat = os.stat(full_path)
                        file_list.append({
                            'name': file_path,
                            'type': 'file',
                            'size': stat.st_size,
                            'modified': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                
                for folder_path in folders:
                    full_path = os.path.join(self.editor_controller.workspace_path, folder_path)
                    if os.path.exists(full_path):
                        stat = os.stat(full_path)
                        file_list.append({
                            'name': folder_path,
                            'type': 'folder',
                            'size': '-',
                            'modified': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                
                self.send_json_response({
                    'files': file_list,
                    'total_files': len([f for f in file_list if f['type'] == 'file']),
                    'total_folders': len([f for f in file_list if f['type'] == 'folder'])
                })
            else:
                self.send_error_response('Контроллер редактора недоступен')
        except Exception as e:
            self.send_error_response(f'Ошибка получения списка файлов: {str(e)}')
    
    def send_stats_response(self):
        """Отправляет статистику"""
        try:
            conn = sqlite3.connect(self.editor_controller.db_path if self.editor_controller else 'rubin_editor.db')
            cursor = conn.cursor()
            
            # Статистика команд редактора
            cursor.execute('SELECT COUNT(*) FROM editor_commands')
            total_commands = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM editor_commands WHERE success = 1')
            successful_commands = cursor.fetchone()[0]
            
            # Статистика по типам действий
            cursor.execute('SELECT action, COUNT(*) FROM editor_commands GROUP BY action')
            action_stats = dict(cursor.fetchall())
            
            # Статистика документов
            cursor.execute('SELECT COUNT(*) FROM created_documents')
            total_documents = cursor.fetchone()[0]
            
            # Статистика операций с файлами
            cursor.execute('SELECT COUNT(*) FROM file_operations')
            total_file_operations = cursor.fetchone()[0]
            
            conn.close()
            
            self.send_json_response({
                'total_commands': total_commands,
                'successful_commands': successful_commands,
                'action_stats': action_stats,
                'total_documents': total_documents,
                'total_file_operations': total_file_operations,
                'editor_controller_available': EDITOR_CONTROLLER_AVAILABLE
            })
            
        except Exception as e:
            self.send_error_response(f'Ошибка получения статистики: {str(e)}')
    
    def send_history_response(self):
        """Отправляет историю операций"""
        try:
            conn = sqlite3.connect(self.editor_controller.db_path if self.editor_controller else 'rubin_editor.db')
            cursor = conn.cursor()
            
            # Последние команды редактора
            cursor.execute('''
                SELECT id, action, target, success, timestamp
                FROM editor_commands 
                ORDER BY timestamp DESC 
                LIMIT 20
            ''')
            
            commands = []
            for row in cursor.fetchall():
                commands.append({
                    'id': row[0],
                    'action': row[1],
                    'target': row[2],
                    'success': bool(row[3]),
                    'timestamp': row[4]
                })
            
            # Последние созданные документы
            cursor.execute('''
                SELECT id, document_type, title, file_path, created_at
                FROM created_documents 
                ORDER BY created_at DESC 
                LIMIT 10
            ''')
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'id': row[0],
                    'document_type': row[1],
                    'title': row[2],
                    'file_path': row[3],
                    'created_at': row[4]
                })
            
            # Последние операции с файлами
            cursor.execute('''
                SELECT id, operation_type, file_path, timestamp
                FROM file_operations 
                ORDER BY timestamp DESC 
                LIMIT 15
            ''')
            
            file_operations = []
            for row in cursor.fetchall():
                file_operations.append({
                    'id': row[0],
                    'operation_type': row[1],
                    'file_path': row[2],
                    'timestamp': row[3]
                })
            
            conn.close()
            
            self.send_json_response({
                'commands': commands,
                'documents': documents,
                'file_operations': file_operations,
                'commands_count': len(commands),
                'documents_count': len(documents),
                'file_operations_count': len(file_operations)
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

def run_server(port=8086):
    """Запуск сервера"""
    print("🎮 Запуск Smart Rubin AI Editor Control Server...")
    print(f"📡 Порт: {port}")
    print(f"🎮 Контроллер редактора: {'✅ Доступен' if EDITOR_CONTROLLER_AVAILABLE else '❌ Недоступен'}")
    print("=" * 60)
    
    try:
        server = HTTPServer(('localhost', port), RubinEditorControlHandler)
        print(f"🎉 Сервер запущен на http://localhost:{port}")
        print("📋 Доступные endpoints:")
        print(f"   POST http://localhost:{port}/api/file/operation - операции с файлами")
        print(f"   POST http://localhost:{port}/api/code/save - сохранение кода")
        print(f"   POST http://localhost:{port}/api/code/format - форматирование кода")
        print(f"   POST http://localhost:{port}/api/code/fix - исправление кода")
        print(f"   POST http://localhost:{port}/api/document/create - создание документов")
        print(f"   POST http://localhost:{port}/api/code/generate - генерация кода")
        print(f"   GET  http://localhost:{port}/health - статус сервера")
        print(f"   GET  http://localhost:{port}/api/workspace/status - статус рабочего пространства")
        print(f"   GET  http://localhost:{port}/api/files/list - список файлов")
        print(f"   GET  http://localhost:{port}/api/stats - статистика")
        print(f"   GET  http://localhost:{port}/api/history - история операций")
        print(f"   GET  http://localhost:{port}/rubin_editor_control_ui.html - веб-интерфейс")
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
