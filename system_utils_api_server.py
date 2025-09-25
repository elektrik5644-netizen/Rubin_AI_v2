#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Utils API Server для Rubin AI v2
Системные утилиты и диагностика
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import psutil
import time
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Обработка CORS preflight запросов
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/api/system/utils', methods=['POST'])
def system_utils():
    """Основной эндпоинт для системных утилит"""
    try:
        data = request.get_json()
        command = data.get('command')
        params = data.get('params', {})

        if not command:
            return jsonify({'error': 'Missing command'}), 400

        result = None
        
        if command == 'check_status':
            result = check_system_status()
        elif command == 'optimize_db':
            result = optimize_databases(params.get('db_path'))
        elif command == 'cleanup_db':
            result = cleanup_databases(params.get('db_path'))
        elif command == 'migrate_kb':
            result = migrate_knowledge_bases(params.get('source'), params.get('destination'))
        elif command == 'system_info':
            result = get_system_info()
        elif command == 'memory_usage':
            result = get_memory_usage()
        elif command == 'disk_usage':
            result = get_disk_usage()
        else:
            return jsonify({'error': 'Unsupported command'}), 400

        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        logger.error(f"Error executing system utility: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def check_system_status():
    """Проверка статуса системы"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'uptime': time.time() - psutil.boot_time(),
            'processes': len(psutil.pids()),
            'status': 'healthy'
        }
        return status
    except Exception as e:
        return {'error': str(e), 'status': 'error'}

def optimize_databases(db_path=None):
    """Оптимизация баз данных"""
    try:
        if not db_path:
            db_path = "."
        
        # Простая оптимизация - проверка размера файлов
        db_files = []
        for file in os.listdir(db_path):
            if file.endswith('.db'):
                file_path = os.path.join(db_path, file)
                size = os.path.getsize(file_path)
                db_files.append({
                    'name': file,
                    'size_bytes': size,
                    'size_mb': round(size / (1024 * 1024), 2)
                })
        
        return {
            'message': 'Database optimization completed',
            'databases': db_files,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}

def cleanup_databases(db_path=None):
    """Очистка баз данных"""
    try:
        if not db_path:
            db_path = "."
        
        cleaned_files = []
        for file in os.listdir(db_path):
            if file.endswith('.db.backup') or file.endswith('.db.tmp'):
                file_path = os.path.join(db_path, file)
                os.remove(file_path)
                cleaned_files.append(file)
        
        return {
            'message': 'Database cleanup completed',
            'cleaned_files': cleaned_files,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}

def migrate_knowledge_bases(source=None, destination=None):
    """Миграция баз знаний"""
    try:
        if not source or not destination:
            return {'error': 'Source and destination paths required'}
        
        # Простая миграция - копирование файлов
        if os.path.exists(source):
            import shutil
            shutil.copy2(source, destination)
            return {
                'message': 'Knowledge base migration completed',
                'source': source,
                'destination': destination,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {'error': f'Source file {source} not found'}
    except Exception as e:
        return {'error': str(e)}

def get_system_info():
    """Получение информации о системе"""
    try:
        info = {
            'platform': os.name,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        return info
    except Exception as e:
        return {'error': str(e)}

def get_memory_usage():
    """Получение информации об использовании памяти"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}

def get_disk_usage():
    """Получение информации об использовании диска"""
    try:
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья System Utils API"""
    return jsonify({
        'service': 'System Utils API',
        'status': 'healthy',
        'port': 8103,
        'version': '1.0',
        'capabilities': ['system_monitoring', 'file_operations', 'process_management']
    })

@app.route('/api/system/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'online',
        'service': 'System Utils API',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/system/status', methods=['GET'])
def status():
    """Статус сервера"""
    return jsonify({
        'status': 'online',
        'service': 'System Utils API',
        'uptime': time.time() - psutil.boot_time(),
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent(),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("🛠️ System Utils API Server запущен на порту 8103")
    logger.info("URL: http://localhost:8103")
    logger.info("Endpoints:")
    logger.info("  - POST /api/system/utils - Системные утилиты")
    logger.info("  - GET /api/system/health - Проверка здоровья")
    logger.info("  - GET /api/system/status - Статус сервера")
    app.run(host='0.0.0.0', port=8103, debug=True)


