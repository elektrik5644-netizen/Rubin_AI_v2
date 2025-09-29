#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Utils Server для системных утилит
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime
import psutil
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'healthy',
        'service': 'System Utils Server',
        'port': 8103,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/system/utils', methods=['GET', 'POST'])
def system_utils():
    """Системные утилиты"""
    try:
        if request.method == 'GET':
            action = request.args.get('action', 'status')
        else:
            data = request.get_json()
            action = data.get('action', 'status')
        
        logger.info(f"🔧 Получен запрос системных утилит: {action}")
        
        result = {
            'status': 'success',
            'action': action,
            'system_info': {},
            'service': 'system_utils',
            'timestamp': datetime.now().isoformat()
        }
        
        if action == 'status':
            # Получение системной информации
            result['system_info'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                'process_count': len(psutil.pids()),
                'uptime': 'N/A'  # Упрощенно
            }
        
        elif action == 'processes':
            # Список процессов
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            result['system_info'] = {
                'processes': processes[:10],  # Первые 10 процессов
                'total_processes': len(processes)
            }
        
        elif action == 'cleanup':
            # Очистка (имитация)
            result['system_info'] = {
                'cleanup_performed': True,
                'freed_space': 'N/A',
                'cleaned_files': 0
            }
        
        else:
            result['system_info'] = {
                'available_actions': ['status', 'processes', 'cleanup'],
                'description': 'Системные утилиты для мониторинга и обслуживания'
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка системных утилит: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🔧 System Utils Server запущен")
    print("URL: http://localhost:8103")
    print("Доступные эндпоинты:")
    print("  - GET/POST /api/system/utils - системные утилиты")
    print("  - GET /api/health - проверка здоровья")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8103, debug=False)