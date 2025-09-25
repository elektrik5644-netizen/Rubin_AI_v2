#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для управления модулями Rubin AI v2.0
"""

import os
import sys
import subprocess
import time
import json
import signal
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

class ModuleManager:
    def __init__(self):
        self.processes = {}
        self.modules = {
            'main': {
                'name': 'AI Чат (Основной сервер)',
                'port': 8084,
                'command': [sys.executable, 'api/rubin_ai_v2_server.py'],
                'status': 'stopped'
            },
            'electrical': {
                'name': 'Электротехника',
                'port': 8087,
                'command': [sys.executable, 'api/electrical_api.py'],
                'status': 'stopped'
            },
            'radiomechanics': {
                'name': 'Радиомеханика',
                'port': 8089,
                'command': [sys.executable, 'api/radiomechanics_api.py'],
                'status': 'stopped'
            },
            'controllers': {
                'name': 'Контроллеры',
                'port': 8090,
                'command': [sys.executable, 'api/controllers_api.py'],
                'status': 'stopped'
            }
        }
    
    def start_module(self, module_key):
        """Запуск модуля"""
        if module_key not in self.modules:
            return False, f"Модуль {module_key} не найден"
        
        module = self.modules[module_key]
        
        if module_key in self.processes:
            return False, f"Модуль {module['name']} уже запущен"
        
        try:
            # Устанавливаем переменную окружения для корректной кодировки
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = subprocess.Popen(
                module['command'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='ignore',
                env=env
            )
            
            self.processes[module_key] = process
            module['status'] = 'running'
            
            return True, f"Модуль {module['name']} запущен (PID: {process.pid})"
            
        except Exception as e:
            module['status'] = 'error'
            return False, f"Ошибка запуска {module['name']}: {str(e)}"
    
    def stop_module(self, module_key):
        """Остановка модуля"""
        if module_key not in self.modules:
            return False, f"Модуль {module_key} не найден"
        
        module = self.modules[module_key]
        
        if module_key not in self.processes:
            return False, f"Модуль {module['name']} не запущен"
        
        try:
            process = self.processes[module_key]
            process.terminate()
            
            # Ждем завершения процесса
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            del self.processes[module_key]
            module['status'] = 'stopped'
            
            return True, f"Модуль {module['name']} остановлен"
            
        except Exception as e:
            return False, f"Ошибка остановки {module['name']}: {str(e)}"
    
    def get_status(self):
        """Получение статуса всех модулей"""
        status = {}
        for module_key, module in self.modules.items():
            status[module_key] = {
                'name': module['name'],
                'port': module['port'],
                'status': module['status'],
                'running': module_key in self.processes
            }
        return status
    
    def start_all(self):
        """Запуск всех модулей"""
        results = {}
        for module_key in self.modules:
            success, message = self.start_module(module_key)
            results[module_key] = {'success': success, 'message': message}
            time.sleep(1)  # Небольшая задержка между запусками
        return results
    
    def stop_all(self):
        """Остановка всех модулей"""
        results = {}
        for module_key in list(self.processes.keys()):
            success, message = self.stop_module(module_key)
            results[module_key] = {'success': success, 'message': message}
        return results

# Глобальный экземпляр менеджера
manager = ModuleManager()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Получение статуса всех модулей"""
    return jsonify({
        'success': True,
        'modules': manager.get_status()
    })

@app.route('/api/start/<module_key>', methods=['POST'])
def start_module(module_key):
    """Запуск конкретного модуля"""
    success, message = manager.start_module(module_key)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/stop/<module_key>', methods=['POST'])
def stop_module(module_key):
    """Остановка конкретного модуля"""
    success, message = manager.stop_module(module_key)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/start-all', methods=['POST'])
def start_all():
    """Запуск всех модулей"""
    results = manager.start_all()
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/api/stop-all', methods=['POST'])
def stop_all():
    """Остановка всех модулей"""
    results = manager.stop_all()
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/api/restart-all', methods=['POST'])
def restart_all():
    """Перезапуск всех модулей"""
    # Сначала останавливаем все
    stop_results = manager.stop_all()
    time.sleep(3)
    
    # Затем запускаем все
    start_results = manager.start_all()
    
    return jsonify({
        'success': True,
        'stop_results': stop_results,
        'start_results': start_results
    })

@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья API"""
    return jsonify({
        'status': 'healthy',
        'service': 'Rubin AI Module Manager API',
        'version': '2.0'
    })

def cleanup_on_exit():
    """Очистка при выходе"""
    print("Остановка всех модулей...")
    manager.stop_all()
    print("Все модули остановлены")

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_on_exit)
    
    print("🚀 Запуск API менеджера модулей Rubin AI v2.0...")
    print("📊 API доступен по адресу: http://localhost:8086")
    print("🎛️  Веб-интерфейс: http://localhost:8086/RubinModuleManager.html")
    print("📋 Документация: http://localhost:8086/health")
    print("\nНажмите Ctrl+C для остановки")
    
    try:
        app.run(host='0.0.0.0', port=8086, debug=False)
    except KeyboardInterrupt:
        print("\nПолучен сигнал остановки...")
        cleanup_on_exit()


















