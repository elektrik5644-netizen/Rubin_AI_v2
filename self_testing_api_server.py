#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для системы самотестирования Rubin AI
Предоставляет endpoints для самотестирования и описания возможностей
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
from datetime import datetime
import json

from rubin_self_testing import RubinSelfTesting

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация системы самотестирования
self_testing_system = RubinSelfTesting()

@app.route('/api/self_testing/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера самотестирования"""
    return jsonify({
        "status": "Rubin AI Self-Testing Server ONLINE",
        "timestamp": datetime.now().isoformat(),
        "total_servers": len(self_testing_system.servers),
        "total_capabilities": sum(len(caps) for caps in self_testing_system.capabilities.values())
    }), 200

@app.route('/api/self_testing/full_test', methods=['POST'])
def run_full_test():
    """Запуск полного самотестирования"""
    try:
        logger.info("🚀 Запуск полного самотестирования Rubin AI...")
        
        # Запускаем полное тестирование
        report = self_testing_system.run_full_self_test()
        
        response_data = {
            "success": True,
            "report": report,
            "message": "Полное самотестирование завершено успешно",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка полного самотестирования: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка самотестирования: {str(e)}"
        }), 500

@app.route('/api/self_testing/server_status', methods=['GET'])
def check_server_status():
    """Проверка статуса всех серверов"""
    try:
        logger.info("🔍 Проверка статуса серверов...")
        
        server_status = self_testing_system._check_all_servers()
        
        response_data = {
            "success": True,
            "server_status": server_status,
            "total_servers": len(server_status),
            "online_servers": len([s for s in server_status.values() if s['status'] == 'online']),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки статуса серверов: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка проверки статуса: {str(e)}"
        }), 500

@app.route('/api/self_testing/functionality_test', methods=['POST'])
def test_functionality():
    """Тестирование функциональности серверов"""
    try:
        logger.info("🧪 Тестирование функциональности серверов...")
        
        functionality_tests = self_testing_system._test_functionality()
        
        response_data = {
            "success": True,
            "functionality_tests": functionality_tests,
            "total_tests": len(functionality_tests),
            "working_tests": len([t for t in functionality_tests.values() if t['status'] == 'working']),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования функциональности: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка тестирования функциональности: {str(e)}"
        }), 500

@app.route('/api/self_testing/capabilities', methods=['GET'])
def get_capabilities():
    """Получение списка возможностей Rubin AI"""
    try:
        logger.info("📊 Получение списка возможностей...")
        
        capabilities_report = self_testing_system._generate_capabilities_report()
        
        response_data = {
            "success": True,
            "capabilities_report": capabilities_report,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения возможностей: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения возможностей: {str(e)}"
        }), 500

@app.route('/api/self_testing/self_description', methods=['GET'])
def get_self_description():
    """Получение описания возможностей Rubin AI"""
    try:
        logger.info("🤖 Генерация описания возможностей Rubin AI...")
        
        description = self_testing_system.generate_self_description()
        
        response_data = {
            "success": True,
            "description": description,
            "message": "Описание возможностей Rubin AI сгенерировано",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации описания: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка генерации описания: {str(e)}"
        }), 500

@app.route('/api/self_testing/quick_test', methods=['POST'])
def quick_test():
    """Быстрое тестирование основных серверов"""
    try:
        logger.info("⚡ Быстрое тестирование основных серверов...")
        
        # Тестируем только основные серверы
        main_servers = ['smart_dispatcher', 'math_server', 'electrical_server', 'programming_server']
        
        quick_status = {}
        for server_id in main_servers:
            if server_id in self_testing_system.servers:
                server_info = self_testing_system.servers[server_id]
                try:
                    health_url = f"http://localhost:{server_info['port']}{server_info['endpoint']}"
                    response = requests.get(health_url, timeout=2)
                    
                    if response.status_code == 200:
                        quick_status[server_id] = {
                            "status": "online",
                            "name": server_info['name'],
                            "port": server_info['port']
                        }
                    else:
                        quick_status[server_id] = {
                            "status": "error",
                            "name": server_info['name'],
                            "port": server_info['port'],
                            "error": f"HTTP {response.status_code}"
                        }
                except:
                    quick_status[server_id] = {
                        "status": "offline",
                        "name": server_info['name'],
                        "port": server_info['port']
                    }
        
        response_data = {
            "success": True,
            "quick_status": quick_status,
            "online_count": len([s for s in quick_status.values() if s['status'] == 'online']),
            "total_count": len(quick_status),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка быстрого тестирования: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка быстрого тестирования: {str(e)}"
        }), 500

@app.route('/api/self_testing/chat_integration', methods=['POST'])
def chat_integration():
    """Интеграция с чатом RubinDeveloper"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        
        logger.info(f"💬 Интеграция самотестирования с чатом: {message}")
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["самотестирование", "самотест", "что умеешь", "возможности", "функции"]):
            # Генерируем описание возможностей
            description = self_testing_system.generate_self_description()
            
            response_data = {
                "success": True,
                "type": "self_description",
                "description": description,
                "message": "Описание возможностей Rubin AI",
                "timestamp": datetime.now().isoformat()
            }
            
        elif any(word in message_lower for word in ["статус", "проверь", "диагностика", "состояние"]):
            # Проверяем статус серверов
            server_status = self_testing_system._check_all_servers()
            online_count = len([s for s in server_status.values() if s['status'] == 'online'])
            total_count = len(server_status)
            
            status_message = f"🔍 **СТАТУС СИСТЕМЫ RUBIN AI:**\n\n"
            status_message += f"📊 **Общая статистика:**\n"
            status_message += f"• Всего модулей: {total_count}\n"
            status_message += f"• Онлайн модулей: {online_count}\n"
            status_message += f"• Доступность: {(online_count/total_count*100):.1f}%\n\n"
            
            status_message += f"📋 **Детальный статус:**\n"
            for server_id, status in server_status.items():
                server_name = self_testing_system.servers[server_id]['name']
                if status['status'] == 'online':
                    status_message += f"✅ {server_name} (порт {status['port']}) - ОНЛАЙН\n"
                elif status['status'] == 'offline':
                    status_message += f"❌ {server_name} (порт {status['port']}) - НЕДОСТУПЕН\n"
                else:
                    status_message += f"⚠️ {server_name} (порт {status['port']}) - ОШИБКА\n"
            
            response_data = {
                "success": True,
                "type": "server_status",
                "status_message": status_message,
                "server_status": server_status,
                "timestamp": datetime.now().isoformat()
            }
            
        else:
            # Общий ответ
            response_data = {
                "success": True,
                "type": "general",
                "message": "🧪 Я могу провести самотестирование и рассказать о своих возможностях. Спросите: 'что умеешь?', 'статус системы', 'самотестирование'",
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка интеграции с чатом: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка интеграции с чатом: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = 8102  # Новый порт для сервера самотестирования
    logger.info(f"🧪 Запуск Rubin AI Self-Testing сервера на порту {port}...")
    app.run(port=port, debug=True, use_reloader=False)










