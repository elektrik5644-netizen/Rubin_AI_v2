#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный API сервер с подробными логическими ответами
Интегрирует улучшенный Smart Dispatcher с Центральной Базой Знаний
"""

from flask import Flask, request, jsonify
import logging
import json
from datetime import datetime
from enhanced_smart_dispatcher import get_enhanced_dispatcher
from intelligent_knowledge_chat import get_rubin_chat

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Инициализация компонентов
enhanced_dispatcher = get_enhanced_dispatcher()
knowledge_chat = get_rubin_chat()

@app.route('/api/enhanced/chat', methods=['POST'])
def enhanced_chat():
    """Улучшенный чат с подробными ответами"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'default')
        enable_knowledge = data.get('enable_knowledge', True)
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        logger.info(f"💬 Получено сообщение от {user_id}: {message[:50]}...")
        
        # Обрабатываем через улучшенный диспетчер
        dispatcher_result = enhanced_dispatcher.route_question(message)
        
        # Если включена интеграция с базой знаний
        knowledge_enhancement = ""
        if enable_knowledge:
            try:
                knowledge_result = knowledge_chat.process_message(message, user_id)
                if knowledge_result and len(knowledge_result) > 100:
                    knowledge_enhancement = f"\n\n**🧠 ДОПОЛНЕНИЯ ИЗ БАЗЫ ЗНАНИЙ:**\n{knowledge_result}"
            except Exception as e:
                logger.warning(f"⚠️ Ошибка интеграции с базой знаний: {e}")
        
        # Формируем финальный ответ
        final_response = dispatcher_result['response']
        if knowledge_enhancement:
            final_response += knowledge_enhancement
        
        # Добавляем метаинформацию
        meta_info = f"""
        
**📊 МЕТАИНФОРМАЦИЯ ОТВЕТА:**
• Длина ответа: {len(final_response)} символов
• Уровень детализации: {dispatcher_result['complexity']['detail_level']}/4
• Сложность вопроса: {dispatcher_result['complexity']['level']}
• Обработано модулем: {dispatcher_result['server_type']}
• Улучшен знаниями: {'Да' if knowledge_enhancement else 'Нет'}
• Время обработки: ~{dispatcher_result['complexity']['estimated_time']} секунд
• Статус: {'✅ Успешно обработано' if dispatcher_result['success'] else '❌ Ошибка обработки'}
"""
        
        final_response += meta_info
        
        return jsonify({
            'response': final_response,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'success': dispatcher_result['success'],
                'server_type': dispatcher_result['server_type'],
                'complexity': dispatcher_result['complexity'],
                'response_length': len(final_response),
                'enhanced': dispatcher_result['enhanced'],
                'knowledge_integration': bool(knowledge_enhancement)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в enhanced chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/analyze', methods=['POST'])
def analyze_question():
    """Анализирует вопрос без генерации ответа"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'Вопрос не может быть пустым'}), 400
        
        # Анализируем сложность
        complexity = enhanced_dispatcher.analyze_question_complexity(question)
        
        # Определяем тип сервера
        server_type = enhanced_dispatcher._determine_server_type(question)
        
        # Проверяем доступность сервера
        server_info = enhanced_dispatcher.servers.get(server_type, {})
        server_available = False
        
        if server_info:
            try:
                import requests
                health_url = f"http://localhost:{server_info['port']}{server_info['endpoint']}"
                response = requests.get(health_url, timeout=2)
                server_available = response.status_code == 200
            except:
                server_available = False
        
        return jsonify({
            'question': question,
            'analysis': {
                'complexity': complexity,
                'server_type': server_type,
                'server_available': server_available,
                'estimated_response_length': complexity['detail_level'] * 5000 + 2000,
                'processing_time': complexity['estimated_time']
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в analyze endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/stats', methods=['GET'])
def get_enhanced_stats():
    """Получает расширенную статистику системы"""
    try:
        # Статистика диспетчера
        dispatcher_stats = enhanced_dispatcher.get_system_stats()
        
        # Статистика базы знаний
        knowledge_stats = knowledge_chat.get_knowledge_stats()
        
        # Общая статистика
        total_stats = {
            'dispatcher': dispatcher_stats,
            'knowledge_base': knowledge_stats,
            'integration': {
                'knowledge_base_integration': enhanced_dispatcher.knowledge_base_integration,
                'enhanced_responses': True,
                'detailed_analysis': True
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(total_stats)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в stats endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/configure', methods=['POST'])
def configure_enhanced():
    """Настраивает параметры улучшенного диспетчера"""
    try:
        data = request.get_json()
        
        # Настройки интеграции с базой знаний
        if 'knowledge_integration' in data:
            enhanced_dispatcher.knowledge_base_integration = data['knowledge_integration']
        
        # Настройки шаблонов ответов
        if 'response_templates' in data:
            enhanced_dispatcher.response_templates.update(data['response_templates'])
        
        return jsonify({
            'status': 'success',
            'message': 'Настройки обновлены',
            'settings': {
                'knowledge_integration': enhanced_dispatcher.knowledge_base_integration,
                'response_templates_count': len(enhanced_dispatcher.response_templates)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в configure endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/health', methods=['GET'])
def enhanced_health():
    """Проверка здоровья улучшенной системы"""
    try:
        # Проверяем компоненты
        dispatcher_stats = enhanced_dispatcher.get_system_stats()
        knowledge_available = dispatcher_stats['knowledge_base_available']
        
        # Определяем общий статус
        if dispatcher_stats['available_servers'] >= 5 and knowledge_available:
            status = 'excellent'
            message = 'Все системы работают отлично'
        elif dispatcher_stats['available_servers'] >= 3:
            status = 'good'
            message = 'Основные системы работают'
        else:
            status = 'degraded'
            message = 'Некоторые системы недоступны'
        
        return jsonify({
            'status': status,
            'message': message,
            'components': {
                'dispatcher': 'healthy',
                'knowledge_base': 'healthy' if knowledge_available else 'unavailable',
                'servers_available': dispatcher_stats['available_servers'],
                'total_servers': dispatcher_stats['total_servers']
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/enhanced/demo', methods=['GET'])
def demo_enhanced():
    """Демонстрация возможностей улучшенной системы"""
    try:
        demo_questions = [
            "Что такое закон Ома?",
            "Реши уравнение x^2 + 5x + 6 = 0",
            "Как работает транзистор?",
            "Напиши программу на Python для сортировки",
            "Объясни принцип работы ПЛК"
        ]
        
        demo_results = []
        
        for question in demo_questions:
            result = enhanced_dispatcher.route_question(question)
            demo_results.append({
                'question': question,
                'success': result['success'],
                'server_type': result['server_type'],
                'complexity': result['complexity']['level'],
                'response_length': result['response_length'],
                'preview': result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
            })
        
        return jsonify({
            'demo_results': demo_results,
            'summary': {
                'total_questions': len(demo_questions),
                'successful_responses': sum(1 for r in demo_results if r['success']),
                'average_response_length': sum(r['response_length'] for r in demo_results) // len(demo_results),
                'complexity_distribution': {
                    'simple': sum(1 for r in demo_results if r['complexity'] == 'simple'),
                    'medium': sum(1 for r in demo_results if r['complexity'] == 'medium'),
                    'complex': sum(1 for r in demo_results if r['complexity'] == 'complex'),
                    'expert': sum(1 for r in demo_results if r['complexity'] == 'expert')
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в demo endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint не найден'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == "__main__":
    print("🚀 Запуск улучшенного API сервера Rubin AI")
    print("📡 Доступные endpoints:")
    print("  POST /api/enhanced/chat - Улучшенный чат с подробными ответами")
    print("  POST /api/enhanced/analyze - Анализ вопроса")
    print("  GET  /api/enhanced/stats - Расширенная статистика")
    print("  POST /api/enhanced/configure - Настройка параметров")
    print("  GET  /api/enhanced/health - Проверка здоровья")
    print("  GET  /api/enhanced/demo - Демонстрация возможностей")
    
    # Запуск сервера
    app.run(host='0.0.0.0', port=8094, debug=True)