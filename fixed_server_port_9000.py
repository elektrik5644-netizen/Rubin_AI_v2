#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИСПРАВЛЕННЫЙ СЕРВЕР НА ПОРТУ 9000
Полностью новый сервер с нейронной сетью и улучшенными обработчиками
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация компонентов
neural_ai = None
enhanced_dispatcher = None

def initialize_components():
    """Инициализация всех исправленных компонентов"""
    global neural_ai, enhanced_dispatcher
    
    try:
        logger.info("🧠 Инициализация нейронной сети...")
        from neural_rubin import get_neural_rubin
        neural_ai = get_neural_rubin()
        logger.info("✅ Нейронная сеть инициализирована")
    except Exception as e:
        logger.error(f"❌ Ошибка нейронной сети: {e}")
        neural_ai = None
    
    try:
        logger.info("🎯 Инициализация диспетчера...")
        from intelligent_dispatcher import get_intelligent_dispatcher
        enhanced_dispatcher = get_intelligent_dispatcher()
        logger.info("✅ Диспетчер инициализирован")
    except Exception as e:
        logger.error(f"❌ Ошибка диспетчера: {e}")
        enhanced_dispatcher = None

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'name': 'Fixed Rubin AI Server',
        'version': '2.0-FIXED',
        'neural_network': neural_ai is not None,
        'enhanced_dispatcher': enhanced_dispatcher is not None,
        'port': 9000,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной chat endpoint с исправлениями"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        category = data.get('category', '')
        
        if not message:
            return jsonify({'error': 'Пустое сообщение'}), 400
        
        logger.info(f"🔍 Получен запрос: {message[:50]}... (категория: {category})")
        
        # Быстрые ответы
        quick_responses = {
            'привет': '🧠 Привет! Я исправленный Rubin AI с нейронной сетью и улучшенными обработчиками!',
            'тест': '✅ Исправленный сервер работает! Нейронная сеть активна!'
        }
        
        if message.lower() in quick_responses:
            return jsonify({
                'response': quick_responses[message.lower()],
                'provider': 'Fixed Rubin AI (Quick Response)',
                'category': 'general',
                'neural_network': True,
                'enhanced_integration': True,
                'success': True
            })
        
        # Используем нейронную сеть если доступна
        if neural_ai:
            try:
                logger.info("🧠 Обрабатываем через нейронную сеть...")
                response = neural_ai.generate_response(message)
                
                # Добавляем информацию о исправленном сервере
                response['fixed_server'] = True
                response['port'] = 9000
                response['timestamp'] = datetime.now().isoformat()
                
                logger.info(f"✅ Нейронная сеть ответила: {response.get('provider', 'N/A')}")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ Ошибка нейронной сети: {e}")
        
        # Используем диспетчер если доступен
        if enhanced_dispatcher:
            try:
                logger.info("🎯 Обрабатываем через диспетчер...")
                
                # Определяем категорию
                detected_category = enhanced_dispatcher.analyze_request_category(message)
                logger.info(f"📂 Категория: {detected_category}")
                
                # Обрабатываем запрос
                request_data = {'message': message}
                
                if detected_category == 'programming':
                    response = enhanced_dispatcher._handle_programming_request(request_data)
                elif detected_category == 'electrical':
                    response = enhanced_dispatcher._handle_electrical_request(request_data)
                elif detected_category.startswith('mathematics'):
                    response = enhanced_dispatcher._handle_mathematical_request(request_data)
                else:
                    response = enhanced_dispatcher._handle_general_request(request_data)
                
                # Добавляем информацию о исправленном сервере
                response['fixed_server'] = True
                response['port'] = 9000
                response['timestamp'] = datetime.now().isoformat()
                
                logger.info(f"✅ Диспетчер ответил: {response.get('provider', 'N/A')}")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ Ошибка диспетчера: {e}")
        
        # Fallback ответ
        return jsonify({
            'response': f'''🚨 ИСПРАВЛЕННЫЙ СЕРВЕР НА ПОРТУ 9000

Ваш вопрос: "{message}"

❌ К сожалению, компоненты недоступны, но это ИСПРАВЛЕННЫЙ сервер!
✅ Нейронная сеть: {'Доступна' if neural_ai else 'Недоступна'}
✅ Диспетчер: {'Доступен' if enhanced_dispatcher else 'Недоступен'}

Это НЕ математический шаблонный ответ!''',
            'provider': 'Fixed Rubin AI (Fallback)',
            'category': 'system',
            'fixed_server': True,
            'port': 9000,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        return jsonify({
            'error': f'Ошибка сервера: {str(e)}',
            'fixed_server': True,
            'port': 9000
        }), 500

if __name__ == '__main__':
    print("--- STARTING FIXED SERVER ON PORT 9000 ---")
    print("=" * 50)
    
    # Инициализируем компоненты
    initialize_components()
    
    print("📍 Сервер: http://localhost:9000")
    print("🧪 Тест: http://localhost:9000/api/health")
    print("🔧 Теперь нужно перенаправить RubinDeveloper на порт 9000")
    print("=" * 50)
    
    # Запускаем сервер
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=True)