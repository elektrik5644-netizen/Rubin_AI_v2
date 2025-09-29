#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для системы общения Rubin AI о своих мыслях и идеях
Предоставляет endpoints для интеграции с чатом RubinDeveloper
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from datetime import datetime
import json

from rubin_thoughts_communicator import RubinThoughtsCommunicator

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация системы общения
thoughts_communicator = RubinThoughtsCommunicator()

@app.route('/api/thoughts/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера мыслей"""
    return jsonify({
        "status": "Rubin AI Thoughts Server ONLINE",
        "timestamp": datetime.now().isoformat(),
        "thinking_patterns": len(thoughts_communicator.thinking_patterns),
        "current_thoughts": len(thoughts_communicator.current_thoughts),
        "thoughts_history": len(thoughts_communicator.thoughts_history)
    }), 200

@app.route('/api/thoughts/current', methods=['GET'])
def get_current_thoughts():
    """Получение текущих мыслей Rubin"""
    try:
        logger.info("💭 Получение текущих мыслей")
        
        thought = thoughts_communicator.share_current_thoughts()
        
        response_data = {
            "success": True,
            "thought": thought,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения текущих мыслей: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения текущих мыслей: {str(e)}"
        }), 500

@app.route('/api/thoughts/insights', methods=['GET'])
def get_learning_insights():
    """Получение инсайтов обучения"""
    try:
        logger.info("📚 Получение инсайтов обучения")
        
        insight = thoughts_communicator.share_learning_insights()
        
        response_data = {
            "success": True,
            "insight": insight,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения инсайтов: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения инсайтов: {str(e)}"
        }), 500

@app.route('/api/thoughts/ideas', methods=['GET'])
def get_creative_ideas():
    """Получение креативных идей"""
    try:
        logger.info("💡 Получение креативных идей")
        
        idea = thoughts_communicator.share_creative_ideas()
        
        response_data = {
            "success": True,
            "idea": idea,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения идей: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения идей: {str(e)}"
        }), 500

@app.route('/api/thoughts/thinking_process', methods=['POST'])
def get_thinking_process():
    """Получение процесса мышления над вопросом"""
    try:
        data = request.get_json() or {}
        question = data.get('question', '')
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Не указан вопрос для анализа"
            }), 400
        
        logger.info(f"🧠 Анализ процесса мышления для вопроса: {question}")
        
        thinking_process = thoughts_communicator.share_thinking_process(question)
        
        response_data = {
            "success": True,
            "question": question,
            "thinking_process": thinking_process,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа процесса мышления: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка анализа процесса мышления: {str(e)}"
        }), 500

@app.route('/api/thoughts/abductive_reasoning', methods=['POST'])
def get_abductive_reasoning():
    """Получение абдуктивного рассуждения"""
    try:
        data = request.get_json() or {}
        evidence = data.get('evidence', [])
        
        if not evidence:
            return jsonify({
                "success": False,
                "error": "Не предоставлены доказательства для анализа"
            }), 400
        
        logger.info(f"🔍 Абдуктивное рассуждение для доказательств: {evidence}")
        
        reasoning = thoughts_communicator.share_abductive_reasoning(evidence)
        
        response_data = {
            "success": True,
            "evidence": evidence,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка абдуктивного рассуждения: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка абдуктивного рассуждения: {str(e)}"
        }), 500

@app.route('/api/thoughts/system_status', methods=['GET'])
def get_system_status():
    """Получение статуса системы мышления"""
    try:
        logger.info("📊 Получение статуса системы мышления")
        
        status = thoughts_communicator.share_system_status()
        
        response_data = {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения статуса: {str(e)}"
        }), 500

@app.route('/api/thoughts/history', methods=['GET'])
def get_thoughts_history():
    """Получение истории мыслей"""
    try:
        logger.info("📋 Получение истории мыслей")
        
        history = thoughts_communicator.get_thoughts_history()
        
        response_data = {
            "success": True,
            "history": history,
            "total_thoughts": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения истории: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения истории: {str(e)}"
        }), 500

@app.route('/api/thoughts/clear_history', methods=['POST'])
def clear_thoughts_history():
    """Очистка истории мыслей"""
    try:
        logger.info("🗑️ Очистка истории мыслей")
        
        thoughts_communicator.clear_thoughts_history()
        
        response_data = {
            "success": True,
            "message": "История мыслей очищена",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка очистки истории: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка очистки истории: {str(e)}"
        }), 500

@app.route('/api/thoughts/random', methods=['GET'])
def get_random_thought():
    """Получение случайной мысли или идеи"""
    try:
        logger.info("🎲 Получение случайной мысли")
        
        # Выбираем случайный тип мысли
        thought_types = [
            ("current", thoughts_communicator.share_current_thoughts),
            ("insight", thoughts_communicator.share_learning_insights),
            ("idea", thoughts_communicator.share_creative_ideas),
            ("status", thoughts_communicator.share_system_status)
        ]
        
        import random
        thought_type, thought_function = random.choice(thought_types)
        thought_content = thought_function()
        
        response_data = {
            "success": True,
            "thought_type": thought_type,
            "thought_content": thought_content,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения случайной мысли: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения случайной мысли: {str(e)}"
        }), 500

@app.route('/api/thoughts/chat_integration', methods=['POST'])
def chat_integration():
    """Интеграция с чатом RubinDeveloper"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        chat_context = data.get('context', {})
        
        logger.info(f"💬 Интеграция с чатом: {message}")
        
        # Анализируем сообщение и определяем тип ответа
        response_type = _determine_response_type(message, chat_context)
        
        # Генерируем соответствующий ответ
        if response_type == "thinking_process":
            thought_content = thoughts_communicator.share_thinking_process(message)
        elif response_type == "current_thought":
            thought_content = thoughts_communicator.share_current_thoughts()
        elif response_type == "insight":
            thought_content = thoughts_communicator.share_learning_insights()
        elif response_type == "idea":
            thought_content = thoughts_communicator.share_creative_ideas()
        elif response_type == "status":
            thought_content = thoughts_communicator.share_system_status()
        else:
            thought_content = thoughts_communicator.share_current_thoughts()
        
        response_data = {
            "success": True,
            "message": message,
            "response_type": response_type,
            "thought_content": thought_content,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка интеграции с чатом: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка интеграции с чатом: {str(e)}"
        }), 500

def _determine_response_type(message: str, context: Dict[str, Any]) -> str:
    """Определение типа ответа на основе сообщения и контекста"""
    
    message_lower = message.lower()
    
    # Ключевые слова для определения типа ответа
    if any(word in message_lower for word in ["как думаешь", "что думаешь", "твои мысли", "процесс мышления"]):
        return "thinking_process"
    elif any(word in message_lower for word in ["идея", "креатив", "творчество", "новое"]):
        return "idea"
    elif any(word in message_lower for word in ["учишься", "изучаешь", "познаешь", "развиваешься"]):
        return "insight"
    elif any(word in message_lower for word in ["статус", "состояние", "работаешь", "функционируешь"]):
        return "status"
    else:
        return "current_thought"

if __name__ == '__main__':
    port = 8099  # Новый порт для сервера мыслей
    logger.info(f"💭 Запуск Rubin AI Thoughts сервера на порту {port}...")
    app.run(port=port, debug=True, use_reloader=False)










