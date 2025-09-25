#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для системы аргументации Rubin AI
Предоставляет endpoints для споров, доказательств и признания ошибок
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from datetime import datetime
import json

from rubin_argumentation_system import RubinArgumentationSystem, Argument, Evidence, EvidenceType, ArgumentStrength

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация системы аргументации
argumentation_system = RubinArgumentationSystem()

@app.route('/api/argumentation/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера аргументации"""
    return jsonify({
        "status": "Rubin AI Argumentation Server ONLINE",
        "timestamp": datetime.now().isoformat(),
        "knowledge_domains": list(argumentation_system.knowledge_base.keys()),
        "evidence_count": len(argumentation_system.evidence_database),
        "debate_history": len(argumentation_system.debate_history)
    }), 200

@app.route('/api/argumentation/create_argument', methods=['POST'])
def create_argument():
    """Создание аргумента"""
    try:
        data = request.get_json() or {}
        claim = data.get('claim', '')
        domain = data.get('domain', 'general')
        evidence_ids = data.get('evidence_ids', [])
        
        if not claim:
            return jsonify({
                "success": False,
                "error": "Не указано утверждение для аргумента"
            }), 400
        
        logger.info(f"⚖️ Создание аргумента: {claim}")
        
        argument = argumentation_system.create_argument(claim, domain, evidence_ids)
        
        # Преобразуем в JSON-сериализуемый формат
        argument_data = {
            "id": argument.id,
            "claim": argument.claim,
            "evidence_list": [
                {
                    "id": ev.id,
                    "description": ev.description,
                    "evidence_type": ev.evidence_type.value,
                    "strength": ev.strength.value,
                    "source": ev.source,
                    "domain": ev.domain,
                    "reliability": ev.reliability
                } for ev in argument.evidence_list
            ],
            "reasoning": argument.reasoning,
            "strength": argument.strength.value,
            "domain": argument.domain,
            "counter_arguments": argument.counter_arguments,
            "timestamp": argument.timestamp
        }
        
        response_data = {
            "success": True,
            "argument": argument_data,
            "message": f"Аргумент '{claim}' создан успешно",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 201
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания аргумента: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка создания аргумента: {str(e)}"
        }), 500

@app.route('/api/argumentation/defend_position', methods=['POST'])
def defend_position():
    """Защита позиции в споре"""
    try:
        data = request.get_json() or {}
        position = data.get('position', '')
        domain = data.get('domain', 'general')
        
        if not position:
            return jsonify({
                "success": False,
                "error": "Не указана позиция для защиты"
            }), 400
        
        logger.info(f"🛡️ Защита позиции: {position}")
        
        debate_position = argumentation_system.defend_position(position, domain)
        
        # Преобразуем аргументы в JSON-сериализуемый формат
        arguments_data = []
        for arg in debate_position.arguments:
            arguments_data.append({
                "id": arg.id,
                "claim": arg.claim,
                "evidence_list": [
                    {
                        "id": ev.id,
                        "description": ev.description,
                        "evidence_type": ev.evidence_type.value,
                        "strength": ev.strength.value,
                        "source": ev.source,
                        "domain": ev.domain,
                        "reliability": ev.reliability
                    } for ev in arg.evidence_list
                ],
                "reasoning": arg.reasoning,
                "strength": arg.strength.value,
                "domain": arg.domain,
                "counter_arguments": arg.counter_arguments
            })
        
        response_data = {
            "success": True,
            "position": debate_position.position,
            "arguments": arguments_data,
            "confidence": debate_position.confidence,
            "evidence_support": debate_position.evidence_support,
            "logical_consistency": debate_position.logical_consistency,
            "message": f"Позиция '{position}' защищена с уверенностью {debate_position.confidence:.2%}",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка защиты позиции: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка защиты позиции: {str(e)}"
        }), 500

@app.route('/api/argumentation/challenge_argument', methods=['POST'])
def challenge_argument():
    """Оспаривание аргумента"""
    try:
        data = request.get_json() or {}
        argument_data = data.get('argument', {})
        challenge = data.get('challenge', '')
        
        if not argument_data or not challenge:
            return jsonify({
                "success": False,
                "error": "Не указан аргумент или вызов"
            }), 400
        
        logger.info(f"🤔 Оспаривание аргумента: {argument_data.get('claim', '')}")
        
        # Восстанавливаем объект Argument из данных
        argument = Argument(
            id=argument_data.get('id', ''),
            claim=argument_data.get('claim', ''),
            evidence_list=[],  # Упрощаем для демонстрации
            reasoning=argument_data.get('reasoning', ''),
            strength=ArgumentStrength(argument_data.get('strength', 'weak')),
            domain=argument_data.get('domain', 'general'),
            counter_arguments=argument_data.get('counter_arguments', []),
            timestamp=argument_data.get('timestamp', '')
        )
        
        challenge_result = argumentation_system.challenge_argument(argument, challenge)
        
        response_data = {
            "success": True,
            "challenge": challenge,
            "challenge_strength": challenge_result['challenge_strength'],
            "fallacies_detected": challenge_result['fallacies_detected'],
            "response": challenge_result['response'],
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка оспаривания аргумента: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка оспаривания аргумента: {str(e)}"
        }), 500

@app.route('/api/argumentation/acknowledge_error', methods=['POST'])
def acknowledge_error():
    """Признание ошибки"""
    try:
        data = request.get_json() or {}
        argument_data = data.get('argument', {})
        error_description = data.get('error_description', '')
        
        if not argument_data or not error_description:
            return jsonify({
                "success": False,
                "error": "Не указан аргумент или описание ошибки"
            }), 400
        
        logger.info(f"✅ Признание ошибки в аргументе: {argument_data.get('claim', '')}")
        
        # Восстанавливаем объект Argument из данных
        argument = Argument(
            id=argument_data.get('id', ''),
            claim=argument_data.get('claim', ''),
            evidence_list=[],  # Упрощаем для демонстрации
            reasoning=argument_data.get('reasoning', ''),
            strength=ArgumentStrength(argument_data.get('strength', 'weak')),
            domain=argument_data.get('domain', 'general'),
            counter_arguments=argument_data.get('counter_arguments', []),
            timestamp=argument_data.get('timestamp', '')
        )
        
        acknowledgment = argumentation_system.acknowledge_error(argument, error_description)
        
        response_data = {
            "success": True,
            "argument_id": argument.id,
            "claim": argument.claim,
            "error_description": error_description,
            "acknowledgment": acknowledgment,
            "message": "Ошибка признана и проанализирована",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка признания ошибки: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка признания ошибки: {str(e)}"
        }), 500

@app.route('/api/argumentation/debate', methods=['POST'])
def conduct_debate():
    """Проведение спора"""
    try:
        data = request.get_json() or {}
        topic = data.get('topic', '')
        position = data.get('position', '')
        domain = data.get('domain', 'general')
        opponent_arguments = data.get('opponent_arguments', [])
        
        if not topic or not position:
            return jsonify({
                "success": False,
                "error": "Не указана тема или позиция для спора"
            }), 400
        
        logger.info(f"⚖️ Проведение спора: {topic}")
        
        # Защищаем свою позицию
        my_position = argumentation_system.defend_position(position, domain)
        
        # Анализируем аргументы оппонента
        opponent_analysis = []
        for opp_arg in opponent_arguments:
            analysis = {
                "argument": opp_arg,
                "strength": argumentation_system._evaluate_challenge_strength(opp_arg, my_position.arguments[0]) if my_position.arguments else 0.5,
                "fallacies": argumentation_system._detect_logical_fallacies(opp_arg),
                "response": argumentation_system._defend_against_challenge(my_position.arguments[0], opp_arg) if my_position.arguments else "Нет аргументов для ответа"
            }
            opponent_analysis.append(analysis)
        
        # Определяем результат спора
        my_confidence = my_position.confidence
        opponent_strength = sum(analysis['strength'] for analysis in opponent_analysis) / max(len(opponent_analysis), 1)
        
        if my_confidence > opponent_strength + 0.2:
            debate_result = "victory"
            result_message = f"Моя позиция '{position}' обоснована сильнее"
        elif opponent_strength > my_confidence + 0.2:
            debate_result = "defeat"
            result_message = f"Аргументы оппонента сильнее моей позиции"
        else:
            debate_result = "draw"
            result_message = f"Спор завершился ничьей - обе стороны имеют веские аргументы"
        
        response_data = {
            "success": True,
            "topic": topic,
            "my_position": {
                "position": my_position.position,
                "confidence": my_position.confidence,
                "evidence_support": my_position.evidence_support,
                "logical_consistency": my_position.logical_consistency,
                "arguments_count": len(my_position.arguments)
            },
            "opponent_analysis": opponent_analysis,
            "debate_result": debate_result,
            "result_message": result_message,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка проведения спора: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка проведения спора: {str(e)}"
        }), 500

@app.route('/api/argumentation/evidence', methods=['GET'])
def get_evidence():
    """Получение списка доказательств"""
    try:
        domain = request.args.get('domain', '')
        
        logger.info("📋 Получение списка доказательств")
        
        evidence_list = argumentation_system.evidence_database
        
        # Фильтруем по домену если указан
        if domain:
            evidence_list = [ev for ev in evidence_list if ev.domain == domain]
        
        # Преобразуем в JSON-сериализуемый формат
        evidence_data = [
            {
                "id": ev.id,
                "description": ev.description,
                "evidence_type": ev.evidence_type.value,
                "strength": ev.strength.value,
                "source": ev.source,
                "domain": ev.domain,
                "reliability": ev.reliability,
                "timestamp": ev.timestamp
            } for ev in evidence_list
        ]
        
        response_data = {
            "success": True,
            "evidence": evidence_data,
            "total_count": len(evidence_data),
            "domain_filter": domain,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения доказательств: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения доказательств: {str(e)}"
        }), 500

@app.route('/api/argumentation/domains', methods=['GET'])
def get_domains():
    """Получение доступных доменов знаний"""
    try:
        logger.info("📚 Получение доменов знаний")
        
        domains_info = {}
        for domain, knowledge in argumentation_system.knowledge_base.items():
            domains_info[domain] = {
                "laws": knowledge.get("laws", []),
                "principles": knowledge.get("principles", []),
                "facts": knowledge.get("facts", []),
                "evidence_count": len([ev for ev in argumentation_system.evidence_database if ev.domain == domain])
            }
        
        response_data = {
            "success": True,
            "domains": domains_info,
            "total_domains": len(domains_info),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения доменов: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения доменов: {str(e)}"
        }), 500

@app.route('/api/argumentation/history', methods=['GET'])
def get_debate_history():
    """Получение истории споров"""
    try:
        logger.info("📋 Получение истории споров")
        
        response_data = {
            "success": True,
            "debate_history": argumentation_system.debate_history,
            "total_debates": len(argumentation_system.debate_history),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения истории: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения истории: {str(e)}"
        }), 500

@app.route('/api/argumentation/chat_integration', methods=['POST'])
def chat_integration():
    """Интеграция с чатом RubinDeveloper"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        context = data.get('context', {})
        
        logger.info(f"💬 Интеграция аргументации с чатом: {message}")
        
        # Анализируем сообщение и определяем тип запроса
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["докажи", "обоснуй", "аргумент", "спор"]):
            # Создание аргумента
            domain = context.get('domain', 'general')
            argument = argumentation_system.create_argument(message, domain)
            
            response_data = {
                "success": True,
                "type": "argument",
                "argument": {
                    "claim": argument.claim,
                    "reasoning": argument.reasoning,
                    "strength": argument.strength.value,
                    "evidence_count": len(argument.evidence_list)
                },
                "message": f"Аргумент создан: {argument.claim}",
                "timestamp": datetime.now().isoformat()
            }
            
        elif any(word in message_lower for word in ["ошибка", "неправильно", "неверно"]):
            # Признание ошибки
            response_data = {
                "success": True,
                "type": "error_acknowledgment",
                "message": "🤔 Вы правы, я готов признать ошибку. Пожалуйста, уточните, в чем именно я ошибся?",
                "timestamp": datetime.now().isoformat()
            }
            
        elif any(word in message_lower for word in ["защити", "защищай", "отстаивай"]):
            # Защита позиции
            domain = context.get('domain', 'general')
            position = argumentation_system.defend_position(message, domain)
            
            response_data = {
                "success": True,
                "type": "position_defense",
                "position": {
                    "position": position.position,
                    "confidence": position.confidence,
                    "arguments_count": len(position.arguments)
                },
                "message": f"Позиция защищена с уверенностью {position.confidence:.2%}",
                "timestamp": datetime.now().isoformat()
            }
            
        else:
            # Общий ответ
            response_data = {
                "success": True,
                "type": "general",
                "message": "⚖️ Я готов к аргументированному обсуждению. Что именно вы хотите обсудить или оспорить?",
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
    port = 8100  # Новый порт для сервера аргументации
    logger.info(f"⚖️ Запуск Rubin AI Argumentation сервера на порту {port}...")
    app.run(port=port, debug=True, use_reloader=False)





