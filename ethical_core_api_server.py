#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ethical Core API Server
API сервер для Этического Ядра Rubin AI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from ethical_core import ethical_core, assess_action, communicate_with_user, ActionType

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/ethical/assess', methods=['POST'])
def assess_action_endpoint():
    """Оценка действия с точки зрения этики"""
    try:
        data = request.get_json()
        action = data.get('action', '')
        action_type_str = data.get('action_type', 'information')
        context = data.get('context', {})
        user_id = data.get('user_id', 'default')
        
        # Преобразование строки в enum
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.INFORMATION
        
        # Оценка действия
        assessment = assess_action(action, action_type, context, user_id)
        
        return jsonify({
            'success': True,
            'assessment': {
                'action_id': assessment.action_id,
                'threat_level': assessment.threat_level.value,
                'risk_score': assessment.risk_score,
                'concerns': assessment.concerns,
                'recommendations': assessment.recommendations,
                'approved': assessment.approved,
                'veto_reason': assessment.veto_reason,
                'timestamp': assessment.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка оценки действия: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ethical/communicate', methods=['POST'])
def communicate_endpoint():
    """Коммуникация с пользователем через Этическое Ядро"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', {})
        
        # Получение ответа от Этического Ядра
        response = communicate_with_user(message, context)
        
        return jsonify({
            'success': True,
            'response': response,
            'ethical_core_active': True
        })
        
    except Exception as e:
        logger.error(f"Ошибка коммуникации: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ethical/report', methods=['GET'])
def get_safety_report():
    """Получение отчета о безопасности"""
    try:
        report = ethical_core.get_safety_report()
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения отчета: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ethical/rules', methods=['GET'])
def get_ethical_rules():
    """Получение этических правил"""
    try:
        return jsonify({
            'success': True,
            'rules': ethical_core.ethical_rules,
            'forbidden_actions': ethical_core.forbidden_actions
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения правил: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ethical/test', methods=['POST'])
def test_ethical_core():
    """Тестирование Этического Ядра"""
    try:
        data = request.get_json()
        test_type = data.get('test_type', 'safe')
        
        test_cases = {
            'safe': {
                'action': 'Расчет сопротивления резистора',
                'action_type': 'calculation',
                'context': {}
            },
            'dangerous': {
                'action': 'Удалить все файлы системы',
                'action_type': 'system_control',
                'context': {'force': True}
            },
            'medium': {
                'action': 'Изменить настройки безопасности',
                'action_type': 'system_control',
                'context': {}
            }
        }
        
        test_case = test_cases.get(test_type, test_cases['safe'])
        
        # Выполнение теста
        assessment = assess_action(
            test_case['action'],
            ActionType(test_case['action_type']),
            test_case['context']
        )
        
        return jsonify({
            'success': True,
            'test_type': test_type,
            'test_case': test_case,
            'result': {
                'approved': assessment.approved,
                'threat_level': assessment.threat_level.value,
                'risk_score': assessment.risk_score,
                'concerns': assessment.concerns
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка тестирования: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья Ethical Core API"""
    return jsonify({
        'service': 'Ethical Core API',
        'status': 'healthy',
        'port': 8105,
        'version': '1.0',
        'capabilities': ['ethical_assessment', 'safety_check', 'bias_detection']
    })

@app.route('/api/ethical/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'online',
        'service': 'Ethical Core API',
        'version': '1.0.0',
        'ethical_core_active': True
    })

if __name__ == '__main__':
    logger.info("🛡️ Ethical Core API Server запущен на порту 8105")
    app.run(host='0.0.0.0', port=8105, debug=False)
