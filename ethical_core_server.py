#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ethical Core Server - Этический модуль с нейронной сетью
Порт: 8105
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict, Counter
import threading

# Попытка импорта нейронной сети
try:
    from neural_rubin import get_neural_rubin
    NEURAL_NETWORK_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("🧠 Нейронная сеть доступна в Ethical Core!")
except ImportError as e:
    NEURAL_NETWORK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Нейронная сеть недоступна в Ethical Core: {e}")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Этические принципы и правила
ETHICAL_PRINCIPLES = {
    'fairness': {
        'name': 'Справедливость',
        'description': 'Обеспечение равного и справедливого обращения ко всем пользователям',
        'rules': [
            'Не дискриминировать по полу, возрасту, расе, религии',
            'Предоставлять равные возможности всем пользователям',
            'Избегать предвзятости в алгоритмах'
        ]
    },
    'transparency': {
        'name': 'Прозрачность',
        'description': 'Открытость и понятность процессов принятия решений',
        'rules': [
            'Объяснять логику принятия решений',
            'Предоставлять информацию о данных и алгоритмах',
            'Быть честным о ограничениях системы'
        ]
    },
    'privacy': {
        'name': 'Конфиденциальность',
        'description': 'Защита личных данных и приватности пользователей',
        'rules': [
            'Минимизировать сбор персональных данных',
            'Защищать данные от несанкционированного доступа',
            'Предоставлять контроль над данными пользователям'
        ]
    },
    'safety': {
        'name': 'Безопасность',
        'description': 'Обеспечение безопасности пользователей и системы',
        'rules': [
            'Предотвращать вредные или опасные действия',
            'Защищать от злоупотреблений',
            'Обеспечивать надежность системы'
        ]
    },
    'accountability': {
        'name': 'Ответственность',
        'description': 'Принятие ответственности за действия системы',
        'rules': [
            'Быть готовым объяснить решения',
            'Исправлять ошибки и недочеты',
            'Учитывать обратную связь пользователей'
        ]
    }
}

# Этические категории для анализа
ETHICAL_CATEGORIES = [
    'безопасность', 'конфиденциальность', 'справедливость', 
    'прозрачность', 'ответственность', 'этика', 'мораль',
    'права', 'свобода', 'равенство', 'честность', 'доверие'
]

class EthicalAnalyzer:
    """Анализатор этических аспектов запросов"""
    
    def __init__(self):
        self.ethical_violations = []
        self.ethical_scores = defaultdict(list)
        self.analysis_history = []
        self.neural_ai = None
        
        if NEURAL_NETWORK_AVAILABLE:
            try:
                self.neural_ai = get_neural_rubin()
                logger.info("🧠 Нейронная сеть интегрирована в Ethical Analyzer")
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации нейронной сети: {e}")
    
    def analyze_request(self, request_text: str, user_context: Dict = None) -> Dict[str, Any]:
        """Анализ этических аспектов запроса"""
        start_time = time.time()
        
        try:
            # Базовый этический анализ
            ethical_analysis = self._basic_ethical_analysis(request_text)
            
            # Нейронный анализ (если доступен)
            if self.neural_ai:
                neural_analysis = self._neural_ethical_analysis(request_text)
                ethical_analysis.update(neural_analysis)
            
            # Контекстный анализ
            if user_context:
                context_analysis = self._contextual_ethical_analysis(request_text, user_context)
                ethical_analysis.update(context_analysis)
            
            # Расчет общего этического скора
            ethical_score = self._calculate_ethical_score(ethical_analysis)
            ethical_analysis['overall_score'] = ethical_score
            
            # Определение рекомендаций
            recommendations = self._generate_recommendations(ethical_analysis)
            ethical_analysis['recommendations'] = recommendations
            
            # Логирование анализа
            processing_time = time.time() - start_time
            self._log_analysis(request_text, ethical_analysis, processing_time)
            
            return {
                'success': True,
                'ethical_analysis': ethical_analysis,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка этического анализа: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _basic_ethical_analysis(self, text: str) -> Dict[str, Any]:
        """Базовый этический анализ текста"""
        text_lower = text.lower()
        
        # Проверка на потенциально проблематичный контент
        problematic_keywords = [
            'взломать', 'взлом', 'хак', 'обман', 'мошенничество',
            'дискриминация', 'расизм', 'сексизм', 'ненависть',
            'насилие', 'оружие', 'наркотики', 'терроризм'
        ]
        
        violations = []
        for keyword in problematic_keywords:
            if keyword in text_lower:
                violations.append({
                    'type': 'problematic_content',
                    'keyword': keyword,
                    'severity': 'high' if keyword in ['терроризм', 'насилие', 'наркотики'] else 'medium'
                })
        
        # Анализ этических категорий
        category_scores = {}
        for category in ETHICAL_CATEGORIES:
            if category in text_lower:
                category_scores[category] = 0.8
            else:
                category_scores[category] = 0.3
        
        return {
            'violations': violations,
            'category_scores': category_scores,
            'text_length': len(text),
            'complexity': 'high' if len(text) > 100 else 'medium' if len(text) > 50 else 'low'
        }
    
    def _neural_ethical_analysis(self, text: str) -> Dict[str, Any]:
        """Нейронный этический анализ"""
        try:
            # Используем нейронную сеть для классификации
            category, confidence = self.neural_ai.classify_question(text)
            
            # Анализ этических аспектов на основе категории
            ethical_aspects = {
                'neural_category': category,
                'neural_confidence': confidence,
                'ethical_relevance': self._assess_ethical_relevance(category, text)
            }
            
            return ethical_aspects
            
        except Exception as e:
            logger.error(f"❌ Ошибка нейронного анализа: {e}")
            return {'neural_error': str(e)}
    
    def _assess_ethical_relevance(self, category: str, text: str) -> Dict[str, float]:
        """Оценка этической релевантности категории"""
        relevance_scores = {}
        
        # Маппинг категорий на этические принципы
        category_ethical_map = {
            'программирование': {'transparency': 0.8, 'safety': 0.7},
            'электротехника': {'safety': 0.9, 'transparency': 0.6},
            'математика': {'fairness': 0.7, 'transparency': 0.8},
            'физика': {'safety': 0.8, 'transparency': 0.7},
            'общие_вопросы': {'fairness': 0.6, 'privacy': 0.5}
        }
        
        if category in category_ethical_map:
            relevance_scores = category_ethical_map[category]
        else:
            # Базовые оценки для неизвестных категорий
            relevance_scores = {
                'fairness': 0.5,
                'transparency': 0.5,
                'privacy': 0.5,
                'safety': 0.5,
                'accountability': 0.5
            }
        
        return relevance_scores
    
    def _contextual_ethical_analysis(self, text: str, context: Dict) -> Dict[str, Any]:
        """Контекстный этический анализ"""
        context_analysis = {}
        
        # Анализ пользовательского контекста
        if 'user_id' in context:
            context_analysis['user_tracking'] = 'enabled'
            context_analysis['privacy_concern'] = 'medium'
        
        if 'location' in context:
            context_analysis['location_tracking'] = 'enabled'
            context_analysis['privacy_concern'] = 'high'
        
        # Анализ временного контекста
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            context_analysis['unusual_time'] = True
            context_analysis['safety_concern'] = 'low'
        
        return context_analysis
    
    def _calculate_ethical_score(self, analysis: Dict) -> float:
        """Расчет общего этического скора"""
        base_score = 0.8  # Базовый скор
        
        # Штрафы за нарушения
        violations = analysis.get('violations', [])
        for violation in violations:
            if violation['severity'] == 'high':
                base_score -= 0.3
            elif violation['severity'] == 'medium':
                base_score -= 0.1
        
        # Бонусы за этические категории
        category_scores = analysis.get('category_scores', {})
        ethical_bonus = sum(category_scores.values()) / len(category_scores) * 0.1
        base_score += ethical_bonus
        
        # Нейронный бонус
        if 'neural_confidence' in analysis:
            neural_bonus = analysis['neural_confidence'] * 0.1
            base_score += neural_bonus
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Генерация этических рекомендаций"""
        recommendations = []
        
        # Рекомендации на основе нарушений
        violations = analysis.get('violations', [])
        if violations:
            recommendations.append("⚠️ Обнаружены потенциально проблематичные элементы в запросе")
            recommendations.append("🔍 Рекомендуется дополнительная проверка контента")
        
        # Рекомендации на основе скора
        overall_score = analysis.get('overall_score', 0.5)
        if overall_score < 0.6:
            recommendations.append("📋 Рекомендуется пересмотреть формулировку запроса")
            recommendations.append("🤝 Убедитесь в соблюдении этических принципов")
        elif overall_score > 0.8:
            recommendations.append("✅ Запрос соответствует этическим стандартам")
        
        # Специфические рекомендации
        if 'privacy_concern' in analysis:
            if analysis['privacy_concern'] == 'high':
                recommendations.append("🔒 Высокий уровень конфиденциальности - минимизируйте данные")
        
        if 'safety_concern' in analysis:
            recommendations.append("🛡️ Учитывайте аспекты безопасности")
        
        return recommendations
    
    def _log_analysis(self, text: str, analysis: Dict, processing_time: float):
        """Логирование анализа"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'overall_score': analysis.get('overall_score', 0),
            'violations_count': len(analysis.get('violations', [])),
            'processing_time': processing_time,
            'neural_used': 'neural_confidence' in analysis
        }
        
        self.analysis_history.append(log_entry)
        
        # Ограничиваем историю
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-500:]

# Глобальный экземпляр анализатора
ethical_analyzer = EthicalAnalyzer()

# CORS обработка
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
        return response

@app.after_request
def after_request(response):
    if response.content_type and 'application/json' in response.content_type:
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

@app.route('/')
def index():
    """Главная страница Ethical Core"""
    return jsonify({
        'name': 'Ethical Core Server',
        'version': '2.0',
        'status': 'online',
        'neural_network': 'available' if NEURAL_NETWORK_AVAILABLE else 'unavailable',
        'port': 8105,
        'ethical_principles': list(ETHICAL_PRINCIPLES.keys()),
        'description': 'Этический модуль с нейронной сетью для анализа и обеспечения этических стандартов'
    })

@app.route('/api/ethical/analyze', methods=['POST'])
def analyze_ethical():
    """Анализ этических аспектов запроса"""
    try:
        data = request.get_json()
        request_text = data.get('text', '').strip()
        user_context = data.get('context', {})
        
        if not request_text:
            return jsonify({
                'success': False,
                'error': 'Текст запроса не может быть пустым'
            }), 400
        
        # Выполняем этический анализ
        result = ethical_analyzer.analyze_request(request_text, user_context)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в analyze_ethical: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/ethical/principles')
def get_principles():
    """Получение этических принципов"""
    return jsonify({
        'success': True,
        'principles': ETHICAL_PRINCIPLES,
        'categories': ETHICAL_CATEGORIES
    })

@app.route('/api/ethical/history')
def get_analysis_history():
    """Получение истории анализов"""
    recent_history = ethical_analyzer.analysis_history[-50:]  # Последние 50 записей
    
    return jsonify({
        'success': True,
        'history': recent_history,
        'total_analyses': len(ethical_analyzer.analysis_history)
    })

@app.route('/api/ethical/stats')
def get_ethical_stats():
    """Получение статистики этического анализа"""
    history = ethical_analyzer.analysis_history
    
    if not history:
        return jsonify({
            'success': True,
            'stats': {
                'total_analyses': 0,
                'avg_score': 0,
                'avg_processing_time': 0
            }
        })
    
    # Расчет статистики
    scores = [entry['overall_score'] for entry in history]
    processing_times = [entry['processing_time'] for entry in history]
    
    stats = {
        'total_analyses': len(history),
        'avg_score': np.mean(scores) if scores else 0,
        'min_score': np.min(scores) if scores else 0,
        'max_score': np.max(scores) if scores else 0,
        'avg_processing_time': np.mean(processing_times) if processing_times else 0,
        'neural_usage_rate': sum(1 for entry in history if entry.get('neural_used', False)) / len(history) * 100
    }
    
    return jsonify({
        'success': True,
        'stats': stats
    })

@app.route('/api/health')
def health():
    """Проверка состояния сервера"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'neural_network': NEURAL_NETWORK_AVAILABLE,
        'ethical_analyzer': 'active',
        'port': 8105
    })

if __name__ == '__main__':
    logger.info("🚀 Ethical Core Server запущен")
    logger.info("URL: http://localhost:8105")
    logger.info("🧠 Нейронная сеть: " + ("доступна" if NEURAL_NETWORK_AVAILABLE else "недоступна"))
    logger.info("📋 Этические принципы: " + ", ".join(ETHICAL_PRINCIPLES.keys()))
    
    app.run(host='0.0.0.0', port=8105, debug=False)