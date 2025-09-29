#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для интегрированной системы мышления Rubin AI
Предоставляет endpoints для стимулирования мышления и абдуктивного рассуждения
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from datetime import datetime

from rubin_ai_thinking_system import RubinAIThinkingSystem
from ai_thinking_datasets import AIThinkingDatasets
from non_trivial_queries import NonTrivialQueryGenerator
from abductive_reasoning import AbductiveReasoningEngine, Evidence

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация системы мышления
thinking_system = RubinAIThinkingSystem()

@app.route('/api/thinking/health', methods=['GET'])
def health_check():
    """Проверка состояния системы мышления"""
    return jsonify({
        "status": "Rubin AI Thinking System ONLINE",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "datasets": "active",
            "query_generator": "active", 
            "reasoning_engine": "active",
            "thinking_system": "active"
        }
    }), 200

@app.route('/api/thinking/stimulate', methods=['POST'])
def stimulate_thinking():
    """Стимулирование мышления через нетривиальные запросы"""
    try:
        data = request.get_json() or {}
        domain = data.get('domain', 'general')
        complexity_level = data.get('complexity_level', 4)
        
        if not isinstance(complexity_level, int) or complexity_level < 1 or complexity_level > 5:
            complexity_level = 4
        
        logger.info(f"🎯 Стимулирование мышления: домен={domain}, сложность={complexity_level}")
        
        start_time = time.time()
        result = thinking_system.stimulate_thinking(domain, complexity_level)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        response_data = {
            "success": True,
            "query": result["query"],
            "query_type": result["query_type"],
            "domain": domain,
            "complexity_level": complexity_level,
            "thinking_result": {
                "thinking_depth": result["thinking_result"]["thinking_depth"],
                "confidence_score": result["thinking_result"]["reasoning_result"].confidence_score,
                "best_hypothesis": result["thinking_result"]["reasoning_result"].best_hypothesis.description,
                "evidence_count": result["thinking_result"]["evidence_count"],
                "knowledge_items_used": result["thinking_result"]["knowledge_items_used"]
            },
            "extended_response": result["thinking_result"]["extended_response"],
            "relevant_knowledge": result["relevant_knowledge"],
            "processing_time_seconds": f"{processing_time:.2f}",
            "timestamp": result["timestamp"]
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка стимулирования мышления: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка стимулирования мышления: {str(e)}"
        }), 500

@app.route('/api/thinking/generate_query', methods=['POST'])
def generate_query():
    """Генерация нетривиального запроса"""
    try:
        data = request.get_json() or {}
        domain = data.get('domain', 'general')
        complexity_level = data.get('complexity_level', 4)
        
        logger.info(f"🎯 Генерация запроса: домен={domain}, сложность={complexity_level}")
        
        query_data = thinking_system.query_generator.generate_non_trivial_query(domain, complexity_level)
        complexity_analysis = thinking_system.query_generator.analyze_query_complexity(query_data["query"])
        
        response_data = {
            "success": True,
            "query": query_data["query"],
            "query_type": query_data["query_type"],
            "stimulus_type": query_data["stimulus_type"],
            "thinking_level": query_data["thinking_level"],
            "domain": domain,
            "complexity_analysis": complexity_analysis,
            "timestamp": query_data["generated_at"]
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации запроса: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка генерации запроса: {str(e)}"
        }), 500

@app.route('/api/thinking/reasoning', methods=['POST'])
def perform_reasoning():
    """Выполнение абдуктивного рассуждения"""
    try:
        data = request.get_json() or {}
        evidence_list = data.get('evidence', [])
        domain = data.get('domain', 'general')
        
        if not evidence_list:
            return jsonify({
                "success": False,
                "error": "Не предоставлены доказательства для рассуждения"
            }), 400
        
        logger.info(f"🧠 Абдуктивное рассуждение: домен={domain}, доказательств={len(evidence_list)}")
        
        # Создаем объекты доказательств
        evidence_objects = []
        for i, evidence_data in enumerate(evidence_list):
            evidence = Evidence(
                id=f"ev_{i:03d}",
                description=evidence_data.get('description', ''),
                domain=domain,
                confidence=evidence_data.get('confidence', 0.8),
                timestamp=datetime.now().isoformat(),
                source=evidence_data.get('source', 'user_input')
            )
            evidence_objects.append(evidence)
            thinking_system.reasoning_engine.add_evidence(evidence)
        
        # Выполняем рассуждение
        evidence_ids = [ev.id for ev in evidence_objects]
        reasoning_result = thinking_system.reasoning_engine.perform_abductive_reasoning(evidence_ids, domain)
        
        # Генерируем объяснение
        explanation = thinking_system.reasoning_engine.explain_reasoning(reasoning_result)
        
        response_data = {
            "success": True,
            "reasoning_result": {
                "best_hypothesis": {
                    "description": reasoning_result.best_hypothesis.description,
                    "probability": reasoning_result.best_hypothesis.probability,
                    "complexity_score": reasoning_result.best_hypothesis.complexity_score
                },
                "alternative_hypotheses": [
                    {
                        "description": hyp.description,
                        "probability": hyp.probability
                    }
                    for hyp in reasoning_result.alternative_hypotheses
                ],
                "confidence_score": reasoning_result.confidence_score,
                "reasoning_steps": reasoning_result.reasoning_steps,
                "evidence_count": len(reasoning_result.evidence_used)
            },
            "explanation": explanation,
            "domain": domain,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка абдуктивного рассуждения: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка абдуктивного рассуждения: {str(e)}"
        }), 500

@app.route('/api/thinking/knowledge', methods=['GET'])
def get_knowledge():
    """Получение знаний из специализированных наборов данных"""
    try:
        domain = request.args.get('domain', 'all')
        count = int(request.args.get('count', 5))
        
        logger.info(f"📚 Получение знаний: домен={domain}, количество={count}")
        
        if domain == 'all':
            # Возвращаем знания из всех доменов
            all_knowledge = {}
            for dom in ['electrical', 'math', 'programming', 'controllers']:
                knowledge_items = thinking_system.datasets.get_diverse_representative_data(dom, count)
                all_knowledge[dom] = [
                    {
                        "concept": item.concept,
                        "definition": item.definition,
                        "context": item.context,
                        "examples": item.examples,
                        "complexity_level": item.complexity_level,
                        "confidence_score": item.confidence_score
                    }
                    for item in knowledge_items
                ]
            
            response_data = {
                "success": True,
                "knowledge": all_knowledge,
                "total_domains": len(all_knowledge),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Возвращаем знания из конкретного домена
            knowledge_items = thinking_system.datasets.get_diverse_representative_data(domain, count)
            
            response_data = {
                "success": True,
                "domain": domain,
                "knowledge": [
                    {
                        "concept": item.concept,
                        "definition": item.definition,
                        "context": item.context,
                        "examples": item.examples,
                        "complexity_level": item.complexity_level,
                        "confidence_score": item.confidence_score
                    }
                    for item in knowledge_items
                ],
                "count": len(knowledge_items),
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения знаний: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения знаний: {str(e)}"
        }), 500

@app.route('/api/thinking/analytics', methods=['GET'])
def get_thinking_analytics():
    """Получение аналитики мышления"""
    try:
        logger.info("📊 Получение аналитики мышления")
        
        analytics = thinking_system.get_thinking_analytics()
        
        response_data = {
            "success": True,
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения аналитики: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения аналитики: {str(e)}"
        }), 500

@app.route('/api/thinking/recommendations', methods=['GET'])
def get_learning_recommendations():
    """Получение рекомендаций по обучению"""
    try:
        domain = request.args.get('domain', 'all')
        
        logger.info(f"💡 Получение рекомендаций: домен={domain}")
        
        if domain == 'all':
            recommendations = {}
            for dom in ['electrical', 'math', 'programming', 'controllers']:
                recommendations[dom] = thinking_system.generate_learning_recommendations(dom)
            
            response_data = {
                "success": True,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
        else:
            recommendations = thinking_system.generate_learning_recommendations(domain)
            
            response_data = {
                "success": True,
                "domain": domain,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения рекомендаций: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения рекомендаций: {str(e)}"
        }), 500

@app.route('/api/thinking/dataset_stats', methods=['GET'])
def get_dataset_statistics():
    """Получение статистики наборов данных"""
    try:
        logger.info("📈 Получение статистики наборов данных")
        
        stats = thinking_system.datasets.get_dataset_statistics()
        
        response_data = {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения статистики: {str(e)}"
        }), 500

@app.route('/api/thinking/query_stats', methods=['GET'])
def get_query_statistics():
    """Получение статистики генерации запросов"""
    try:
        logger.info("🎯 Получение статистики генерации запросов")
        
        stats = thinking_system.query_generator.get_thinking_stimulation_stats()
        
        response_data = {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики запросов: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения статистики запросов: {str(e)}"
        }), 500

@app.route('/api/thinking/reasoning_stats', methods=['GET'])
def get_reasoning_statistics():
    """Получение статистики абдуктивного рассуждения"""
    try:
        logger.info("🧠 Получение статистики абдуктивного рассуждения")
        
        stats = thinking_system.reasoning_engine.get_reasoning_statistics()
        
        response_data = {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики рассуждения: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка получения статистики рассуждения: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = 8095  # Новый порт для системы мышления
    logger.info(f"🧠 Запуск Rubin AI Thinking System API сервера на порту {port}...")
    app.run(port=port, debug=True, use_reloader=False)










