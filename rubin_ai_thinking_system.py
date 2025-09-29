#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интегрированная система мышления Rubin AI
Объединяет специализированные наборы данных, нетривиальные запросы и абдуктивное рассуждение
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from ai_thinking_datasets import AIThinkingDatasets, KnowledgeItem
from non_trivial_queries import NonTrivialQueryGenerator
from abductive_reasoning import AbductiveReasoningEngine, Evidence, AbductiveReasoning

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class RubinAIThinkingSystem:
    """Интегрированная система мышления Rubin AI"""
    
    def __init__(self):
        self.datasets = AIThinkingDatasets()
        self.query_generator = NonTrivialQueryGenerator()
        self.reasoning_engine = AbductiveReasoningEngine()
        self.thinking_history = []
        self.learning_progress = {}
        self._initialize_thinking_system()
        logger.info("🧠 Интегрированная система мышления Rubin AI инициализирована")
    
    def _initialize_thinking_system(self):
        """Инициализация системы мышления"""
        
        # Инициализируем прогресс обучения по доменам
        domains = ["electrical", "math", "programming", "controllers"]
        for domain in domains:
            self.learning_progress[domain] = {
                "knowledge_items_processed": 0,
                "queries_generated": 0,
                "reasoning_sessions": 0,
                "complexity_level": 1,
                "last_learning_session": None,
                "thinking_patterns": [],
                "improvement_areas": []
            }
        
        logger.info(f"✅ Инициализирован прогресс обучения для {len(domains)} доменов")
    
    def stimulate_thinking(self, domain: str, complexity_level: int = 4) -> Dict[str, Any]:
        """Стимулирование мышления через нетривиальные запросы"""
        
        logger.info(f"🎯 Стимулирование мышления в домене {domain} (уровень {complexity_level})")
        
        # Генерируем нетривиальный запрос
        query_data = self.query_generator.generate_non_trivial_query(domain, complexity_level)
        
        # Анализируем сложность запроса
        complexity_analysis = self.query_generator.analyze_query_complexity(query_data["query"])
        
        # Получаем релевантные знания
        relevant_knowledge = self.datasets.get_diverse_representative_data(domain, 3)
        
        # Генерируем ответ с использованием абдуктивного рассуждения
        thinking_result = self._process_thinking_query(query_data, relevant_knowledge, domain)
        
        # Обновляем прогресс обучения
        self._update_learning_progress(domain, query_data, thinking_result)
        
        # Сохраняем в историю мышления
        self._save_thinking_session(query_data, thinking_result, domain)
        
        return {
            "query": query_data["query"],
            "query_type": query_data["query_type"],
            "thinking_result": thinking_result,
            "complexity_analysis": complexity_analysis,
            "relevant_knowledge": [item.concept for item in relevant_knowledge],
            "domain": domain,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_thinking_query(self, query_data: Dict, knowledge_items: List[KnowledgeItem], domain: str) -> Dict[str, Any]:
        """Обработка запроса мышления"""
        
        # Создаем доказательства на основе знаний
        evidence_list = []
        for i, item in enumerate(knowledge_items):
            evidence = Evidence(
                id=f"ev_{domain}_{i:03d}",
                description=f"Знание: {item.concept} - {item.definition}",
                domain=domain,
                confidence=item.confidence_score,
                timestamp=datetime.now().isoformat(),
                source="knowledge_base"
            )
            evidence_list.append(evidence)
            self.reasoning_engine.add_evidence(evidence)
        
        # Выполняем абдуктивное рассуждение
        evidence_ids = [ev.id for ev in evidence_list]
        reasoning_result = self.reasoning_engine.perform_abductive_reasoning(evidence_ids, domain)
        
        # Генерируем расширенный ответ
        extended_response = self._generate_extended_response(query_data, reasoning_result, knowledge_items)
        
        return {
            "reasoning_result": reasoning_result,
            "extended_response": extended_response,
            "evidence_count": len(evidence_list),
            "knowledge_items_used": len(knowledge_items),
            "thinking_depth": self._calculate_thinking_depth(reasoning_result, knowledge_items)
        }
    
    def _generate_extended_response(self, query_data: Dict, reasoning_result: AbductiveReasoning, 
                                   knowledge_items: List[KnowledgeItem]) -> str:
        """Генерация расширенного ответа"""
        
        response_parts = []
        
        # Введение
        response_parts.append(f"🧠 **РАСШИРЕННОЕ МЫШЛЕНИЕ RUBIN AI**")
        response_parts.append(f"")
        response_parts.append(f"**🎯 Запрос:** {query_data['query']}")
        response_parts.append(f"**📊 Тип мышления:** {query_data['stimulus_type']}")
        response_parts.append(f"**🧠 Уровень сложности:** {query_data['thinking_level']}/5")
        response_parts.append(f"")
        
        # Абдуктивное рассуждение
        response_parts.append(f"**🔍 АБДУКТИВНОЕ РАССУЖДЕНИЕ:**")
        response_parts.append(f"")
        
        explanation = self.reasoning_engine.explain_reasoning(reasoning_result)
        response_parts.append(explanation)
        
        # Междоменные связи
        response_parts.append(f"")
        response_parts.append(f"**🔗 МЕЖДОМЕННЫЕ СВЯЗИ:**")
        response_parts.append(f"")
        
        cross_domain_connections = self.datasets.find_cross_domain_connections(
            reasoning_result.best_hypothesis.description
        )
        
        if cross_domain_connections:
            response_parts.append(f"Обнаружены связи с другими доменами:")
            for connection in cross_domain_connections[:3]:
                response_parts.append(f"• {connection['target_concept']} ({connection['target_domain']}) - сила связи: {connection['strength']:.2f}")
        else:
            response_parts.append(f"Междоменные связи не обнаружены для данной гипотезы.")
        
        # Практические примеры
        response_parts.append(f"")
        response_parts.append(f"**📚 ПРАКТИЧЕСКИЕ ПРИМЕРЫ:**")
        response_parts.append(f"")
        
        for item in knowledge_items:
            response_parts.append(f"**{item.concept}:**")
            response_parts.append(f"{item.definition}")
            if item.examples:
                response_parts.append(f"Примеры: {'; '.join(item.examples[:2])}")
            response_parts.append(f"")
        
        # Творческие инсайты
        response_parts.append(f"**💡 ТВОРЧЕСКИЕ ИНСАЙТЫ:**")
        response_parts.append(f"")
        
        insights = self._generate_creative_insights(query_data, reasoning_result, knowledge_items)
        for insight in insights:
            response_parts.append(f"• {insight}")
        
        # Заключение
        response_parts.append(f"")
        response_parts.append(f"**✅ ЗАКЛЮЧЕНИЕ:**")
        response_parts.append(f"")
        response_parts.append(f"На основе абдуктивного рассуждения и анализа знаний, наилучшим объяснением является:")
        response_parts.append(f"**{reasoning_result.best_hypothesis.description}**")
        response_parts.append(f"")
        response_parts.append(f"Уверенность в результате: {reasoning_result.confidence_score:.2f}")
        response_parts.append(f"Глубина мышления: {self._calculate_thinking_depth(reasoning_result, knowledge_items):.2f}")
        
        return "\n".join(response_parts)
    
    def _generate_creative_insights(self, query_data: Dict, reasoning_result: AbductiveReasoning, 
                                   knowledge_items: List[KnowledgeItem]) -> List[str]:
        """Генерация творческих инсайтов"""
        
        insights = []
        
        # Инсайты на основе типа запроса
        if query_data["stimulus_type"] == "paradox_resolution":
            insights.append("Парадоксы часто указывают на недостаток в нашем понимании системы")
            insights.append("Решение парадокса может привести к новому пониманию принципов")
        
        elif query_data["stimulus_type"] == "cross_domain_connections":
            insights.append("Аналогии между доменами могут раскрыть универсальные принципы")
            insights.append("Перенос знаний между областями стимулирует инновации")
        
        elif query_data["stimulus_type"] == "abductive_inference":
            insights.append("Лучшее объяснение не всегда самое простое")
            insights.append("Абдуктивное рассуждение позволяет находить скрытые причины")
        
        # Инсайты на основе уверенности в результате
        if reasoning_result.confidence_score > 0.8:
            insights.append("Высокая уверенность в результате указывает на надежность гипотезы")
        elif reasoning_result.confidence_score < 0.5:
            insights.append("Низкая уверенность требует дополнительных доказательств")
        
        # Инсайты на основе сложности
        if reasoning_result.best_hypothesis.complexity_score > 0.7:
            insights.append("Сложные гипотезы требуют глубокого понимания системы")
        else:
            insights.append("Простые объяснения часто оказываются наиболее эффективными")
        
        return insights
    
    def _calculate_thinking_depth(self, reasoning_result: AbductiveReasoning, knowledge_items: List[KnowledgeItem]) -> float:
        """Расчет глубины мышления"""
        
        # Факторы глубины мышления
        evidence_factor = len(reasoning_result.evidence_used) / 5.0  # Нормализация до 1.0
        hypothesis_complexity = reasoning_result.best_hypothesis.complexity_score
        alternative_count = len(reasoning_result.alternative_hypotheses) / 3.0  # Нормализация
        knowledge_diversity = len(set(item.domain for item in knowledge_items)) / 4.0  # Нормализация
        
        # Взвешенная сумма
        thinking_depth = (
            evidence_factor * 0.3 +
            hypothesis_complexity * 0.3 +
            alternative_count * 0.2 +
            knowledge_diversity * 0.2
        )
        
        return min(1.0, thinking_depth)
    
    def _update_learning_progress(self, domain: str, query_data: Dict, thinking_result: Dict):
        """Обновление прогресса обучения"""
        
        if domain not in self.learning_progress:
            return
        
        progress = self.learning_progress[domain]
        
        # Обновляем счетчики
        progress["queries_generated"] += 1
        progress["reasoning_sessions"] += 1
        progress["last_learning_session"] = datetime.now().isoformat()
        
        # Обновляем уровень сложности
        if query_data["thinking_level"] > progress["complexity_level"]:
            progress["complexity_level"] = query_data["thinking_level"]
        
        # Добавляем паттерны мышления
        thinking_pattern = {
            "query_type": query_data["query_type"],
            "stimulus_type": query_data["stimulus_type"],
            "thinking_depth": thinking_result["thinking_depth"],
            "timestamp": datetime.now().isoformat()
        }
        progress["thinking_patterns"].append(thinking_pattern)
        
        # Определяем области для улучшения
        if thinking_result["thinking_depth"] < 0.6:
            if "глубина мышления" not in progress["improvement_areas"]:
                progress["improvement_areas"].append("глубина мышления")
        
        if len(thinking_result["reasoning_result"].alternative_hypotheses) < 2:
            if "альтернативные гипотезы" not in progress["improvement_areas"]:
                progress["improvement_areas"].append("альтернативные гипотезы")
    
    def _save_thinking_session(self, query_data: Dict, thinking_result: Dict, domain: str):
        """Сохранение сессии мышления"""
        
        session = {
            "domain": domain,
            "query": query_data["query"],
            "query_type": query_data["query_type"],
            "thinking_depth": thinking_result["thinking_depth"],
            "confidence_score": thinking_result["reasoning_result"].confidence_score,
            "timestamp": datetime.now().isoformat(),
            "session_id": f"session_{len(self.thinking_history) + 1:04d}"
        }
        
        self.thinking_history.append(session)
        
        # Ограничиваем историю последними 100 сессиями
        if len(self.thinking_history) > 100:
            self.thinking_history = self.thinking_history[-100:]
    
    def get_thinking_analytics(self) -> Dict[str, Any]:
        """Получение аналитики мышления"""
        
        analytics = {
            "total_sessions": len(self.thinking_history),
            "domain_breakdown": {},
            "thinking_depth_distribution": {i: 0 for i in range(1, 6)},
            "learning_progress": self.learning_progress,
            "average_thinking_depth": 0.0,
            "improvement_trends": {},
            "most_challenging_domains": []
        }
        
        if not self.thinking_history:
            return analytics
        
        # Анализ по доменам
        for session in self.thinking_history:
            domain = session["domain"]
            if domain not in analytics["domain_breakdown"]:
                analytics["domain_breakdown"][domain] = 0
            analytics["domain_breakdown"][domain] += 1
        
        # Распределение глубины мышления
        total_depth = 0
        for session in self.thinking_history:
            depth = session["thinking_depth"]
            depth_level = int(depth * 5) + 1
            depth_level = min(5, depth_level)
            analytics["thinking_depth_distribution"][depth_level] += 1
            total_depth += depth
        
        analytics["average_thinking_depth"] = total_depth / len(self.thinking_history)
        
        # Тренды улучшения
        for domain, progress in self.learning_progress.items():
            if progress["thinking_patterns"]:
                recent_patterns = progress["thinking_patterns"][-5:]  # Последние 5 сессий
                avg_recent_depth = sum(p["thinking_depth"] for p in recent_patterns) / len(recent_patterns)
                
                if len(progress["thinking_patterns"]) >= 10:
                    older_patterns = progress["thinking_patterns"][-10:-5]  # Предыдущие 5 сессий
                    avg_older_depth = sum(p["thinking_depth"] for p in older_patterns) / len(older_patterns)
                    
                    improvement = avg_recent_depth - avg_older_depth
                    analytics["improvement_trends"][domain] = improvement
        
        # Самые сложные домены
        domain_difficulties = {}
        for domain, progress in self.learning_progress.items():
            if progress["thinking_patterns"]:
                avg_depth = sum(p["thinking_depth"] for p in progress["thinking_patterns"]) / len(progress["thinking_patterns"])
                domain_difficulties[domain] = avg_depth
        
        analytics["most_challenging_domains"] = sorted(
            domain_difficulties.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return analytics
    
    def generate_learning_recommendations(self, domain: str) -> List[str]:
        """Генерация рекомендаций по обучению"""
        
        recommendations = []
        
        if domain not in self.learning_progress:
            return ["Домен не найден в системе обучения"]
        
        progress = self.learning_progress[domain]
        
        # Рекомендации на основе уровня сложности
        if progress["complexity_level"] < 3:
            recommendations.append("Увеличить сложность запросов для развития мышления")
        
        # Рекомендации на основе областей улучшения
        for area in progress["improvement_areas"]:
            if area == "глубина мышления":
                recommendations.append("Фокусироваться на более глубоком анализе проблем")
            elif area == "альтернативные гипотезы":
                recommendations.append("Рассматривать больше альтернативных объяснений")
        
        # Рекомендации на основе частоты сессий
        if progress["reasoning_sessions"] < 5:
            recommendations.append("Увеличить количество сессий мышления в этом домене")
        
        # Рекомендации на основе последней сессии
        if progress["last_learning_session"]:
            last_session = datetime.fromisoformat(progress["last_learning_session"])
            days_since_last = (datetime.now() - last_session).days
            
            if days_since_last > 7:
                recommendations.append("Возобновить регулярные сессии мышления")
        
        return recommendations

if __name__ == "__main__":
    print("🧠 Тестирование интегрированной системы мышления Rubin AI")
    
    thinking_system = RubinAIThinkingSystem()
    
    # Тест стимулирования мышления
    print("\n🎯 Тест стимулирования мышления:")
    domains = ["electrical", "math", "programming", "controllers"]
    
    for domain in domains:
        print(f"\n📋 Домен: {domain}")
        result = thinking_system.stimulate_thinking(domain, 4)
        
        print(f"  Запрос: {result['query']}")
        print(f"  Тип: {result['query_type']}")
        print(f"  Глубина мышления: {result['thinking_result']['thinking_depth']:.2f}")
        print(f"  Уверенность: {result['thinking_result']['reasoning_result'].confidence_score:.2f}")
        
        # Показываем превью ответа
        response_preview = result['thinking_result']['extended_response'][:200]
        print(f"  Превью ответа: {response_preview}...")
    
    # Аналитика
    print("\n📊 Аналитика мышления:")
    analytics = thinking_system.get_thinking_analytics()
    print(f"  Всего сессий: {analytics['total_sessions']}")
    print(f"  Средняя глубина мышления: {analytics['average_thinking_depth']:.2f}")
    print(f"  Распределение по доменам: {analytics['domain_breakdown']}")
    
    # Рекомендации
    print("\n💡 Рекомендации по обучению:")
    for domain in domains:
        recommendations = thinking_system.generate_learning_recommendations(domain)
        print(f"  {domain}: {', '.join(recommendations)}")
    
    print("\n✅ Тестирование завершено!")










