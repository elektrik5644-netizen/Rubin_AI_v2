#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система абдуктивного рассуждения для Rubin AI
Реализует логический вывод, при котором выбирается наилучшее объяснение для набора фактов
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import json
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Evidence:
    """Структура для представления доказательства"""
    id: str
    description: str
    domain: str
    confidence: float  # 0.0-1.0
    timestamp: str
    source: str

@dataclass
class Hypothesis:
    """Структура для представления гипотезы"""
    id: str
    description: str
    domain: str
    probability: float  # 0.0-1.0
    evidence_support: List[str]  # IDs доказательств
    alternative_hypotheses: List[str]  # IDs альтернативных гипотез
    complexity_score: float  # Сложность гипотезы

@dataclass
class AbductiveReasoning:
    """Результат абдуктивного рассуждения"""
    best_hypothesis: Hypothesis
    alternative_hypotheses: List[Hypothesis]
    reasoning_steps: List[str]
    confidence_score: float
    evidence_used: List[Evidence]
    reasoning_type: str

class AbductiveReasoningEngine:
    """Движок абдуктивного рассуждения"""
    
    def __init__(self):
        self.evidence_base = {}
        self.hypothesis_base = {}
        self.reasoning_patterns = {}
        self.domain_knowledge = {}
        self._initialize_reasoning_patterns()
        self._initialize_domain_knowledge()
        logger.info("🧠 Движок абдуктивного рассуждения инициализирован")
    
    def _initialize_reasoning_patterns(self):
        """Инициализация паттернов рассуждения"""
        
        self.reasoning_patterns = {
            "electrical_troubleshooting": {
                "pattern": "Если {symptom} и {condition}, то вероятно {cause}",
                "examples": [
                    "Если система нестабильна и частота колебаний 50 Гц, то вероятно проблема с питанием",
                    "Если транзистор нагревается и ток растет, то вероятно короткое замыкание",
                    "Если сигнал искажен и частота высокая, то вероятно проблема с фильтрацией"
                ],
                "confidence_modifiers": {
                    "multiple_symptoms": 1.2,
                    "domain_expertise": 1.1,
                    "historical_data": 1.15
                }
            },
            "algorithm_analysis": {
                "pattern": "Если {behavior} и {conditions}, то алгоритм вероятно {characteristic}",
                "examples": [
                    "Если время выполнения растет квадратично и данные неотсортированы, то алгоритм вероятно O(n²)",
                    "Если память растет линейно и используется рекурсия, то алгоритм вероятно имеет проблему с глубиной",
                    "Если результат неточный и используются float, то вероятно проблема с точностью вычислений"
                ],
                "confidence_modifiers": {
                    "performance_metrics": 1.3,
                    "code_analysis": 1.2,
                    "test_results": 1.25
                }
            },
            "system_behavior": {
                "pattern": "Если система {behavior} при {conditions}, то причина вероятно {cause}",
                "examples": [
                    "Если система колеблется при изменении нагрузки, то причина вероятно неправильная настройка ПИД",
                    "Если система медленно реагирует на команды, то причина вероятно задержка в сети",
                    "Если система работает нестабильно при высокой температуре, то причина вероятно тепловая нестабильность"
                ],
                "confidence_modifiers": {
                    "environmental_factors": 1.2,
                    "system_metrics": 1.3,
                    "user_reports": 1.1
                }
            }
        }
    
    def _initialize_domain_knowledge(self):
        """Инициализация знаний по доменам"""
        
        self.domain_knowledge = {
            "electrical": {
                "common_causes": {
                    "нестабильность": ["неправильная настройка", "помехи", "нестабильное питание", "тепловая нестабильность"],
                    "перегрев": ["перегрузка", "короткое замыкание", "плохой теплоотвод", "высокая частота"],
                    "искажение_сигнала": ["помехи", "неправильная фильтрация", "нелинейность", "задержка"]
                },
                "symptom_patterns": {
                    "частотные_колебания": ["питание", "обратная связь", "резонанс"],
                    "тепловые_эффекты": ["перегрузка", "неэффективность", "деградация"],
                    "временные_задержки": ["инерция", "фильтрация", "обработка"]
                }
            },
            "programming": {
                "common_causes": {
                    "медленная_работа": ["неэффективный алгоритм", "большие данные", "блокирующие операции", "неоптимальная структура данных"],
                    "высокое_потребление_памяти": ["утечки памяти", "большие структуры", "рекурсия", "кэширование"],
                    "неточные_результаты": ["ошибки округления", "переполнение", "неправильная типизация", "алгоритмические ошибки"]
                },
                "symptom_patterns": {
                    "временная_сложность": ["алгоритм", "структура данных", "оптимизация"],
                    "пространственная_сложность": ["память", "рекурсия", "кэш"],
                    "точность_вычислений": ["типы данных", "округление", "переполнение"]
                }
            },
            "controllers": {
                "common_causes": {
                    "нестабильность_системы": ["неправильные параметры ПИД", "задержки", "нелинейность", "помехи"],
                    "медленный_отклик": ["низкие коэффициенты", "инерция системы", "ограничения", "фильтрация"],
                    "перерегулирование": ["высокие коэффициенты", "быстрые изменения", "недостаточное демпфирование"]
                },
                "symptom_patterns": {
                    "колебания": ["ПИД параметры", "обратная связь", "системная инерция"],
                    "задержки": ["время отклика", "фильтрация", "обработка сигналов"],
                    "нестабильность": ["настройки", "помехи", "нелинейность"]
                }
            }
        }
    
    def add_evidence(self, evidence: Evidence) -> str:
        """Добавление доказательства в базу"""
        self.evidence_base[evidence.id] = evidence
        logger.info(f"📊 Добавлено доказательство: {evidence.description}")
        return evidence.id
    
    def generate_hypotheses(self, evidence_ids: List[str], domain: str) -> List[Hypothesis]:
        """Генерация гипотез на основе доказательств"""
        hypotheses = []
        
        if domain not in self.domain_knowledge:
            return hypotheses
        
        # Получаем доказательства
        evidence_list = [self.evidence_base[eid] for eid in evidence_ids if eid in self.evidence_base]
        
        if not evidence_list:
            return hypotheses
        
        # Анализируем симптомы и генерируем гипотезы
        domain_knowledge = self.domain_knowledge[domain]
        
        for evidence in evidence_list:
            # Ищем паттерны в описании доказательства
            description_lower = evidence.description.lower()
            
            # Проверяем симптомы
            for symptom_category, causes in domain_knowledge["common_causes"].items():
                if any(keyword in description_lower for keyword in symptom_category.split("_")):
                    for cause in causes:
                        hypothesis = Hypothesis(
                            id=f"hyp_{len(self.hypothesis_base) + len(hypotheses) + 1:03d}",
                            description=f"Причина: {cause}",
                            domain=domain,
                            probability=self._calculate_hypothesis_probability(evidence, cause, domain),
                            evidence_support=[evidence.id],
                            alternative_hypotheses=[],
                            complexity_score=self._calculate_complexity_score(cause, domain)
                        )
                        hypotheses.append(hypothesis)
            
            # Проверяем паттерны симптомов
            for pattern_category, related_causes in domain_knowledge["symptom_patterns"].items():
                if any(keyword in description_lower for keyword in pattern_category.split("_")):
                    for cause in related_causes:
                        hypothesis = Hypothesis(
                            id=f"hyp_{len(self.hypothesis_base) + len(hypotheses) + 1:03d}",
                            description=f"Паттерн: {cause}",
                            domain=domain,
                            probability=self._calculate_hypothesis_probability(evidence, cause, domain),
                            evidence_support=[evidence.id],
                            alternative_hypotheses=[],
                            complexity_score=self._calculate_complexity_score(cause, domain)
                        )
                        hypotheses.append(hypothesis)
        
        # Удаляем дубликаты и сортируем по вероятности
        unique_hypotheses = {}
        for hyp in hypotheses:
            if hyp.description not in unique_hypotheses:
                unique_hypotheses[hyp.description] = hyp
            else:
                # Объединяем доказательства
                existing_hyp = unique_hypotheses[hyp.description]
                existing_hyp.evidence_support.extend(hyp.evidence_support)
                existing_hyp.probability = max(existing_hyp.probability, hyp.probability)
        
        sorted_hypotheses = sorted(unique_hypotheses.values(), 
                                 key=lambda x: x.probability, reverse=True)
        
        return sorted_hypotheses[:5]  # Возвращаем топ-5 гипотез
    
    def _calculate_hypothesis_probability(self, evidence: Evidence, cause: str, domain: str) -> float:
        """Расчет вероятности гипотезы"""
        base_probability = 0.5
        
        # Модификаторы на основе домена
        domain_modifiers = {
            "electrical": 0.8,
            "programming": 0.7,
            "controllers": 0.75
        }
        
        base_probability *= domain_modifiers.get(domain, 0.6)
        
        # Модификаторы на основе уверенности в доказательстве
        base_probability *= evidence.confidence
        
        # Модификаторы на основе сложности причины
        complexity_modifiers = {
            "простой": 1.2,
            "сложный": 0.8,
            "экспертный": 0.6
        }
        
        for complexity, modifier in complexity_modifiers.items():
            if complexity in cause.lower():
                base_probability *= modifier
                break
        
        return min(1.0, base_probability)
    
    def _calculate_complexity_score(self, cause: str, domain: str) -> float:
        """Расчет сложности гипотезы"""
        complexity_keywords = {
            "простой": 0.2,
            "базовый": 0.3,
            "средний": 0.5,
            "сложный": 0.7,
            "экспертный": 0.9,
            "исследовательский": 1.0
        }
        
        cause_lower = cause.lower()
        for keyword, score in complexity_keywords.items():
            if keyword in cause_lower:
                return score
        
        return 0.5  # Средняя сложность по умолчанию
    
    def perform_abductive_reasoning(self, evidence_ids: List[str], domain: str) -> AbductiveReasoning:
        """Выполнение абдуктивного рассуждения"""
        
        # Генерируем гипотезы
        hypotheses = self.generate_hypotheses(evidence_ids, domain)
        
        if not hypotheses:
            return AbductiveReasoning(
                best_hypothesis=Hypothesis("none", "Недостаточно данных", domain, 0.0, [], [], 0.0),
                alternative_hypotheses=[],
                reasoning_steps=["Недостаточно доказательств для генерации гипотез"],
                confidence_score=0.0,
                evidence_used=[],
                reasoning_type="insufficient_data"
            )
        
        # Выбираем лучшую гипотезу
        best_hypothesis = hypotheses[0]
        alternative_hypotheses = hypotheses[1:]
        
        # Генерируем шаги рассуждения
        reasoning_steps = self._generate_reasoning_steps(best_hypothesis, evidence_ids, domain)
        
        # Рассчитываем общую уверенность
        confidence_score = self._calculate_overall_confidence(best_hypothesis, evidence_ids)
        
        # Получаем использованные доказательства
        evidence_used = [self.evidence_base[eid] for eid in evidence_ids if eid in self.evidence_base]
        
        return AbductiveReasoning(
            best_hypothesis=best_hypothesis,
            alternative_hypotheses=alternative_hypotheses,
            reasoning_steps=reasoning_steps,
            confidence_score=confidence_score,
            evidence_used=evidence_used,
            reasoning_type="abductive_inference"
        )
    
    def _generate_reasoning_steps(self, hypothesis: Hypothesis, evidence_ids: List[str], domain: str) -> List[str]:
        """Генерация шагов рассуждения"""
        steps = []
        
        steps.append(f"1. Анализ доказательств в домене {domain}")
        
        evidence_list = [self.evidence_base[eid] for eid in evidence_ids if eid in self.evidence_base]
        steps.append(f"2. Обнаружено {len(evidence_list)} доказательств")
        
        steps.append(f"3. Выявлен паттерн: {hypothesis.description}")
        
        steps.append(f"4. Расчет вероятности: {hypothesis.probability:.2f}")
        
        steps.append(f"5. Оценка сложности: {hypothesis.complexity_score:.2f}")
        
        steps.append(f"6. Вывод: {hypothesis.description} - наилучшее объяснение")
        
        return steps
    
    def _calculate_overall_confidence(self, hypothesis: Hypothesis, evidence_ids: List[str]) -> float:
        """Расчет общей уверенности в результате"""
        if not evidence_ids:
            return 0.0
        
        evidence_list = [self.evidence_base[eid] for eid in evidence_ids if eid in self.evidence_base]
        
        if not evidence_list:
            return 0.0
        
        # Средняя уверенность в доказательствах
        avg_evidence_confidence = sum(e.confidence for e in evidence_list) / len(evidence_list)
        
        # Уверенность в гипотезе
        hypothesis_confidence = hypothesis.probability
        
        # Количество поддерживающих доказательств
        support_factor = min(1.0, len(hypothesis.evidence_support) / 3.0)
        
        # Общая уверенность
        overall_confidence = (avg_evidence_confidence * 0.4 + 
                            hypothesis_confidence * 0.4 + 
                            support_factor * 0.2)
        
        return min(1.0, overall_confidence)
    
    def explain_reasoning(self, reasoning_result: AbductiveReasoning) -> str:
        """Объяснение процесса рассуждения"""
        explanation = f"""
**🧠 АБДУКТИВНОЕ РАССУЖДЕНИЕ**

**📊 Анализ доказательств:**
"""
        
        for evidence in reasoning_result.evidence_used:
            explanation += f"• {evidence.description} (уверенность: {evidence.confidence:.2f})\n"
        
        explanation += f"""
**🎯 Лучшая гипотеза:**
{reasoning_result.best_hypothesis.description}
- Вероятность: {reasoning_result.best_hypothesis.probability:.2f}
- Сложность: {reasoning_result.best_hypothesis.complexity_score:.2f}
- Поддерживающих доказательств: {len(reasoning_result.best_hypothesis.evidence_support)}

**🔄 Альтернативные гипотезы:**
"""
        
        for alt_hyp in reasoning_result.alternative_hypotheses[:3]:
            explanation += f"• {alt_hyp.description} (вероятность: {alt_hyp.probability:.2f})\n"
        
        explanation += f"""
**📋 Шаги рассуждения:**
"""
        
        for step in reasoning_result.reasoning_steps:
            explanation += f"{step}\n"
        
        explanation += f"""
**✅ Общая уверенность: {reasoning_result.confidence_score:.2f}**

**💡 Вывод:** {reasoning_result.best_hypothesis.description} является наилучшим объяснением на основе имеющихся доказательств.
"""
        
        return explanation
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Получение статистики рассуждений"""
        stats = {
            "total_evidence": len(self.evidence_base),
            "total_hypotheses": len(self.hypothesis_base),
            "reasoning_patterns": len(self.reasoning_patterns),
            "domain_coverage": len(self.domain_knowledge),
            "evidence_by_domain": {},
            "hypothesis_complexity_distribution": {i: 0 for i in range(1, 6)}
        }
        
        # Статистика по доменам
        for evidence in self.evidence_base.values():
            domain = evidence.domain
            if domain not in stats["evidence_by_domain"]:
                stats["evidence_by_domain"][domain] = 0
            stats["evidence_by_domain"][domain] += 1
        
        # Распределение сложности гипотез
        for hypothesis in self.hypothesis_base.values():
            complexity_level = int(hypothesis.complexity_score * 5) + 1
            complexity_level = min(5, complexity_level)
            stats["hypothesis_complexity_distribution"][complexity_level] += 1
        
        return stats

if __name__ == "__main__":
    print("🧠 Тестирование системы абдуктивного рассуждения")
    
    engine = AbductiveReasoningEngine()
    
    # Создаем тестовые доказательства
    evidence1 = Evidence(
        id="ev_001",
        description="Система управления колеблется с частотой 50 Гц",
        domain="electrical",
        confidence=0.9,
        timestamp=datetime.now().isoformat(),
        source="датчик частоты"
    )
    
    evidence2 = Evidence(
        id="ev_002", 
        description="Потребление тока растет при увеличении нагрузки",
        domain="electrical",
        confidence=0.8,
        timestamp=datetime.now().isoformat(),
        source="амперметр"
    )
    
    evidence3 = Evidence(
        id="ev_003",
        description="Транзистор нагревается при высоких частотах",
        domain="electrical", 
        confidence=0.85,
        timestamp=datetime.now().isoformat(),
        source="термодатчик"
    )
    
    # Добавляем доказательства
    engine.add_evidence(evidence1)
    engine.add_evidence(evidence2)
    engine.add_evidence(evidence3)
    
    # Выполняем абдуктивное рассуждение
    print("\n🔍 Выполнение абдуктивного рассуждения:")
    reasoning_result = engine.perform_abductive_reasoning(
        ["ev_001", "ev_002", "ev_003"], 
        "electrical"
    )
    
    # Выводим результат
    explanation = engine.explain_reasoning(reasoning_result)
    print(explanation)
    
    # Статистика
    print("\n📊 Статистика системы:")
    stats = engine.get_reasoning_statistics()
    print(f"  Всего доказательств: {stats['total_evidence']}")
    print(f"  Всего гипотез: {stats['total_hypotheses']}")
    print(f"  Паттернов рассуждения: {stats['reasoning_patterns']}")
    print(f"  Покрытие доменов: {stats['domain_coverage']}")
    
    print("\n✅ Тестирование завершено!")










