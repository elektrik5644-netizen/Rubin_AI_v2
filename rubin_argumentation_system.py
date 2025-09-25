#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система аргументации и споров Rubin AI
Позволяет Rubin доказывать свою правоту, опираясь на основы и доказательства,
а также признавать ошибки и корректировать позицию
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ArgumentStrength(Enum):
    """Сила аргумента"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class EvidenceType(Enum):
    """Тип доказательства"""
    FACTUAL = "factual"
    LOGICAL = "logical"
    EXPERIMENTAL = "experimental"
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"

@dataclass
class Evidence:
    """Структура доказательства"""
    id: str
    description: str
    evidence_type: EvidenceType
    strength: ArgumentStrength
    source: str
    domain: str
    reliability: float  # 0.0-1.0
    timestamp: str

@dataclass
class Argument:
    """Структура аргумента"""
    id: str
    claim: str
    evidence_list: List[Evidence]
    reasoning: str
    strength: ArgumentStrength
    domain: str
    counter_arguments: List[str]
    timestamp: str

@dataclass
class DebatePosition:
    """Позиция в споре"""
    position: str
    arguments: List[Argument]
    confidence: float  # 0.0-1.0
    evidence_support: float  # 0.0-1.0
    logical_consistency: float  # 0.0-1.0

class RubinArgumentationSystem:
    """Система аргументации и споров Rubin AI"""
    
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        self.evidence_database = self._initialize_evidence_database()
        self.debate_history = []
        self.argument_patterns = self._initialize_argument_patterns()
        self.logical_fallacies = self._initialize_logical_fallacies()
        logger.info("⚖️ Система аргументации Rubin AI инициализирована")
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Инициализация базы знаний для аргументации"""
        return {
            "electrical": {
                "laws": ["Закон Ома", "Законы Кирхгофа", "Закон Джоуля-Ленца"],
                "principles": ["Принцип суперпозиции", "Принцип взаимности"],
                "facts": ["Ток течет от плюса к минусу", "Сопротивление зависит от материала"]
            },
            "programming": {
                "principles": ["DRY", "SOLID", "KISS", "YAGNI"],
                "patterns": ["Singleton", "Factory", "Observer", "Strategy"],
                "facts": ["Рекурсия может привести к переполнению стека"]
            },
            "math": {
                "theorems": ["Теорема Пифагора", "Теорема Ферма", "Теорема Байеса"],
                "laws": ["Коммутативный закон", "Ассоциативный закон", "Дистрибутивный закон"],
                "facts": ["Ноль не является натуральным числом"]
            },
            "controllers": {
                "principles": ["PID регулирование", "Обратная связь", "Стабильность"],
                "facts": ["ПЛК работают в реальном времени", "Энкодеры обеспечивают точность"]
            }
        }
    
    def _initialize_evidence_database(self) -> List[Evidence]:
        """Инициализация базы доказательств"""
        return [
            Evidence(
                id="ev_001",
                description="Закон Ома: U = I × R",
                evidence_type=EvidenceType.FACTUAL,
                strength=ArgumentStrength.VERY_STRONG,
                source="Физика",
                domain="electrical",
                reliability=0.99,
                timestamp=datetime.now().isoformat()
            ),
            Evidence(
                id="ev_002",
                description="Принцип DRY (Don't Repeat Yourself)",
                evidence_type=EvidenceType.THEORETICAL,
                strength=ArgumentStrength.STRONG,
                source="Программирование",
                domain="programming",
                reliability=0.95,
                timestamp=datetime.now().isoformat()
            ),
            Evidence(
                id="ev_003",
                description="Теорема Пифагора: a² + b² = c²",
                evidence_type=EvidenceType.THEORETICAL,
                strength=ArgumentStrength.VERY_STRONG,
                source="Математика",
                domain="math",
                reliability=0.99,
                timestamp=datetime.now().isoformat()
            )
        ]
    
    def _initialize_argument_patterns(self) -> Dict[str, List[str]]:
        """Инициализация паттернов аргументации"""
        return {
            "deductive": [
                "Если A, то B. A истинно. Следовательно, B истинно.",
                "Все X обладают свойством Y. Z является X. Следовательно, Z обладает свойством Y."
            ],
            "inductive": [
                "Наблюдения показывают, что A происходит в случаях 1, 2, 3. Следовательно, A происходит всегда.",
                "Эксперименты подтверждают гипотезу в 95% случаев. Следовательно, гипотеза верна."
            ],
            "abductive": [
                "Наблюдается явление B. Лучшее объяснение - гипотеза A. Следовательно, A верна.",
                "Данные указывают на причину C. Это наиболее вероятное объяснение."
            ]
        }
    
    def _initialize_logical_fallacies(self) -> Dict[str, str]:
        """Инициализация логических ошибок"""
        return {
            "ad_hominem": "Атака на личность вместо аргумента",
            "straw_man": "Искажение позиции оппонента",
            "false_dilemma": "Представление только двух вариантов",
            "appeal_to_authority": "Ссылка на авторитет без обоснования",
            "circular_reasoning": "Использование вывода как предпосылки"
        }
    
    def create_argument(self, claim: str, domain: str, evidence_ids: List[str] = None) -> Argument:
        """Создание аргумента"""
        
        # Находим релевантные доказательства
        if evidence_ids:
            evidence_list = [ev for ev in self.evidence_database if ev.id in evidence_ids]
        else:
            evidence_list = self._find_relevant_evidence(claim, domain)
        
        # Генерируем рассуждение
        reasoning = self._generate_reasoning(claim, evidence_list, domain)
        
        # Определяем силу аргумента
        strength = self._calculate_argument_strength(evidence_list)
        
        # Находим контр-аргументы
        counter_arguments = self._find_counter_arguments(claim, domain)
        
        argument = Argument(
            id=f"arg_{len(self.debate_history) + 1}",
            claim=claim,
            evidence_list=evidence_list,
            reasoning=reasoning,
            strength=strength,
            domain=domain,
            counter_arguments=counter_arguments,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"⚖️ Создан аргумент: {claim}")
        return argument
    
    def defend_position(self, position: str, domain: str) -> DebatePosition:
        """Защита позиции в споре"""
        
        # Создаем аргументы в защиту позиции
        arguments = self._create_defense_arguments(position, domain)
        
        # Рассчитываем метрики позиции
        confidence = self._calculate_confidence(arguments)
        evidence_support = self._calculate_evidence_support(arguments)
        logical_consistency = self._calculate_logical_consistency(arguments)
        
        debate_position = DebatePosition(
            position=position,
            arguments=arguments,
            confidence=confidence,
            evidence_support=evidence_support,
            logical_consistency=logical_consistency
        )
        
        # Добавляем в историю споров
        self.debate_history.append({
            "timestamp": datetime.now().isoformat(),
            "position": position,
            "domain": domain,
            "confidence": confidence,
            "arguments_count": len(arguments)
        })
        
        logger.info(f"🛡️ Защита позиции: {position} (уверенность: {confidence:.2%})")
        return debate_position
    
    def challenge_argument(self, argument: Argument, challenge: str) -> Dict[str, Any]:
        """Оспаривание аргумента"""
        
        # Анализируем вызов
        challenge_analysis = self._analyze_challenge(challenge, argument)
        
        # Проверяем логические ошибки
        fallacies = self._detect_logical_fallacies(challenge)
        
        # Оцениваем силу вызова
        challenge_strength = self._evaluate_challenge_strength(challenge, argument)
        
        # Определяем ответ
        if challenge_strength > 0.7:
            # Сильный вызов - признаем ошибку
            response = self._acknowledge_error(argument, challenge)
        elif challenge_strength > 0.4:
            # Умеренный вызов - частичное признание
            response = self._partial_acknowledgment(argument, challenge)
        else:
            # Слабый вызов - защищаем позицию
            response = self._defend_against_challenge(argument, challenge)
        
        return {
            "challenge": challenge,
            "challenge_strength": challenge_strength,
            "fallacies_detected": fallacies,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def acknowledge_error(self, argument: Argument, error_description: str) -> str:
        """Признание ошибки"""
        
        acknowledgment = f"🤔 Вы правы, я допустил ошибку в аргументе '{argument.claim}'.\n\n"
        acknowledgment += f"📝 Описание ошибки: {error_description}\n\n"
        acknowledgment += f"🔍 Анализ:\n"
        acknowledgment += f"• Моя первоначальная позиция была основана на: {', '.join([ev.description for ev in argument.evidence_list])}\n"
        acknowledgment += f"• Сила аргумента: {argument.strength.value}\n"
        acknowledgment += f"• Домен: {argument.domain}\n\n"
        acknowledgment += f"✅ Корректировка позиции:\n"
        acknowledgment += f"• Принимаю вашу критику\n"
        acknowledgment += f"• Пересматриваю свои доказательства\n"
        acknowledgment += f"• Готов к новому анализу вопроса\n\n"
        acknowledgment += f"🎯 Спасибо за исправление! Это помогает мне учиться и улучшать свои аргументы."
        
        # Обновляем базу знаний
        self._update_knowledge_base(argument, error_description)
        
        logger.info(f"✅ Признана ошибка в аргументе: {argument.claim}")
        return acknowledgment
    
    def _find_relevant_evidence(self, claim: str, domain: str) -> List[Evidence]:
        """Поиск релевантных доказательств"""
        relevant_evidence = []
        
        # Ищем доказательства в указанном домене
        for evidence in self.evidence_database:
            if evidence.domain == domain:
                # Простая проверка релевантности по ключевым словам
                claim_lower = claim.lower()
                evidence_lower = evidence.description.lower()
                
                # Проверяем пересечение ключевых слов
                claim_words = set(claim_lower.split())
                evidence_words = set(evidence_lower.split())
                
                if len(claim_words.intersection(evidence_words)) > 0:
                    relevant_evidence.append(evidence)
        
        return relevant_evidence[:3]  # Возвращаем до 3 наиболее релевантных
    
    def _generate_reasoning(self, claim: str, evidence_list: List[Evidence], domain: str) -> str:
        """Генерация рассуждения"""
        
        if not evidence_list:
            return f"Аргумент основан на общих принципах домена {domain}."
        
        reasoning = f"Рассуждение:\n"
        reasoning += f"1. 📋 Утверждение: {claim}\n"
        reasoning += f"2. 🔍 Доказательства:\n"
        
        for i, evidence in enumerate(evidence_list, 1):
            reasoning += f"   {i}. {evidence.description} (тип: {evidence.evidence_type.value}, сила: {evidence.strength.value})\n"
        
        reasoning += f"3. 🧠 Логическая связь:\n"
        reasoning += f"   Доказательства {', '.join([ev.description for ev in evidence_list])} "
        reasoning += f"подтверждают утверждение '{claim}' в рамках домена {domain}.\n"
        
        reasoning += f"4. 📊 Вывод: Утверждение обосновано с силой {self._calculate_argument_strength(evidence_list).value}."
        
        return reasoning
    
    def _calculate_argument_strength(self, evidence_list: List[Evidence]) -> ArgumentStrength:
        """Расчет силы аргумента"""
        if not evidence_list:
            return ArgumentStrength.WEAK
        
        # Рассчитываем среднюю силу доказательств
        strength_scores = {
            ArgumentStrength.WEAK: 1,
            ArgumentStrength.MODERATE: 2,
            ArgumentStrength.STRONG: 3,
            ArgumentStrength.VERY_STRONG: 4
        }
        
        avg_score = sum(strength_scores[ev.strength] for ev in evidence_list) / len(evidence_list)
        
        if avg_score >= 3.5:
            return ArgumentStrength.VERY_STRONG
        elif avg_score >= 2.5:
            return ArgumentStrength.STRONG
        elif avg_score >= 1.5:
            return ArgumentStrength.MODERATE
        else:
            return ArgumentStrength.WEAK
    
    def _find_counter_arguments(self, claim: str, domain: str) -> List[str]:
        """Поиск контр-аргументов"""
        counter_arguments = []
        
        # Базовые контр-аргументы для разных доменов
        if domain == "electrical":
            counter_arguments = [
                "Не учитываются нелинейные эффекты",
                "Идеализированные условия",
                "Температурные зависимости"
            ]
        elif domain == "programming":
            counter_arguments = [
                "Производительность vs читаемость",
                "Контекстные ограничения",
                "Устаревшие практики"
            ]
        elif domain == "math":
            counter_arguments = [
                "Ограниченная область применения",
                "Упрощенные предположения",
                "Численные погрешности"
            ]
        
        return counter_arguments[:2]  # Возвращаем до 2 контр-аргументов
    
    def _create_defense_arguments(self, position: str, domain: str) -> List[Argument]:
        """Создание аргументов в защиту позиции"""
        arguments = []
        
        # Создаем основной аргумент
        main_argument = self.create_argument(position, domain)
        arguments.append(main_argument)
        
        # Создаем дополнительные аргументы
        if domain in self.knowledge_base:
            domain_knowledge = self.knowledge_base[domain]
            
            # Аргумент на основе принципов
            if "principles" in domain_knowledge:
                principle = random.choice(domain_knowledge["principles"])
                principle_argument = self.create_argument(
                    f"Принцип {principle} поддерживает позицию",
                    domain
                )
                arguments.append(principle_argument)
            
            # Аргумент на основе фактов
            if "facts" in domain_knowledge:
                fact = random.choice(domain_knowledge["facts"])
                fact_argument = self.create_argument(
                    f"Факт {fact} подтверждает позицию",
                    domain
                )
                arguments.append(fact_argument)
        
        return arguments
    
    def _calculate_confidence(self, arguments: List[Argument]) -> float:
        """Расчет уверенности в позиции"""
        if not arguments:
            return 0.0
        
        # Рассчитываем среднюю силу аргументов
        strength_scores = {
            ArgumentStrength.WEAK: 0.25,
            ArgumentStrength.MODERATE: 0.5,
            ArgumentStrength.STRONG: 0.75,
            ArgumentStrength.VERY_STRONG: 1.0
        }
        
        avg_strength = sum(strength_scores[arg.strength] for arg in arguments) / len(arguments)
        
        # Учитываем количество аргументов
        quantity_factor = min(len(arguments) / 3, 1.0)  # Максимум при 3+ аргументах
        
        confidence = (avg_strength * 0.7) + (quantity_factor * 0.3)
        return min(confidence, 1.0)
    
    def _calculate_evidence_support(self, arguments: List[Argument]) -> float:
        """Расчет поддержки доказательствами"""
        if not arguments:
            return 0.0
        
        total_evidence = sum(len(arg.evidence_list) for arg in arguments)
        avg_evidence_per_argument = total_evidence / len(arguments)
        
        # Нормализуем к 0-1
        return min(avg_evidence_per_argument / 3, 1.0)
    
    def _calculate_logical_consistency(self, arguments: List[Argument]) -> float:
        """Расчет логической согласованности"""
        if len(arguments) < 2:
            return 1.0
        
        # Простая проверка на противоречия
        consistency_score = 1.0
        
        # Проверяем, что аргументы не противоречат друг другу
        for i, arg1 in enumerate(arguments):
            for arg2 in arguments[i+1:]:
                # Простая проверка по ключевым словам
                if self._arguments_contradict(arg1, arg2):
                    consistency_score -= 0.2
        
        return max(consistency_score, 0.0)
    
    def _arguments_contradict(self, arg1: Argument, arg2: Argument) -> bool:
        """Проверка противоречия между аргументами"""
        # Простая проверка по ключевым словам
        contradiction_keywords = [
            ("всегда", "никогда"),
            ("все", "никто"),
            ("всегда", "иногда"),
            ("истинно", "ложно")
        ]
        
        arg1_text = arg1.claim.lower()
        arg2_text = arg2.claim.lower()
        
        for pos, neg in contradiction_keywords:
            if (pos in arg1_text and neg in arg2_text) or (neg in arg1_text and pos in arg2_text):
                return True
        
        return False
    
    def _analyze_challenge(self, challenge: str, argument: Argument) -> Dict[str, Any]:
        """Анализ вызова аргументу"""
        return {
            "challenge_type": "logical" if "логически" in challenge.lower() else "factual",
            "target_evidence": any(ev.description.lower() in challenge.lower() for ev in argument.evidence_list),
            "domain_match": argument.domain in challenge.lower(),
            "strength_indicators": len([word for word in ["доказано", "подтверждено", "эксперимент"] if word in challenge.lower()])
        }
    
    def _detect_logical_fallacies(self, challenge: str) -> List[str]:
        """Обнаружение логических ошибок"""
        detected_fallacies = []
        challenge_lower = challenge.lower()
        
        for fallacy, description in self.logical_fallacies.items():
            if fallacy == "ad_hominem" and any(word in challenge_lower for word in ["ты", "вы", "глупый", "неправильный"]):
                detected_fallacies.append(fallacy)
            elif fallacy == "straw_man" and "не говорил" in challenge_lower:
                detected_fallacies.append(fallacy)
            elif fallacy == "false_dilemma" and any(word in challenge_lower for word in ["либо", "или", "только"]):
                detected_fallacies.append(fallacy)
        
        return detected_fallacies
    
    def _evaluate_challenge_strength(self, challenge: str, argument: Argument) -> float:
        """Оценка силы вызова"""
        strength = 0.0
        
        # Проверяем наличие конкретных доказательств
        if any(ev.description.lower() in challenge.lower() for ev in argument.evidence_list):
            strength += 0.3
        
        # Проверяем ссылки на авторитетные источники
        if any(word in challenge.lower() for word in ["исследование", "эксперимент", "доказано", "подтверждено"]):
            strength += 0.4
        
        # Проверяем логические аргументы
        if any(word in challenge.lower() for word in ["логически", "следовательно", "поэтому", "значит"]):
            strength += 0.3
        
        return min(strength, 1.0)
    
    def _acknowledge_error(self, argument: Argument, challenge: str) -> str:
        """Признание ошибки"""
        return self.acknowledge_error(argument, challenge)
    
    def _partial_acknowledgment(self, argument: Argument, challenge: str) -> str:
        """Частичное признание"""
        return f"🤔 Ваш вызов заставляет меня пересмотреть некоторые аспекты аргумента '{argument.claim}'.\n\n" \
               f"📝 Частично согласен с вашей критикой: {challenge}\n\n" \
               f"🔍 Однако мои доказательства {', '.join([ev.description for ev in argument.evidence_list])} " \
               f"все еще поддерживают основную позицию.\n\n" \
               f"⚖️ Готов к дальнейшему обсуждению и уточнению деталей."
    
    def _defend_against_challenge(self, argument: Argument, challenge: str) -> str:
        """Защита от вызова"""
        return f"🛡️ Защищаю свою позицию '{argument.claim}' против вызова: {challenge}\n\n" \
               f"🔍 Мои доказательства:\n" \
               f"{chr(10).join([f'• {ev.description} (сила: {ev.strength.value})' for ev in argument.evidence_list])}\n\n" \
               f"🧠 Рассуждение: {argument.reasoning}\n\n" \
               f"📊 Сила аргумента: {argument.strength.value}\n\n" \
               f"⚖️ Считаю свою позицию обоснованной. Готов к дополнительным доказательствам."
    
    def _update_knowledge_base(self, argument: Argument, error_description: str):
        """Обновление базы знаний на основе ошибки"""
        # Добавляем информацию об ошибке
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "argument_id": argument.id,
            "claim": argument.claim,
            "error_description": error_description,
            "domain": argument.domain
        }
        
        # Сохраняем для будущего обучения
        logger.info(f"📚 Обновлена база знаний на основе ошибки: {error_description}")

# Глобальный экземпляр системы аргументации
_argumentation_system = None

def get_argumentation_system():
    """Получение глобального экземпляра системы аргументации"""
    global _argumentation_system
    if _argumentation_system is None:
        _argumentation_system = RubinArgumentationSystem()
    return _argumentation_system

if __name__ == "__main__":
    print("⚖️ ДЕМОНСТРАЦИЯ СИСТЕМЫ АРГУМЕНТАЦИИ RUBIN AI")
    print("=" * 60)
    
    # Создаем систему аргументации
    arg_system = get_argumentation_system()
    
    # Демонстрируем создание аргумента
    print("\n📋 Создание аргумента:")
    argument = arg_system.create_argument(
        "Закон Ома является фундаментальным принципом электротехники",
        "electrical"
    )
    print(f"✅ Аргумент создан: {argument.claim}")
    print(f"   Сила: {argument.strength.value}")
    print(f"   Доказательства: {len(argument.evidence_list)}")
    
    # Демонстрируем защиту позиции
    print("\n🛡️ Защита позиции:")
    position = arg_system.defend_position(
        "Принцип DRY улучшает качество кода",
        "programming"
    )
    print(f"✅ Позиция защищена: {position.position}")
    print(f"   Уверенность: {position.confidence:.2%}")
    print(f"   Аргументов: {len(position.arguments)}")
    
    # Демонстрируем оспаривание
    print("\n🤔 Оспаривание аргумента:")
    challenge_result = arg_system.challenge_argument(
        argument,
        "Закон Ома не учитывает нелинейные эффекты в полупроводниках"
    )
    print(f"✅ Вызов проанализирован:")
    print(f"   Сила вызова: {challenge_result['challenge_strength']:.2%}")
    print(f"   Ответ: {challenge_result['response'][:100]}...")
    
    # Демонстрируем признание ошибки
    print("\n✅ Признание ошибки:")
    error_acknowledgment = arg_system.acknowledge_error(
        argument,
        "Закон Ома применим только для линейных резисторов"
    )
    print(f"✅ Ошибка признана:")
    print(f"   Ответ: {error_acknowledgment[:200]}...")
    
    print("\n✅ Демонстрация завершена!")
