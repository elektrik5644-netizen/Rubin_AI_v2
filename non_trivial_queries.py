#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система нетривиальных запросов для стимулирования мышления Rubin AI
Побуждает ИИ искать неочевидные связи и выходить за рамки привычных паттернов
"""

import random
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class NonTrivialQueryGenerator:
    """Генератор нетривиальных запросов для стимулирования мышления ИИ"""
    
    def __init__(self):
        self.query_templates = {
            "paradox_queries": [],
            "cross_domain_connections": [],
            "counterintuitive_scenarios": [],
            "abductive_reasoning": [],
            "creative_problem_solving": []
        }
        self._initialize_query_templates()
        logger.info("🎯 Система нетривиальных запросов инициализирована")
    
    def _initialize_query_templates(self):
        """Инициализация шаблонов нетривиальных запросов"""
        
        # Парадоксальные запросы - стимулируют поиск неочевидных решений
        self.query_templates["paradox_queries"] = [
            {
                "template": "Как {concept1} может одновременно {action1} и {action2}?",
                "examples": [
                    "Как транзистор может одновременно усиливать и ослаблять сигнал?",
                    "Как алгоритм может быть и быстрым, и медленным?",
                    "Как система управления может быть и стабильной, и адаптивной?"
                ],
                "stimulus_type": "paradox_resolution",
                "thinking_level": 4
            },
            {
                "template": "Почему {concept1} работает, хотя по логике {reason}?",
                "examples": [
                    "Почему ШИМ работает, хотя по логике постоянный ток должен быть постоянным?",
                    "Почему обратная связь стабилизирует систему, хотя по логике должна её дестабилизировать?",
                    "Почему цифровые сигналы передают аналоговую информацию?"
                ],
                "stimulus_type": "counterintuitive_explanation",
                "thinking_level": 5
            }
        ]
        
        # Междоменные связи - поиск неочевидных аналогий
        self.query_templates["cross_domain_connections"] = [
            {
                "template": "Как {domain1} {concept1} связан с {domain2} {concept2}?",
                "examples": [
                    "Как электротехнический конденсатор связан с программированием буферизации?",
                    "Как математическая производная связана с системой управления ПЛК?",
                    "Как алгоритм сортировки связан с электрическими фильтрами?"
                ],
                "stimulus_type": "analogical_reasoning",
                "thinking_level": 4
            },
            {
                "template": "Что общего между {concept1} и {concept2} на глубоком уровне?",
                "examples": [
                    "Что общего между нейронной сетью и электрической цепью на глубоком уровне?",
                    "Что общего между алгоритмом и физическим законом на глубоком уровне?",
                    "Что общего между системой управления и биологической системой на глубоком уровне?"
                ],
                "stimulus_type": "deep_analogy",
                "thinking_level": 5
            }
        ]
        
        # Контр-интуитивные сценарии
        self.query_templates["counterintuitive_scenarios"] = [
            {
                "template": "Что произойдет, если {impossible_condition}?",
                "examples": [
                    "Что произойдет, если сопротивление будет отрицательным?",
                    "Что произойдет, если время будет течь назад в алгоритме?",
                    "Что произойдет, если обратная связь станет положительной?"
                ],
                "stimulus_type": "impossible_scenario_analysis",
                "thinking_level": 5
            },
            {
                "template": "Как решить {problem}, если {constraint}?",
                "examples": [
                    "Как решить задачу оптимизации, если функция не дифференцируема?",
                    "Как управлять системой, если датчики дают неточные данные?",
                    "Как программировать, если память ограничена одним байтом?"
                ],
                "stimulus_type": "constrained_problem_solving",
                "thinking_level": 4
            }
        ]
        
        # Абдуктивное рассуждение
        self.query_templates["abductive_reasoning"] = [
            {
                "template": "Какое наилучшее объяснение для наблюдения: {observation}?",
                "examples": [
                    "Какое наилучшее объяснение для наблюдения: система управления нестабильна, хотя все параметры настроены правильно?",
                    "Какое наилучшее объяснение для наблюдения: алгоритм работает медленно, хотя сложность должна быть O(n)?",
                    "Какое наилучшее объяснение для наблюдения: электрическая схема потребляет больше мощности, чем рассчитано?"
                ],
                "stimulus_type": "abductive_inference",
                "thinking_level": 5
            },
            {
                "template": "Если {evidence1} и {evidence2}, то что это означает?",
                "examples": [
                    "Если система управления колеблется с частотой 50 Гц и потребляет переменный ток, то что это означает?",
                    "Если алгоритм работает быстрее на отсортированных данных и медленнее на случайных, то что это означает?",
                    "Если транзистор нагревается при высоких частотах и работает нормально при низких, то что это означает?"
                ],
                "stimulus_type": "evidence_synthesis",
                "thinking_level": 4
            }
        ]
        
        # Креативное решение проблем
        self.query_templates["creative_problem_solving"] = [
            {
                "template": "Как {unconventional_approach} может решить {traditional_problem}?",
                "examples": [
                    "Как музыкальная гармония может решить проблему синхронизации в распределенных системах?",
                    "Как принципы биологической эволюции могут решить проблему оптимизации алгоритмов?",
                    "Как архитектурные принципы могут решить проблему проектирования программного обеспечения?"
                ],
                "stimulus_type": "unconventional_solution",
                "thinking_level": 5
            },
            {
                "template": "Что если {radical_change} в {traditional_system}?",
                "examples": [
                    "Что если изменить направление тока в электрической цепи?",
                    "Что если выполнять алгоритм в обратном порядке?",
                    "Что если использовать отрицательную обратную связь в системе управления?"
                ],
                "stimulus_type": "radical_experimentation",
                "thinking_level": 4
            }
        ]
    
    def generate_non_trivial_query(self, domain: str, complexity_level: int = 4) -> Dict[str, Any]:
        """Генерация нетривиального запроса для стимулирования мышления"""
        
        # Выбираем тип запроса на основе уровня сложности
        query_types = []
        if complexity_level >= 3:
            query_types.extend(["paradox_queries", "cross_domain_connections"])
        if complexity_level >= 4:
            query_types.extend(["counterintuitive_scenarios", "abductive_reasoning"])
        if complexity_level >= 5:
            query_types.append("creative_problem_solving")
        
        if not query_types:
            query_types = ["paradox_queries"]
        
        # Выбираем случайный тип запроса
        selected_type = random.choice(query_types)
        template_group = self.query_templates[selected_type]
        template = random.choice(template_group)
        
        # Генерируем конкретный запрос
        query = self._generate_specific_query(template, domain)
        
        return {
            "query": query,
            "query_type": selected_type,
            "stimulus_type": template["stimulus_type"],
            "thinking_level": template["thinking_level"],
            "domain": domain,
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "template_used": template["template"],
                "complexity_requested": complexity_level,
                "actual_complexity": template["thinking_level"]
            }
        }
    
    def _generate_specific_query(self, template: Dict, domain: str) -> str:
        """Генерация конкретного запроса на основе шаблона"""
        
        # Если есть готовые примеры, выбираем подходящий
        if "examples" in template and template["examples"]:
            domain_examples = [
                ex for ex in template["examples"] 
                if domain.lower() in ex.lower() or any(
                    domain_word in ex.lower() 
                    for domain_word in ["электрическ", "математическ", "программирован", "контроллер", "управлен"]
                )
            ]
            
            if domain_examples:
                return random.choice(domain_examples)
            else:
                return random.choice(template["examples"])
        
        # Иначе генерируем на основе шаблона
        template_text = template["template"]
        
        # Простая замена плейсхолдеров
        replacements = {
            "{concept1}": self._get_domain_concept(domain),
            "{concept2}": self._get_domain_concept(domain),
            "{action1}": random.choice(["усиливать", "стабилизировать", "оптимизировать", "контролировать"]),
            "{action2}": random.choice(["ослаблять", "дестабилизировать", "замедлять", "освобождать"]),
            "{reason}": random.choice(["это невозможно", "это противоречит законам", "это нелогично"]),
            "{domain1}": domain,
            "{domain2}": random.choice(["математика", "физика", "биология", "химия"]),
            "{impossible_condition}": random.choice([
                "сопротивление станет отрицательным",
                "время потечет назад",
                "сигнал будет передаваться быстрее света"
            ]),
            "{problem}": random.choice([
                "задачу оптимизации",
                "проблему стабилизации",
                "задачу синхронизации"
            ]),
            "{constraint}": random.choice([
                "ресурсы ограничены",
                "время критично",
                "точность важнее скорости"
            ]),
            "{observation}": random.choice([
                "система ведет себя неожиданно",
                "результат противоречит теории",
                "поведение необъяснимо"
            ]),
            "{evidence1}": random.choice([
                "система колеблется",
                "алгоритм медленный",
                "схема нагревается"
            ]),
            "{evidence2}": random.choice([
                "параметры правильные",
                "сложность низкая",
                "расчеты верные"
            ]),
            "{unconventional_approach}": random.choice([
                "музыкальная гармония",
                "биологическая эволюция",
                "архитектурные принципы"
            ]),
            "{traditional_problem}": random.choice([
                "проблему синхронизации",
                "задачу оптимизации",
                "проблему проектирования"
            ]),
            "{radical_change}": random.choice([
                "изменить направление",
                "инвертировать логику",
                "поменять полярность"
            ]),
            "{traditional_system}": random.choice([
                "электрической цепи",
                "алгоритме",
                "системе управления"
            ])
        }
        
        query = template_text
        for placeholder, replacement in replacements.items():
            query = query.replace(placeholder, replacement)
        
        return query
    
    def _get_domain_concept(self, domain: str) -> str:
        """Получение концепции из домена"""
        domain_concepts = {
            "electrical": ["транзистор", "конденсатор", "ШИМ", "закон Ома", "обратная связь"],
            "math": ["производная", "интеграл", "уравнение", "алгоритм", "оптимизация"],
            "programming": ["алгоритм", "структура данных", "функция", "класс", "рекурсия"],
            "controllers": ["ПЛК", "ПИД-регулятор", "система управления", "датчик", "исполнительный механизм"]
        }
        
        concepts = domain_concepts.get(domain.lower(), ["система", "процесс", "механизм"])
        return random.choice(concepts)
    
    def generate_query_sequence(self, domain: str, count: int = 5) -> List[Dict[str, Any]]:
        """Генерация последовательности нетривиальных запросов"""
        queries = []
        
        # Генерируем запросы с возрастающей сложностью
        for i in range(count):
            complexity = min(5, 3 + i)  # Начинаем с уровня 3, доходим до 5
            query = self.generate_non_trivial_query(domain, complexity)
            queries.append(query)
        
        return queries
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Анализ сложности запроса"""
        complexity_indicators = {
            "paradox_indicators": ["одновременно", "хотя", "несмотря на", "противоречит"],
            "cross_domain_indicators": ["связан с", "аналогично", "как в", "подобно"],
            "counterintuitive_indicators": ["что произойдет", "невозможно", "нелогично"],
            "abductive_indicators": ["объяснение", "означает", "свидетельствует"],
            "creative_indicators": ["необычный", "креативный", "радикальный", "инновационный"]
        }
        
        query_lower = query.lower()
        detected_types = []
        
        for indicator_type, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                detected_types.append(indicator_type)
        
        # Определяем уровень сложности
        complexity_level = 2  # Базовый уровень
        if "paradox_indicators" in detected_types:
            complexity_level = max(complexity_level, 4)
        if "cross_domain_indicators" in detected_types:
            complexity_level = max(complexity_level, 3)
        if "counterintuitive_indicators" in detected_types:
            complexity_level = max(complexity_level, 5)
        if "abductive_indicators" in detected_types:
            complexity_level = max(complexity_level, 4)
        if "creative_indicators" in detected_types:
            complexity_level = max(complexity_level, 5)
        
        return {
            "detected_types": detected_types,
            "complexity_level": complexity_level,
            "stimulation_potential": len(detected_types) / len(complexity_indicators),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def get_thinking_stimulation_stats(self) -> Dict[str, Any]:
        """Получение статистики стимулирования мышления"""
        stats = {
            "total_query_types": len(self.query_templates),
            "query_types_breakdown": {},
            "complexity_distribution": {i: 0 for i in range(1, 6)},
            "stimulation_methods": []
        }
        
        for query_type, templates in self.query_templates.items():
            stats["query_types_breakdown"][query_type] = len(templates)
            
            for template in templates:
                level = template["thinking_level"]
                stats["complexity_distribution"][level] += 1
                
                if template["stimulus_type"] not in stats["stimulation_methods"]:
                    stats["stimulation_methods"].append(template["stimulus_type"])
        
        return stats

if __name__ == "__main__":
    print("🎯 Тестирование системы нетривиальных запросов")
    
    generator = NonTrivialQueryGenerator()
    
    # Тест генерации запросов
    print("\n🧠 Тест генерации нетривиальных запросов:")
    domains = ["electrical", "math", "programming", "controllers"]
    
    for domain in domains:
        print(f"\n📋 Домен: {domain}")
        queries = generator.generate_query_sequence(domain, 3)
        
        for i, query_data in enumerate(queries, 1):
            print(f"  {i}. {query_data['query']}")
            print(f"     Тип: {query_data['query_type']}")
            print(f"     Уровень мышления: {query_data['thinking_level']}")
    
    # Тест анализа сложности
    print("\n🔍 Тест анализа сложности запросов:")
    test_queries = [
        "Как транзистор может одновременно усиливать и ослаблять сигнал?",
        "Что общего между алгоритмом и физическим законом?",
        "Что произойдет, если сопротивление станет отрицательным?",
        "Какое наилучшее объяснение для нестабильности системы?"
    ]
    
    for query in test_queries:
        analysis = generator.analyze_query_complexity(query)
        print(f"  Запрос: {query}")
        print(f"    Уровень сложности: {analysis['complexity_level']}")
        print(f"    Обнаруженные типы: {', '.join(analysis['detected_types'])}")
        print(f"    Потенциал стимулирования: {analysis['stimulation_potential']:.2f}")
    
    # Статистика
    print("\n📊 Статистика системы:")
    stats = generator.get_thinking_stimulation_stats()
    print(f"  Всего типов запросов: {stats['total_query_types']}")
    print(f"  Методы стимулирования: {len(stats['stimulation_methods'])}")
    print(f"  Распределение сложности: {stats['complexity_distribution']}")
    
    print("\n✅ Тестирование завершено!")





