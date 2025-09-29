#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система специализированных наборов данных для обучения Rubin AI
Включает разнообразие, репрезентативность, очистку и аннотирование данных
"""

import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeItem:
    """Структура для элемента знаний"""
    id: str
    domain: str
    concept: str
    definition: str
    context: str
    examples: List[str]
    relationships: List[str]
    complexity_level: int  # 1-5
    confidence_score: float  # 0.0-1.0
    metadata: Dict[str, Any]

class AIThinkingDatasets:
    """Система специализированных наборов данных для обучения ИИ"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.domain_datasets = {}
        self.relationship_graph = {}
        self.complexity_levels = {
            1: "базовый",
            2: "промежуточный", 
            3: "продвинутый",
            4: "экспертный",
            5: "исследовательский"
        }
        self._initialize_core_datasets()
        logger.info("🧠 Система специализированных наборов данных инициализирована")
    
    def _initialize_core_datasets(self):
        """Инициализация базовых наборов данных"""
        
        # Электротехника - разнообразные данные
        electrical_data = [
            {
                "concept": "Закон Ома",
                "definition": "Основной закон электротехники, связывающий напряжение, ток и сопротивление",
                "context": "Применяется в расчетах электрических цепей постоянного тока",
                "examples": [
                    "U = I × R - основная формула",
                    "Мощность P = U × I = I² × R",
                    "Применение в резистивных цепях"
                ],
                "relationships": ["закон Кирхгофа", "мощность", "энергия"],
                "complexity_level": 2,
                "confidence_score": 0.95
            },
            {
                "concept": "Транзистор",
                "definition": "Полупроводниковый прибор для усиления и переключения сигналов",
                "context": "Основа современной электроники и цифровых схем",
                "examples": [
                    "Биполярный транзистор (BJT)",
                    "Полевой транзистор (FET)",
                    "MOSFET в цифровых схемах"
                ],
                "relationships": ["усилитель", "переключатель", "логические элементы"],
                "complexity_level": 3,
                "confidence_score": 0.90
            },
            {
                "concept": "ШИМ (Широтно-импульсная модуляция)",
                "definition": "Метод управления мощностью путем изменения длительности импульсов",
                "context": "Используется в инверторах, регуляторах скорости, светодиодных драйверах",
                "examples": [
                    "Управление скоростью двигателя",
                    "Регулировка яркости LED",
                    "DC-DC преобразователи"
                ],
                "relationships": ["частотная модуляция", "цифровое управление", "энергоэффективность"],
                "complexity_level": 4,
                "confidence_score": 0.85
            }
        ]
        
        # Математика - репрезентативные данные
        math_data = [
            {
                "concept": "Квадратное уравнение",
                "definition": "Алгебраическое уравнение второй степени вида ax² + bx + c = 0",
                "context": "Основы алгебры, применяется в физике и инженерии",
                "examples": [
                    "Дискриминант D = b² - 4ac",
                    "Формула корней: x = (-b ± √D) / 2a",
                    "Геометрическая интерпретация - парабола"
                ],
                "relationships": ["дискриминант", "парабола", "оптимизация"],
                "complexity_level": 2,
                "confidence_score": 0.98
            },
            {
                "concept": "Дифференциальное уравнение",
                "definition": "Уравнение, связывающее функцию с её производными",
                "context": "Математическое моделирование динамических систем",
                "examples": [
                    "Уравнение движения: F = ma = m(d²x/dt²)",
                    "Экспоненциальный рост: dy/dt = ky",
                    "Гармонические колебания"
                ],
                "relationships": ["производная", "интеграл", "динамические системы"],
                "complexity_level": 4,
                "confidence_score": 0.92
            }
        ]
        
        # Программирование - контекстуальные данные
        programming_data = [
            {
                "concept": "Алгоритм сортировки",
                "definition": "Процедура упорядочивания элементов списка по определенному критерию",
                "context": "Основы алгоритмики, анализ сложности, оптимизация производительности",
                "examples": [
                    "Быстрая сортировка O(n log n)",
                    "Сортировка пузырьком O(n²)",
                    "Сортировка слиянием - стабильная"
                ],
                "relationships": ["сложность алгоритма", "структуры данных", "оптимизация"],
                "complexity_level": 3,
                "confidence_score": 0.88
            },
            {
                "concept": "Объектно-ориентированное программирование",
                "definition": "Парадигма программирования, основанная на объектах и классах",
                "context": "Современная разработка ПО, инкапсуляция, наследование, полиморфизм",
                "examples": [
                    "Класс как шаблон для создания объектов",
                    "Наследование - расширение функциональности",
                    "Полиморфизм - один интерфейс, разные реализации"
                ],
                "relationships": ["инкапсуляция", "наследование", "полиморфизм"],
                "complexity_level": 3,
                "confidence_score": 0.90
            }
        ]
        
        # Контроллеры - технические данные
        controllers_data = [
            {
                "concept": "ПЛК (Программируемый логический контроллер)",
                "definition": "Цифровое устройство для автоматизации промышленных процессов",
                "context": "Промышленная автоматизация, системы управления, мониторинг",
                "examples": [
                    "Ladder Logic - графическое программирование",
                    "SCADA системы для мониторинга",
                    "HMI интерфейсы для операторов"
                ],
                "relationships": ["SCADA", "HMI", "промышленная сеть"],
                "complexity_level": 3,
                "confidence_score": 0.87
            },
            {
                "concept": "ПИД-регулятор",
                "definition": "Система автоматического управления с пропорционально-интегрально-дифференциальным законом",
                "context": "Системы управления, стабилизация процессов, оптимизация параметров",
                "examples": [
                    "Пропорциональная составляющая - быстрый отклик",
                    "Интегральная составляющая - устранение статической ошибки",
                    "Дифференциальная составляющая - предсказание изменений"
                ],
                "relationships": ["обратная связь", "стабилизация", "оптимизация"],
                "complexity_level": 4,
                "confidence_score": 0.89
            }
        ]
        
        # Создаем структурированные наборы данных
        self.domain_datasets = {
            "electrical": self._process_domain_data("electrical", electrical_data),
            "math": self._process_domain_data("math", math_data),
            "programming": self._process_domain_data("programming", programming_data),
            "controllers": self._process_domain_data("controllers", controllers_data)
        }
        
        # Строим граф связей
        self._build_relationship_graph()
        
        logger.info(f"✅ Инициализировано {len(self.domain_datasets)} доменов знаний")
    
    def _process_domain_data(self, domain: str, raw_data: List[Dict]) -> List[KnowledgeItem]:
        """Обработка и очистка данных домена"""
        processed_data = []
        
        for i, item in enumerate(raw_data):
            # Очистка и нормализация
            cleaned_item = self._clean_and_normalize_data(item)
            
            # Создание структурированного элемента
            knowledge_item = KnowledgeItem(
                id=f"{domain}_{i+1:03d}",
                domain=domain,
                concept=cleaned_item["concept"],
                definition=cleaned_item["definition"],
                context=cleaned_item["context"],
                examples=cleaned_item["examples"],
                relationships=cleaned_item["relationships"],
                complexity_level=cleaned_item["complexity_level"],
                confidence_score=cleaned_item["confidence_score"],
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "domain": domain,
                    "data_quality": self._assess_data_quality(cleaned_item),
                    "cross_references": []
                }
            )
            
            processed_data.append(knowledge_item)
        
        return processed_data
    
    def _clean_and_normalize_data(self, item: Dict) -> Dict:
        """Очистка и нормализация данных"""
        cleaned = {}
        
        # Очистка текстовых полей
        for key in ["concept", "definition", "context"]:
            if key in item:
                # Удаление лишних пробелов и символов
                cleaned[key] = re.sub(r'\s+', ' ', str(item[key]).strip())
                # Нормализация регистра для концепций
                if key == "concept":
                    cleaned[key] = cleaned[key].title()
        
        # Очистка списков
        for key in ["examples", "relationships"]:
            if key in item:
                cleaned[key] = [
                    re.sub(r'\s+', ' ', str(example).strip())
                    for example in item[key]
                    if str(example).strip()
                ]
        
        # Нормализация числовых значений
        cleaned["complexity_level"] = max(1, min(5, int(item.get("complexity_level", 3))))
        cleaned["confidence_score"] = max(0.0, min(1.0, float(item.get("confidence_score", 0.8))))
        
        return cleaned
    
    def _assess_data_quality(self, item: Dict) -> Dict[str, Any]:
        """Оценка качества данных"""
        quality_metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "accuracy": 0.0,
            "relevance": 0.0
        }
        
        # Оценка полноты
        required_fields = ["concept", "definition", "context", "examples"]
        completeness = sum(1 for field in required_fields if field in item and item[field]) / len(required_fields)
        quality_metrics["completeness"] = completeness
        
        # Оценка консистентности
        consistency = 1.0 if len(item.get("examples", [])) >= 2 else 0.5
        quality_metrics["consistency"] = consistency
        
        # Оценка точности (на основе confidence_score)
        quality_metrics["accuracy"] = item.get("confidence_score", 0.8)
        
        # Оценка релевантности
        relevance = 1.0 if len(item.get("relationships", [])) >= 2 else 0.7
        quality_metrics["relevance"] = relevance
        
        return quality_metrics
    
    def _build_relationship_graph(self):
        """Построение графа связей между концепциями"""
        self.relationship_graph = {}
        
        for domain, items in self.domain_datasets.items():
            for item in items:
                concept_id = item.id
                self.relationship_graph[concept_id] = {
                    "concept": item.concept,
                    "domain": item.domain,
                    "relationships": [],
                    "cross_domain_links": []
                }
                
                # Поиск связей внутри домена
                for other_item in items:
                    if other_item.id != concept_id:
                        if any(rel in other_item.relationships for rel in item.relationships):
                            self.relationship_graph[concept_id]["relationships"].append(other_item.id)
                
                # Поиск междоменных связей
                for other_domain, other_items in self.domain_datasets.items():
                    if other_domain != domain:
                        for other_item in other_items:
                            if any(rel in other_item.relationships for rel in item.relationships):
                                self.relationship_graph[concept_id]["cross_domain_links"].append(other_item.id)
        
        logger.info(f"🔗 Построен граф связей с {len(self.relationship_graph)} узлами")
    
    def get_diverse_representative_data(self, domain: str, count: int = 5) -> List[KnowledgeItem]:
        """Получение разнообразных и репрезентативных данных"""
        if domain not in self.domain_datasets:
            return []
        
        items = self.domain_datasets[domain]
        
        # Стратегия разнообразия: выбираем из разных уровней сложности
        complexity_groups = {}
        for item in items:
            level = item.complexity_level
            if level not in complexity_groups:
                complexity_groups[level] = []
            complexity_groups[level].append(item)
        
        # Выбираем репрезентативные элементы
        selected_items = []
        for level in sorted(complexity_groups.keys()):
            if len(selected_items) < count:
                # Выбираем элемент с наивысшим качеством данных
                best_item = max(complexity_groups[level], 
                              key=lambda x: x.metadata["data_quality"]["completeness"])
                selected_items.append(best_item)
        
        # Дополняем до нужного количества случайными элементами
        while len(selected_items) < count and len(selected_items) < len(items):
            remaining_items = [item for item in items if item not in selected_items]
            if remaining_items:
                selected_items.append(random.choice(remaining_items))
        
        return selected_items[:count]
    
    def find_cross_domain_connections(self, concept: str) -> List[Dict[str, Any]]:
        """Поиск междоменных связей для концепции"""
        connections = []
        
        for item_id, item_data in self.relationship_graph.items():
            if concept.lower() in item_data["concept"].lower():
                connections.extend([
                    {
                        "source_concept": concept,
                        "target_concept": item_data["concept"],
                        "target_domain": item_data["domain"],
                        "connection_type": "cross_domain",
                        "strength": len(item_data["cross_domain_links"]) / 10.0
                    }
                    for link_id in item_data["cross_domain_links"]
                    if link_id in self.relationship_graph
                ])
        
        return connections
    
    def generate_annotated_training_data(self, domain: str) -> List[Dict[str, Any]]:
        """Генерация аннотированных данных для обучения"""
        if domain not in self.domain_datasets:
            return []
        
        training_data = []
        
        for item in self.domain_datasets[domain]:
            # Создаем различные форматы аннотаций
            annotations = {
                "basic_qa": {
                    "question": f"Что такое {item.concept.lower()}?",
                    "answer": item.definition,
                    "context": item.context,
                    "domain": domain,
                    "complexity": item.complexity_level
                },
                "detailed_explanation": {
                    "question": f"Объясни подробно {item.concept.lower()}",
                    "answer": f"{item.definition}\n\nКонтекст: {item.context}\n\nПримеры: {'; '.join(item.examples)}",
                    "examples": item.examples,
                    "relationships": item.relationships,
                    "domain": domain,
                    "complexity": item.complexity_level
                },
                "application_scenario": {
                    "question": f"Как применяется {item.concept.lower()}?",
                    "answer": f"Применение: {item.context}\n\nПримеры использования: {'; '.join(item.examples)}",
                    "context": item.context,
                    "examples": item.examples,
                    "domain": domain,
                    "complexity": item.complexity_level
                }
            }
            
            training_data.append({
                "item_id": item.id,
                "concept": item.concept,
                "annotations": annotations,
                "metadata": item.metadata,
                "quality_score": item.confidence_score
            })
        
        return training_data
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Получение статистики по наборам данных"""
        stats = {
            "total_domains": len(self.domain_datasets),
            "total_concepts": sum(len(items) for items in self.domain_datasets.values()),
            "domain_breakdown": {},
            "complexity_distribution": {i: 0 for i in range(1, 6)},
            "quality_metrics": {
                "average_confidence": 0.0,
                "high_quality_items": 0,
                "cross_domain_connections": 0
            }
        }
        
        total_confidence = 0
        total_items = 0
        
        for domain, items in self.domain_datasets.items():
            stats["domain_breakdown"][domain] = len(items)
            
            for item in items:
                stats["complexity_distribution"][item.complexity_level] += 1
                total_confidence += item.confidence_score
                total_items += 1
                
                if item.confidence_score >= 0.9:
                    stats["quality_metrics"]["high_quality_items"] += 1
        
        if total_items > 0:
            stats["quality_metrics"]["average_confidence"] = total_confidence / total_items
        
        stats["quality_metrics"]["cross_domain_connections"] = sum(
            len(item_data["cross_domain_links"]) 
            for item_data in self.relationship_graph.values()
        )
        
        return stats

if __name__ == "__main__":
    print("🧠 Тестирование системы специализированных наборов данных")
    
    datasets = AIThinkingDatasets()
    
    # Тест разнообразия данных
    print("\n📊 Тест разнообразия данных:")
    for domain in ["electrical", "math", "programming", "controllers"]:
        diverse_data = datasets.get_diverse_representative_data(domain, 3)
        print(f"  {domain}: {len(diverse_data)} элементов")
        for item in diverse_data:
            print(f"    • {item.concept} (уровень {item.complexity_level})")
    
    # Тест междоменных связей
    print("\n🔗 Тест междоменных связей:")
    connections = datasets.find_cross_domain_connections("управление")
    print(f"  Найдено {len(connections)} связей для 'управление'")
    
    # Тест аннотированных данных
    print("\n📝 Тест аннотированных данных:")
    training_data = datasets.generate_annotated_training_data("electrical")
    print(f"  Сгенерировано {len(training_data)} аннотированных элементов")
    
    # Статистика
    print("\n📈 Статистика наборов данных:")
    stats = datasets.get_dataset_statistics()
    print(f"  Всего доменов: {stats['total_domains']}")
    print(f"  Всего концепций: {stats['total_concepts']}")
    print(f"  Средняя уверенность: {stats['quality_metrics']['average_confidence']:.2f}")
    print(f"  Междоменных связей: {stats['quality_metrics']['cross_domain_connections']}")
    
    print("\n✅ Тестирование завершено!")










