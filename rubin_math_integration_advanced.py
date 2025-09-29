#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 Интеграция расширенного математического решателя с Rubin AI
==============================================================

Интеграция новых возможностей:
- Решение физических формул
- Анализ графиков
- Визуализация данных
- OCR для чтения изображений

Автор: Rubin AI System
Версия: 3.0
"""

import logging
from typing import Dict, Any, Optional
from rubin_advanced_math_solver import AdvancedMathSolver, AdvancedProblemType, AdvancedSolution

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubinAdvancedMathIntegration:
    """Интеграция расширенного математического решателя с Rubin AI"""
    
    def __init__(self):
        self.advanced_solver = AdvancedMathSolver()
        self.integration_status = "active"
        
    def process_advanced_question(self, question: str) -> Dict[str, Any]:
        """Обработка расширенного вопроса"""
        try:
            logger.info(f"🔍 Обработка расширенного вопроса: {question}")
            
            # Решение задачи
            solution = self.advanced_solver.solve_advanced_problem(question)
            
            # Форматирование ответа для Rubin AI
            response = self.format_response_for_rubin(solution)
            
            logger.info(f"✅ Ответ сформирован: {solution.problem_type.value}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки расширенного вопроса: {e}")
            return {
                "response": f"Ошибка обработки расширенного вопроса: {e}",
                "category": "error",
                "confidence": 0.0,
                "type": "advanced_math_error"
            }
    
    def format_response_for_rubin(self, solution: AdvancedSolution) -> Dict[str, Any]:
        """Форматирование ответа для Rubin AI"""
        
        # Базовый ответ
        response_parts = []
        
        # Заголовок в зависимости от типа задачи
        if solution.problem_type == AdvancedProblemType.PHYSICS_FORMULAS:
            response_parts.append("⚡ **ФИЗИЧЕСКИЙ РАСЧЕТ:**")
        elif solution.problem_type == AdvancedProblemType.GRAPH_ANALYSIS:
            response_parts.append("📊 **АНАЛИЗ ГРАФИКА:**")
        elif solution.problem_type == AdvancedProblemType.DATA_VISUALIZATION:
            response_parts.append("📈 **ВИЗУАЛИЗАЦИЯ ДАННЫХ:**")
        else:
            response_parts.append("🧮 **РАСШИРЕННОЕ РЕШЕНИЕ:**")
        
        # Пошаговое решение
        if solution.solution_steps:
            response_parts.append("\n**Пошаговое решение:**")
            for i, step in enumerate(solution.solution_steps, 1):
                response_parts.append(f"{i}. {step}")
        
        # Финальный ответ
        response_parts.append(f"\n**Результат:** {solution.final_answer}")
        
        # Объяснение
        if solution.explanation:
            response_parts.append(f"\n**Объяснение:** {solution.explanation}")
        
        # Дополнительная информация
        if solution.visualization:
            response_parts.append(f"\n**Создан файл:** {solution.visualization}")
        
        if solution.graph_data:
            response_parts.append(f"\n**Данные графика:** {solution.graph_data}")
        
        # Статус верификации
        verification_status = "✅ Проверено" if solution.verification else "⚠️ Требует проверки"
        response_parts.append(f"\n**Статус:** {verification_status}")
        
        # Уверенность
        confidence_percent = solution.confidence * 100
        response_parts.append(f"**Уверенность:** {confidence_percent:.1f}%")
        
        return {
            "response": "\n".join(response_parts),
            "category": solution.problem_type.value,
            "confidence": solution.confidence,
            "type": "advanced_math",
            "verification": solution.verification,
            "visualization": solution.visualization,
            "graph_data": solution.graph_data
        }
    
    def get_supported_problem_types(self) -> Dict[str, str]:
        """Получение поддерживаемых типов задач"""
        return {
            "физические_формулы": "Решение задач по физике с использованием формул",
            "анализ_графиков": "Анализ графиков и изображений с помощью OCR",
            "визуализация_данных": "Создание графиков и диаграмм",
            "химические_расчеты": "Расчеты по химическим формулам",
            "инженерные_расчеты": "Инженерные расчеты и проектирование",
            "статистический_анализ": "Статистический анализ данных"
        }
    
    def test_integration(self) -> Dict[str, Any]:
        """Тестирование интеграции"""
        test_cases = [
            {
                "question": "Найти кинетическую энергию тела массой 2 кг, движущегося со скоростью 10 м/с",
                "expected_type": AdvancedProblemType.PHYSICS_FORMULAS
            },
            {
                "question": "Построить линейный график A: 1,2,3,4,5 B: 2,4,6,8,10",
                "expected_type": AdvancedProblemType.DATA_VISUALIZATION
            },
            {
                "question": "Проанализировать график в файле graph.png",
                "expected_type": AdvancedProblemType.GRAPH_ANALYSIS
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                result = self.process_advanced_question(test_case["question"])
                results.append({
                    "question": test_case["question"],
                    "success": True,
                    "category": result.get("category", "unknown"),
                    "confidence": result.get("confidence", 0.0)
                })
            except Exception as e:
                results.append({
                    "question": test_case["question"],
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "integration_status": self.integration_status,
            "test_results": results,
            "supported_types": self.get_supported_problem_types()
        }

# Интеграция с основной системой Rubin AI
def integrate_with_rubin_ai():
    """Интеграция с основной системой Rubin AI"""
    
    # Импорт основной системы
    try:
        from neural_rubin import NeuralRubinAI
        
        # Расширение класса NeuralRubinAI
        class EnhancedNeuralRubinAI(NeuralRubinAI):
            """Расширенная версия Rubin AI с математическими возможностями"""
            
            def __init__(self):
                super().__init__()
                self.advanced_math_integration = RubinAdvancedMathIntegration()
                logger.info("✅ Расширенный математический решатель интегрирован")
            
            def _solve_advanced_math_neural(self, question: str) -> str:
                """Решение расширенных математических задач"""
                try:
                    result = self.advanced_math_integration.process_advanced_question(question)
                    return result["response"]
                except Exception as e:
                    logger.error(f"Ошибка расширенного математического решения: {e}")
                    return f"Ошибка расширенного математического решения: {e}"
            
            def generate_response(self, question: str) -> Dict[str, Any]:
                """Расширенная генерация ответов"""
                try:
                    # Проверка на расширенные математические задачи
                    if self._is_advanced_math_question(question):
                        result = self.advanced_math_integration.process_advanced_question(question)
                        return {
                            "response": result["response"],
                            "category": result["category"],
                            "confidence": result["confidence"],
                            "type": "advanced_math",
                            "neural_network": True
                        }
                    else:
                        # Обычная обработка
                        return super().generate_response(question)
                        
                except Exception as e:
                    logger.error(f"Ошибка генерации ответа: {e}")
                    return {
                        "response": f"Ошибка обработки вопроса: {e}",
                        "category": "error",
                        "confidence": 0.0,
                        "type": "error"
                    }
            
            def _is_advanced_math_question(self, question: str) -> bool:
                """Проверка на расширенные математические задачи"""
                advanced_keywords = [
                    "кинетическая энергия", "потенциальная энергия", "закон ома",
                    "мощность", "сила тяжести", "ускорение", "путь",
                    "график", "диаграмма", "анализ изображения",
                    "построить", "создать график", "визуализация"
                ]
                
                question_lower = question.lower()
                return any(keyword in question_lower for keyword in advanced_keywords)
        
        return EnhancedNeuralRubinAI
        
    except ImportError as e:
        logger.warning(f"Не удалось импортировать основную систему Rubin AI: {e}")
        return None

# Пример использования
if __name__ == "__main__":
    # Тестирование интеграции
    integration = RubinAdvancedMathIntegration()
    
    print("🧮 ТЕСТИРОВАНИЕ РАСШИРЕННОГО МАТЕМАТИЧЕСКОГО РЕШАТЕЛЯ")
    print("=" * 60)
    
    # Тест 1: Физические формулы
    print("\n⚡ Тест 1: Физические формулы")
    physics_question = "Найти кинетическую энергию тела массой 2 кг, движущегося со скоростью 10 м/с"
    result1 = integration.process_advanced_question(physics_question)
    print(f"Вопрос: {physics_question}")
    print(f"Ответ: {result1['response']}")
    print(f"Категория: {result1['category']}")
    print(f"Уверенность: {result1['confidence']:.1%}")
    
    # Тест 2: Визуализация данных
    print("\n📈 Тест 2: Визуализация данных")
    viz_question = "Построить линейный график A: 1,2,3,4,5 B: 2,4,6,8,10"
    result2 = integration.process_advanced_question(viz_question)
    print(f"Вопрос: {viz_question}")
    print(f"Ответ: {result2['response']}")
    print(f"Категория: {result2['category']}")
    print(f"Уверенность: {result2['confidence']:.1%}")
    
    # Тест 3: Анализ графика
    print("\n📊 Тест 3: Анализ графика")
    graph_question = "Проанализировать график в файле test_graph.png"
    result3 = integration.process_advanced_question(graph_question)
    print(f"Вопрос: {graph_question}")
    print(f"Ответ: {result3['response']}")
    print(f"Категория: {result3['category']}")
    print(f"Уверенность: {result3['confidence']:.1%}")
    
    # Общий тест интеграции
    print("\n🔧 Общий тест интеграции")
    test_results = integration.test_integration()
    print(f"Статус интеграции: {test_results['integration_status']}")
    print(f"Поддерживаемые типы: {list(test_results['supported_types'].keys())}")
    
    print("\n✅ Тестирование завершено!")










