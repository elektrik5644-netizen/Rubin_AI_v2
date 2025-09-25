#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 Интеграция математического решателя с Rubin AI
================================================

Модуль для интеграции математического решателя с основной системой Rubin AI.
Автоматически распознает математические запросы и перенаправляет их в математический модуль.

Автор: Rubin AI System
Версия: 2.0
"""

import re
import logging
from typing import Dict, Any, Optional
from mathematical_problem_solver import MathematicalProblemSolver

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubinMathIntegration:
    """Интеграция математического решателя с Rubin AI"""
    
    def __init__(self):
        """Инициализация интеграции"""
        self.solver = MathematicalProblemSolver()
        self.math_patterns = self._initialize_math_patterns()
        
    def _initialize_math_patterns(self) -> list:
        """Инициализация паттернов для распознавания математических запросов"""
        return [
            # Арифметические операции
            r'\d+\s*[\+\-\*/]\s*\d+',  # 2+2, 3*4, 10-5
            r'вычисли|посчитай|найди\s+значение|результат',
            r'сколько\s+будет|чему\s+равно',
            
            # Уравнения
            r'реши\s+уравнение|найди\s+x|найди\s+корень',
            r'[a-zA-Z]\s*[\+\-]\s*\d+\s*=\s*\d+',  # x+5=10
            r'[a-zA-Z]\^?2\s*[\+\-]',  # x²+5
            
            # Геометрия
            r'площадь|объем|периметр|найди\s+площадь',
            r'треугольник|круг|прямоугольник|квадрат',
            
            # Тригонометрия
            r'sin|cos|tan|синус|косинус|тангенс',
            r'градус|радиан|угол',
            
            # Статистика
            r'среднее|медиана|мода|дисперсия',
            r'статистика|вероятность',
            
            # Проценты
            r'процент|%|\d+\s*процентов',
            r'найти\s+\d+%\s+от',
            
            # Математические символы
            r'[\+\-\*/=]',  # +, -, *, /, =
            r'\d+\.?\d*',  # числа
        ]
    
    def is_math_query(self, query: str) -> bool:
        """Проверка, является ли запрос математическим"""
        query_lower = query.lower()
        
        # Проверка на математические паттерны
        for pattern in self.math_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Дополнительные проверки
        if any(word in query_lower for word in ['математика', 'математический', 'вычислить', 'решить']):
            return True
        
        # Проверка на простые арифметические выражения
        if re.match(r'^\d+\s*[\+\-\*/]\s*\d+\s*[=]?\s*$', query.strip()):
            return True
        
        return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Обработка запроса через математический решатель"""
        try:
            logger.info(f"Обработка математического запроса: {query}")
            
            # Решение задачи
            solution = self.solver.solve_problem(query)
            
            # Формирование ответа в стиле Rubin AI
            response = {
                "success": True,
                "query": query,
                "answer": solution.final_answer,
                "problem_type": solution.problem_type.value,
                "confidence": solution.confidence,
                "explanation": solution.explanation,
                "steps": solution.solution_steps,
                "verification": solution.verification,
                "module": "Mathematical Problem Solver",
                "processing_time": getattr(solution, 'processing_time', 0)
            }
            
            logger.info(f"Математический запрос решен: {solution.final_answer}")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при решении математической задачи: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "module": "Mathematical Problem Solver"
            }
    
    def get_enhanced_response(self, query: str) -> str:
        """Получение улучшенного ответа в стиле Rubin AI"""
        result = self.process_query(query)
        
        if not result["success"]:
            return f"❌ Ошибка при решении математической задачи: {result['error']}"
        
        # Формирование красивого ответа
        response_parts = []
        
        # Основной ответ
        response_parts.append(f"🧮 **Математическое решение:**")
        response_parts.append(f"**Вопрос:** {query}")
        response_parts.append(f"**Ответ:** {result['answer']}")
        
        # Дополнительная информация
        response_parts.append(f"**Тип задачи:** {result['problem_type']}")
        response_parts.append(f"**Уверенность:** {result['confidence']:.1%}")
        
        if result['verification']:
            response_parts.append("✅ **Проверка:** Решение верифицировано")
        
        # Пошаговое решение
        if result['steps']:
            response_parts.append("\n📋 **Пошаговое решение:**")
            for i, step in enumerate(result['steps'], 1):
                response_parts.append(f"{i}. {step}")
        
        # Объяснение
        if result['explanation']:
            response_parts.append(f"\n💡 **Объяснение:** {result['explanation']}")
        
        return "\n".join(response_parts)

def demo_integration():
    """Демонстрация интеграции"""
    print("🔗 ДЕМОНСТРАЦИЯ ИНТЕГРАЦИИ С RUBIN AI")
    print("=" * 60)
    
    integration = RubinMathIntegration()
    
    # Тестовые запросы
    test_queries = [
        "2+2=?",
        "3+4=?",
        "Реши уравнение 2x + 5 = 13",
        "Найди площадь треугольника с основанием 5 и высотой 3",
        "Найди 15% от 200",
        "Вычисли sin(30°)",
        "Найди среднее значение чисел 1, 2, 3, 4, 5"
    ]
    
    for query in test_queries:
        print(f"\n📝 Запрос: {query}")
        
        # Проверка, является ли запрос математическим
        is_math = integration.is_math_query(query)
        print(f"🔍 Распознан как математический: {'✅ Да' if is_math else '❌ Нет'}")
        
        if is_math:
            # Обработка через математический решатель
            response = integration.get_enhanced_response(query)
            print(f"🧮 Ответ математического решателя:")
            print(response)
        else:
            print("📡 Перенаправление к другому модулю Rubin AI")
        
        print("-" * 50)

def simulate_rubin_ai_with_math():
    """Симуляция работы Rubin AI с математическим модулем"""
    print("\n🤖 СИМУЛЯЦИЯ RUBIN AI С МАТЕМАТИЧЕСКИМ МОДУЛЕМ")
    print("=" * 60)
    
    integration = RubinMathIntegration()
    
    # Симуляция запросов пользователя
    user_queries = [
        "2+2=?",
        "Как работает система Rubin AI?",
        "Реши уравнение 3x - 7 = 8",
        "Что такое машинное обучение?",
        "Найди 20% от 150"
    ]
    
    for query in user_queries:
        print(f"\n👤 Пользователь: {query}")
        
        if integration.is_math_query(query):
            print("🔍 Анализирую вопрос: математический запрос")
            print("📡 Направляю к модулю: Математический решатель")
            response = integration.get_enhanced_response(query)
            print(f"🧮 Математический решатель: {response}")
        else:
            print("🔍 Анализирую вопрос: общий запрос")
            print("📡 Направляю к модулю: AI Чат")
            print("🤖 AI Чат: Обрабатываю ваш запрос...")

if __name__ == "__main__":
    demo_integration()
    simulate_rubin_ai_with_math()

