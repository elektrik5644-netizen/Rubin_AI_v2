#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Response Formatter
==============================

Форматирует ответы математического решателя в читаемый вид.
"""

import logging
from typing import Dict, Any
from datetime import datetime

class MathematicalResponseFormatter:
    """Форматтер математических ответов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_structured_response(self, solution: Dict[str, Any], 
                                 category: str, processing_time: float) -> Dict[str, Any]:
        """Создает структурированный ответ"""
        
        # Форматируем основной ответ
        formatted_response = self.format_solution(solution, category)
        
        # Создаем структурированный ответ
        response = {
            "response": formatted_response,
            "provider": "Mathematical Solver",
            "category": f"mathematics_{category}",
            "solution_data": {
                "problem_type": category,
                "final_answer": solution.get("final_answer", "Ответ не найден"),
                "confidence": solution.get("confidence", 0.0),
                "steps": solution.get("steps", []),
                "explanation": solution.get("explanation", ""),
                "verification": solution.get("verification", False)
            },
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        self.logger.info(f"Создан структурированный ответ для {category}")
        return response
    
    def format_solution(self, solution: Dict[str, Any], category: str) -> str:
        """Форматирует решение в читаемый вид"""
        
        # Определяем эмодзи для категории
        category_emoji = {
            "geometry": "📐",
            "physics": "⚡",
            "arithmetic": "🔢"
        }
        
        emoji = category_emoji.get(category, "🧮")
        category_name = {
            "geometry": "Геометрическая задача",
            "physics": "Физическая задача", 
            "arithmetic": "Арифметическая задача"
        }.get(category, "Математическая задача")
        
        # Начинаем форматирование
        formatted = f"{emoji} **{category_name}**\n\n"
        
        # Добавляем пошаговое решение
        steps = solution.get("steps", [])
        if steps:
            formatted += "**📋 Пошаговое решение:**\n"
            for i, step in enumerate(steps, 1):
                formatted += f"{i}. {step}\n"
            formatted += "\n"
        
        # Добавляем финальный ответ
        final_answer = solution.get("final_answer", "Ответ не найден")
        formatted += f"**✅ Ответ: {final_answer}**\n\n"
        
        # Добавляем объяснение
        explanation = solution.get("explanation", "")
        if explanation:
            formatted += f"**💡 Объяснение:** {explanation}\n\n"
        
        # Добавляем уровень уверенности
        confidence = solution.get("confidence", 0.0)
        if confidence > 0:
            confidence_percent = confidence * 100
            formatted += f"**📊 Уверенность:** {confidence_percent:.0f}%\n"
        
        # Добавляем статус проверки
        verification = solution.get("verification", False)
        if verification:
            formatted += "**✓ Решение проверено**"
        
        return formatted
    
    def format_error_response(self, error_message: str, category: str = None) -> Dict[str, Any]:
        """Форматирует ответ об ошибке"""
        
        response = {
            "response": f"❌ **Ошибка решения математической задачи**\n\n{error_message}",
            "provider": "Mathematical Solver",
            "category": f"mathematics_{category}" if category else "mathematics_error",
            "success": False,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def format_simple_response(self, answer: str, category: str) -> str:
        """Форматирует простой ответ без деталей"""
        
        category_emoji = {
            "geometry": "📐",
            "physics": "⚡", 
            "arithmetic": "🔢"
        }
        
        emoji = category_emoji.get(category, "🧮")
        return f"{emoji} **Ответ:** {answer}"

















