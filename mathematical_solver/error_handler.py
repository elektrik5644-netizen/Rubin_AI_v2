#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Error Handler
==========================

Обрабатывает ошибки математического решателя.
"""

import logging
from typing import Dict, Any
from datetime import datetime

class MathematicalErrorHandler:
    """Обработчик ошибок математического решателя"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_detection_error(self, message: str, error: Exception) -> Dict[str, Any]:
        """Обработка ошибок определения типа задачи"""
        
        self.logger.warning(f"Ошибка определения типа задачи: {error}")
        
        return {
            "response": """❌ **Не удалось определить тип математической задачи**

**Возможные причины:**
• Вопрос сформулирован нечетко
• Отсутствуют ключевые математические термины
• Недостаточно данных для решения

**Попробуйте переформулировать вопрос, включив:**
• Конкретные числовые значения
• Единицы измерения (градусы, метры, секунды)
• Четкую постановку вопроса

**Примеры правильных вопросов:**
• "Сумма углов АВС и АВО равна 160°. Являются ли они смежными?"
• "Какую скорость развивает объект, преодолевая 100 м за 10 с?"
• "В корзине 15 яблок. 5 красных, 3 зеленых, остальные желтые. Сколько желтых?"
""",
            "provider": "Mathematical Solver",
            "category": "mathematics_error",
            "error_type": "detection_error",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_parsing_error(self, message: str, category: str, error: Exception) -> Dict[str, Any]:
        """Обработка ошибок извлечения данных"""
        
        self.logger.warning(f"Ошибка извлечения данных для {category}: {error}")
        
        category_tips = {
            "geometry": [
                "Укажите точные значения углов в градусах (например, 160°)",
                "Назовите углы буквами (например, углы АВС и АВО)",
                "Четко сформулируйте вопрос (смежные ли углы?)"
            ],
            "physics": [
                "Укажите расстояние с единицами измерения (например, 332 м)",
                "Укажите время с единицами измерения (например, 40 с)",
                "Четко сформулируйте, что нужно найти (скорость, время, расстояние)"
            ],
            "arithmetic": [
                "Укажите общее количество объектов",
                "Перечислите известные количества",
                "Четко сформулируйте, что нужно найти"
            ]
        }
        
        tips = category_tips.get(category, ["Проверьте правильность формулировки задачи"])
        tips_text = "\n".join(f"• {tip}" for tip in tips)
        
        return {
            "response": f"""❌ **Ошибка обработки данных для {category} задачи**

**Не удалось извлечь необходимые данные из вашего вопроса.**

**Рекомендации для {category} задач:**
{tips_text}

**Ваш вопрос:** "{message}"

**Попробуйте переформулировать вопрос более четко.**
""",
            "provider": "Mathematical Solver",
            "category": f"mathematics_{category}",
            "error_type": "parsing_error",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_solving_error(self, message: str, category: str, error: Exception) -> Dict[str, Any]:
        """Обработка ошибок решения"""
        
        self.logger.error(f"Ошибка решения {category} задачи: {error}")
        
        return {
            "response": f"""❌ **Ошибка при решении {category} задачи**

**Произошла ошибка в процессе вычислений.**

**Возможные причины:**
• Некорректные входные данные
• Математически невозможная задача
• Техническая ошибка в алгоритме решения

**Что можно сделать:**
• Проверьте правильность числовых значений
• Убедитесь в логичности условий задачи
• Попробуйте переформулировать вопрос

**Если проблема повторяется, обратитесь к техническому специалисту.**
""",
            "provider": "Mathematical Solver",
            "category": f"mathematics_{category}",
            "error_type": "solving_error",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_critical_error(self, message: str, error: Exception) -> Dict[str, Any]:
        """Обработка критических ошибок"""
        
        self.logger.error(f"Критическая ошибка математического решателя: {error}")
        
        return {
            "response": """❌ **Критическая ошибка математического решателя**

**Произошла неожиданная ошибка в системе.**

**Временное решение:**
• Попробуйте переформулировать вопрос
• Обратитесь к общему AI помощнику
• Повторите попытку через некоторое время

**Ошибка зарегистрирована для анализа техническими специалистами.**
""",
            "provider": "Mathematical Solver",
            "category": "mathematics_critical_error",
            "error_type": "critical_error",
            "success": False,
            "fallback_available": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def create_fallback_response(self, message: str) -> Dict[str, Any]:
        """Создает fallback ответ для направления к общему AI"""
        
        return {
            "response": """🔄 **Переадресация к общему AI помощнику**

Математический решатель временно недоступен. 
Ваш вопрос будет обработан общим AI помощником.

Это может занять немного больше времени, но вы получите ответ.
""",
            "provider": "Mathematical Solver (Fallback)",
            "category": "mathematics_fallback",
            "fallback": True,
            "original_message": message,
            "timestamp": datetime.now().isoformat()
        }
    
    def handle_fallback_error(self, message: str, original_error: Exception) -> Dict[str, Any]:
        """
        Handles fallback to general AI chat when mathematical solver fails.
        
        Args:
            message: The original user message
            original_error: The original error that caused the fallback
            
        Returns:
            Dictionary with fallback response
        """
        error_msg = "Математический решатель временно недоступен. Перенаправляю к общему AI чату."
        
        # Log the fallback
        self.logger.error(f"Fallback triggered for message '{message}': {str(original_error)}")
        
        return {
            "response": f"🔄 {error_msg}",
            "provider": "Mathematical Solver (Fallback)",
            "category": "mathematics_fallback",
            "error_type": "fallback_triggered",
            "solution_data": {
                "problem_type": "fallback",
                "final_answer": None,
                "confidence": 0.0,
                "explanation": error_msg,
                "steps": [],
                "formulas_used": [],
                "input_data": {}
            },
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.0,
            "success": False,
            "error_message": str(original_error),
            "suggestion": "Обратитесь к общему AI чату для получения помощи",
            "fallback": True
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Gets error statistics for monitoring.
        
        Returns:
            Dictionary with error statistics
        """
        # This would typically query a database or monitoring system
        return {
            "total_errors": 0,  # Placeholder
            "errors_by_type": {},  # Placeholder
            "last_error": None,  # Placeholder
            "error_rate": 0.0  # Placeholder
        }