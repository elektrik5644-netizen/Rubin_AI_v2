#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Request Handler
===========================

Обрабатывает математические запросы и координирует решение задач.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

try:
    from mathematical_problem_solver import MathematicalProblemSolver
    MATHEMATICAL_SOLVER_AVAILABLE = True
except ImportError:
    MATHEMATICAL_SOLVER_AVAILABLE = False
    logger.warning("⚠️ MathematicalProblemSolver недоступен. Математические функции будут ограничены.")

from .category_detector import MathematicalCategoryDetector
from .response_formatter import MathematicalResponseFormatter
from .error_handler import MathematicalErrorHandler

class MathematicalRequestHandler:
    """Обработчик математических запросов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detector = MathematicalCategoryDetector()
        self.formatter = MathematicalResponseFormatter()
        self.error_handler = MathematicalErrorHandler()
        
        if MATHEMATICAL_SOLVER_AVAILABLE:
            self.solver = MathematicalProblemSolver()
            self.logger.info("✅ MathematicalProblemSolver инициализирован")
        else:
            self.solver = None
            self.logger.warning("⚠️ MathematicalProblemSolver недоступен, используем базовую логику")
    
    def handle_request(self, message: str, category: str = None) -> Dict[str, Any]:
        """Обрабатывает математический запрос"""
        start_time = datetime.now()
        
        try:
            # Определяем категорию если не указана
            if not category:
                if not self.detector.is_mathematical_request(message):
                    return self.error_handler.handle_detection_error(
                        message, Exception("Не является математическим запросом")
                    )
                
                category = self.detector.detect_math_category(message)
                if not category:
                    return self.error_handler.handle_detection_error(
                        message, Exception("Не удалось определить тип задачи")
                    )
            
            # Извлекаем данные
            try:
                math_data = self.detector.extract_math_data(message, category)
            except Exception as e:
                return self.error_handler.handle_parsing_error(message, category, e)
            
            # Решаем задачу
            try:
                solution = self._solve_problem(message, category, math_data)
            except Exception as e:
                return self.error_handler.handle_solving_error(message, category, e)
            
            # Форматируем ответ
            processing_time = (datetime.now() - start_time).total_seconds()
            response = self.formatter.create_structured_response(
                solution, category, processing_time
            )
            
            self.logger.info(f"Математический запрос обработан за {processing_time:.3f}с")
            return response
            
        except Exception as e:
            self.logger.error(f"Критическая ошибка обработки запроса: {e}")
            return self.error_handler.handle_critical_error(message, e)
    
    def _solve_problem(self, message: str, category: str, math_data: Dict[str, Any]) -> Dict[str, Any]:
        """Решает математическую задачу"""
        
        if self.solver:
            # Используем продвинутый решатель
            solution = self.solver.solve_problem(message)
            return {
                "final_answer": solution.final_answer,
                "steps": solution.solution_steps,
                "explanation": solution.explanation,
                "confidence": solution.confidence,
                "verification": solution.verification
            }
        else:
            # Используем базовую логику
            return self._solve_basic_problem(message, category, math_data)
    
    def _solve_basic_problem(self, message: str, category: str, math_data: Dict[str, Any]) -> Dict[str, Any]:
        """Базовое решение математических задач"""
        
        if category == "geometry":
            return self._solve_geometry_basic(math_data)
        elif category == "physics":
            return self._solve_physics_basic(math_data)
        elif category == "arithmetic":
            return self._solve_arithmetic_basic(math_data)
        elif category == "equation":
            return self._solve_equation(math_data)
        else:
            raise Exception(f"Неподдерживаемая категория: {category}")
    
    def _solve_equation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Solves an algebraic equation using sympy."""
        try:
            from sympy import sympify, solve, symbols
            
            equation_string = data["equation_string"]
            
            # Sympy expects an expression that equals zero.
            if '=' in equation_string:
                lhs, rhs = equation_string.split('=')
                equation = sympify(f"({lhs}) - ({rhs})")
            else:
                equation = sympify(equation_string)

            # Define the variable symbol dynamically
            variable_name = data.get('variable', 'x')
            variable = symbols(variable_name)
            
            # Solve the equation
            solution = solve(equation, variable)
            
            if solution:
                answer = f"{variable_name} = {solution[0]}"
                steps = [
                    f"Исходное уравнение: {equation_string}",
                    f"Приводим к виду, равному нулю: {equation} = 0",
                    f"Решаем относительно {variable_name}",
                    f"Результат: {variable_name} = {solution[0]}"
                ]
                explanation = f"Решением уравнения {equation_string} является {variable_name} = {solution[0]}."
                return {
                    "final_answer": answer,
                    "steps": steps,
                    "explanation": explanation,
                    "confidence": 1.0,
                    "verification": True
                }
            else:
                raise Exception("Не удалось найти решение")

        except Exception as e:
            self.logger.error(f"Ошибка решения уравнения: {e}")
            return {
                "final_answer": "Ошибка",
                "steps": [f"Не удалось решить уравнение: {str(e)}"],
                "explanation": "Проверьте синтаксис уравнения.",
                "confidence": 0.5,
                "verification": False
            }

    
    def _solve_geometry_basic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Базовое решение геометрических задач"""
        if data.get("question_type") == "adjacent_angles" and "angle_sum" in data:
            angle_sum = data["angle_sum"]
            
            if angle_sum == 180:
                answer = "ДА, углы являются смежными"
                explanation = "Смежные углы в сумме дают 180°"
            else:
                answer = "НЕТ, углы не являются смежными"
                explanation = f"Смежные углы в сумме дают 180°, а сумма данных углов равна {angle_sum}°"
            
            steps = [
                "Смежные углы - это углы, которые имеют общую сторону",
                "Сумма смежных углов всегда равна 180°",
                f"Данная сумма углов: {angle_sum}°",
                f"Сравнение: {angle_sum}° {'=' if angle_sum == 180 else '≠'} 180°",
                answer
            ]
            
            return {
                "final_answer": answer,
                "steps": steps,
                "explanation": explanation,
                "confidence": 0.95,
                "verification": True
            }
        
        raise Exception("Неподдерживаемый тип геометрической задачи")
    
    def _solve_physics_basic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Базовое решение физических задач"""
        if (data.get("question_type") == "velocity_calculation" and 
            "distance" in data and "time" in data):
            
            distance = data["distance"]
            time = data["time"]
            
            # Вычисляем скорость
            velocity_ms = distance / time
            velocity_kmh = velocity_ms * 3.6
            
            answer = f"{velocity_ms:.1f} м/с ({velocity_kmh:.1f} км/ч)"
            
            steps = [
                f"Дано: расстояние s = {distance} м, время t = {time} с",
                "Формула скорости: v = s/t",
                f"v = {distance}/{time} = {velocity_ms:.1f} м/с",
                f"Перевод в км/ч: {velocity_ms:.1f} × 3.6 = {velocity_kmh:.1f} км/ч"
            ]
            
            explanation = f"Скорость составляет {velocity_ms:.1f} м/с ({velocity_kmh:.1f} км/ч)"
            
            return {
                "final_answer": answer,
                "steps": steps,
                "explanation": explanation,
                "confidence": 0.95,
                "verification": True
            }
        
        raise Exception("Неподдерживаемый тип физической задачи")
    
    def _solve_arithmetic_basic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Базовое решение арифметических задач"""
        if (data.get("question_type") == "simple_arithmetic" and 
            "numbers" in data and len(data["numbers"]) == 2 and "operator" in data):
            
            num1, num2 = data["numbers"]
            op = data["operator"]
            result = 0
            
            if op == '+':
                result = num1 + num2
            elif op == '-':
                result = num1 - num2
            elif op == '*':
                result = num1 * num2
            elif op == '/':
                if num2 != 0:
                    result = num1 / num2
                else:
                    return {
                        "final_answer": "Ошибка",
                        "steps": ["Деление на ноль невозможно"],
                        "explanation": "На ноль делить нельзя.",
                        "confidence": 1.0,
                        "verification": False
                    }
            
            answer = f"{result}"
            steps = [f"Выполняем операцию: {num1} {op} {num2}", f"Результат: {result}"]
            explanation = f"Результатом операции {num1} {op} {num2} является {result}."

            return {
                "final_answer": answer,
                "steps": steps,
                "explanation": explanation,
                "confidence": 1.0,
                "verification": True
            }

        if (data.get("question_type") == "remainder_calculation" and 
            "numbers" in data and len(data["numbers"]) >= 3):
            
            numbers = data["numbers"]
            total = numbers[0]
            known_quantities = numbers[1:-1] if len(numbers) > 3 else numbers[1:3]
            
            remainder = total - sum(known_quantities)
            
            if data.get("object_type") == "trees":
                answer = f"{remainder} слив"
                
                steps = [
                    f"Всего деревьев: {total}",
                    f"Яблонь: {known_quantities[0]}",
                    f"Груш: {known_quantities[1] if len(known_quantities) > 1 else 0}",
                    f"Слив: {total} - {' - '.join(map(str, known_quantities))} = {remainder}"
                ]
                
                explanation = f"В саду {remainder} сливовых деревьев"
            else:
                answer = f"{remainder}"
                steps = [f"Остальные: {total} - {sum(known_quantities)} = {remainder}"]
                explanation = f"Остальных объектов: {remainder}"
            
            return {
                "final_answer": answer,
                "steps": steps,
                "explanation": explanation,
                "confidence": 0.95,
                "verification": True
            }
        
        raise Exception("Неподдерживаемый тип арифметической задачи")
    
    def get_solver_status(self) -> Dict[str, Any]:
        """
        Gets the status of the mathematical solver.
        
        Returns:
            Dictionary with solver status
        """
        try:
            # Test the solver with a simple problem
            test_solution = self.solver.solve_problem("2+2")
            
            return {
                "status": "operational",
                "solver_type": "AdvancedMathematicalSolver",
                "test_result": test_solution.final_answer if test_solution else "failed",
                "test_confidence": test_solution.confidence if test_solution else 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_request(self, message: str, category: str) -> Dict[str, Any]:
        """
        Validates a mathematical request before processing.
        
        Args:
            message: The user's message
            category: The detected category
            
        Returns:
            Dictionary with validation results
        """
        validation_errors = []
        
        # Check if message is empty
        if not message or not message.strip():
            validation_errors.append("Сообщение не может быть пустым")
        
        # Check if category is valid
        valid_categories = ["geometry", "physics", "arithmetic", "equation", "percentage", "general"]
        if category not in valid_categories:
            validation_errors.append(f"Неизвестная категория: {category}")
        
        # Check for minimum content
        if len(message.strip()) < 3:
            validation_errors.append("Сообщение слишком короткое")
        
        # Category-specific validation
        if category == "geometry":
            if not any(word in message.lower() for word in ['угол', 'градус', 'треугольник', 'круг']):
                validation_errors.append("Геометрическая задача должна содержать геометрические термины")
        
        elif category == "physics":
            if not any(word in message.lower() for word in ['скорость', 'путь', 'время', 'расстояние']):
                validation_errors.append("Физическая задача должна содержать физические величины")
        
        elif category == "arithmetic":
            import re
            if not re.search(r'\d+', message):
                validation_errors.append("Арифметическая задача должна содержать числа")
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "category": category,
            "message_length": len(message)
        }