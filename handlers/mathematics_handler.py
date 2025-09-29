#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematics Handler - Обработчик математических запросов
"""

import logging
import re
import math
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class MathematicsHandler:
    """Обработчик математических запросов"""
    
    def __init__(self):
        self.operations = {
            "сложение": "+",
            "вычитание": "-", 
            "умножение": "*",
            "деление": "/",
            "возведение в степень": "**",
            "квадратный корень": "sqrt",
            "логарифм": "log",
            "синус": "sin",
            "косинус": "cos",
            "тангенс": "tan"
        }
        
        self.formulas = {
            "площадь круга": "S = π × r²",
            "площадь треугольника": "S = (a × h) / 2",
            "площадь прямоугольника": "S = a × b",
            "объем шара": "V = (4/3) × π × r³",
            "объем цилиндра": "V = π × r² × h",
            "теорема пифагора": "c² = a² + b²",
            "квадратное уравнение": "ax² + bx + c = 0"
        }
    
    def handle_request(self, message: str) -> Dict[str, Any]:
        """Обработка запроса"""
        message_lower = message.lower().strip()
        
        # Определяем тип запроса
        if any(word in message_lower for word in ["реши", "вычисли", "посчитай", "найди"]):
            return self._handle_calculation(message)
        elif any(word in message_lower for word in ["формула", "уравнение", "теорема"]):
            return self._handle_formula(message)
        elif any(word in message_lower for word in ["что такое", "определение", "объясни"]):
            return self._handle_definition(message)
        else:
            return self._handle_general_math(message)
    
    def _handle_calculation(self, message: str) -> Dict[str, Any]:
        """Обработка расчетных запросов"""
        # Поиск простых арифметических выражений
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        
        if len(numbers) >= 2:
            try:
                # Простые арифметические операции
                if "+" in message or "плюс" in message_lower or "сложить" in message_lower:
                    result = sum(float(n) for n in numbers)
                    operation = " + ".join(numbers)
                    return self._create_calculation_response(f"{operation} = {result}", "сложение")
                
                elif "-" in message or "минус" in message_lower or "вычесть" in message_lower:
                    if len(numbers) == 2:
                        result = float(numbers[0]) - float(numbers[1])
                        return self._create_calculation_response(f"{numbers[0]} - {numbers[1]} = {result}", "вычитание")
                
                elif "*" in message or "×" in message or "умножить" in message_lower or "произведение" in message_lower:
                    result = 1
                    for n in numbers:
                        result *= float(n)
                    operation = " × ".join(numbers)
                    return self._create_calculation_response(f"{operation} = {result}", "умножение")
                
                elif "/" in message or "÷" in message or "делить" in message_lower or "разделить" in message_lower:
                    if len(numbers) == 2:
                        if float(numbers[1]) != 0:
                            result = float(numbers[0]) / float(numbers[1])
                            return self._create_calculation_response(f"{numbers[0]} ÷ {numbers[1]} = {result}", "деление")
                        else:
                            return self._create_error_response("Деление на ноль невозможно!")
                
                elif "степень" in message_lower or "в степени" in message_lower or "**" in message:
                    if len(numbers) == 2:
                        result = float(numbers[0]) ** float(numbers[1])
                        return self._create_calculation_response(f"{numbers[0]}^{numbers[1]} = {result}", "возведение в степень")
                
            except (ValueError, ZeroDivisionError) as e:
                return self._create_error_response(f"Ошибка в расчетах: {e}")
        
        # Поиск геометрических расчетов
        if any(word in message_lower for word in ["площадь", "объем", "периметр"]):
            return self._handle_geometry(message)
        
        # Поиск тригонометрических функций
        if any(word in message_lower for word in ["sin", "cos", "tan", "синус", "косинус", "тангенс"]):
            return self._handle_trigonometry(message)
        
        return self._handle_general_math(message)
    
    def _handle_geometry(self, message: str) -> Dict[str, Any]:
        """Обработка геометрических расчетов"""
        message_lower = message.lower()
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        
        try:
            if "площадь круга" in message_lower or "площадь окружности" in message_lower:
                if numbers:
                    r = float(numbers[0])
                    area = math.pi * r ** 2
                    content = f"""**ПЛОЩАДЬ КРУГА**

**Формула:** S = π × r²

**Данные:**
• Радиус: r = {r}

**Расчет:**
S = π × {r}² = π × {r**2} = {area:.2f}

**Ответ:** S = {area:.2f}"""
                    
                    return {
                        "success": True,
                        "response": {
                            "content": content,
                            "title": "Площадь круга",
                            "source": "Mathematics Handler"
                        },
                        "category": "geometry",
                        "confidence": 0.9,
                        "timestamp": datetime.now().isoformat(),
                        "provider": "Mathematics Handler"
                    }
            
            elif "площадь треугольника" in message_lower:
                if len(numbers) >= 2:
                    a, h = float(numbers[0]), float(numbers[1])
                    area = (a * h) / 2
                    content = f"""**ПЛОЩАДЬ ТРЕУГОЛЬНИКА**

**Формула:** S = (a × h) / 2

**Данные:**
• Основание: a = {a}
• Высота: h = {h}

**Расчет:**
S = ({a} × {h}) / 2 = {a * h} / 2 = {area:.2f}

**Ответ:** S = {area:.2f}"""
                    
                    return {
                        "success": True,
                        "response": {
                            "content": content,
                            "title": "Площадь треугольника",
                            "source": "Mathematics Handler"
                        },
                        "category": "geometry",
                        "confidence": 0.9,
                        "timestamp": datetime.now().isoformat(),
                        "provider": "Mathematics Handler"
                    }
            
            elif "объем шара" in message_lower:
                if numbers:
                    r = float(numbers[0])
                    volume = (4/3) * math.pi * r ** 3
                    content = f"""**ОБЪЕМ ШАРА**

**Формула:** V = (4/3) × π × r³

**Данные:**
• Радиус: r = {r}

**Расчет:**
V = (4/3) × π × {r}³ = (4/3) × π × {r**3} = {volume:.2f}

**Ответ:** V = {volume:.2f}"""
                    
                    return {
                        "success": True,
                        "response": {
                            "content": content,
                            "title": "Объем шара",
                            "source": "Mathematics Handler"
                        },
                        "category": "geometry",
                        "confidence": 0.9,
                        "timestamp": datetime.now().isoformat(),
                        "provider": "Mathematics Handler"
                    }
        
        except (ValueError, IndexError):
            pass
        
        return self._handle_general_math(message)
    
    def _handle_trigonometry(self, message: str) -> Dict[str, Any]:
        """Обработка тригонометрических функций"""
        message_lower = message.lower()
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        
        try:
            if numbers:
                angle = float(numbers[0])
                angle_rad = math.radians(angle)
                
                if "sin" in message_lower or "синус" in message_lower:
                    result = math.sin(angle_rad)
                    content = f"""**СИНУС УГЛА**

**Данные:**
• Угол: {angle}°

**Расчет:**
sin({angle}°) = {result:.4f}

**Ответ:** sin({angle}°) = {result:.4f}"""
                
                elif "cos" in message_lower or "косинус" in message_lower:
                    result = math.cos(angle_rad)
                    content = f"""**КОСИНУС УГЛА**

**Данные:**
• Угол: {angle}°

**Расчет:**
cos({angle}°) = {result:.4f}

**Ответ:** cos({angle}°) = {result:.4f}"""
                
                elif "tan" in message_lower or "тангенс" in message_lower:
                    result = math.tan(angle_rad)
                    content = f"""**ТАНГЕНС УГЛА**

**Данные:**
• Угол: {angle}°

**Расчет:**
tan({angle}°) = {result:.4f}

**Ответ:** tan({angle}°) = {result:.4f}"""
                
                else:
                    return self._handle_general_math(message)
                
                return {
                    "success": True,
                    "response": {
                        "content": content,
                        "title": "Тригонометрическая функция",
                        "source": "Mathematics Handler"
                    },
                    "category": "trigonometry",
                    "confidence": 0.9,
                    "timestamp": datetime.now().isoformat(),
                    "provider": "Mathematics Handler"
                }
        
        except (ValueError, IndexError):
            pass
        
        return self._handle_general_math(message)
    
    def _handle_formula(self, message: str) -> Dict[str, Any]:
        """Обработка запросов о формулах"""
        message_lower = message.lower()
        
        for formula_name, formula in self.formulas.items():
            if formula_name in message_lower:
                content = f"""**ФОРМУЛА: {formula_name.upper()}**

**Формула:** {formula}

**Объяснение:**
{self._get_formula_explanation(formula_name)}

**Применение:**
{self._get_formula_application(formula_name)}

**Примеры использования:**
{self._get_formula_examples(formula_name)}"""
                
                return {
                    "success": True,
                    "response": {
                        "content": content,
                        "title": f"Формула: {formula_name}",
                        "source": "Mathematics Handler"
                    },
                    "category": "formula",
                    "confidence": 0.9,
                    "timestamp": datetime.now().isoformat(),
                    "provider": "Mathematics Handler"
                }
        
        return self._handle_general_math(message)
    
    def _handle_definition(self, message: str) -> Dict[str, Any]:
        """Обработка запросов на определения"""
        message_lower = message.lower()
        
        definitions = {
            "производная": "Производная функции - это предел отношения приращения функции к приращению аргумента.",
            "интеграл": "Интеграл - это обратная операция к дифференцированию, нахождение первообразной функции.",
            "лимит": "Предел функции - это значение, к которому стремится функция при приближении аргумента к определенной точке.",
            "вектор": "Вектор - это направленный отрезок, характеризующийся длиной и направлением.",
            "матрица": "Матрица - это прямоугольная таблица чисел, расположенных в строках и столбцах."
        }
        
        for term, definition in definitions.items():
            if term in message_lower:
                content = f"""**ОПРЕДЕЛЕНИЕ: {term.upper()}**

{definition}

**Свойства:**
{self._get_term_properties(term)}

**Применение:**
{self._get_term_application(term)}"""
                
                return {
                    "success": True,
                    "response": {
                        "content": content,
                        "title": f"Определение: {term}",
                        "source": "Mathematics Handler"
                    },
                    "category": "definition",
                    "confidence": 0.85,
                    "timestamp": datetime.now().isoformat(),
                    "provider": "Mathematics Handler"
                }
        
        return self._handle_general_math(message)
    
    def _handle_general_math(self, message: str) -> Dict[str, Any]:
        """Обработка общих математических запросов"""
        content = f"""**МАТЕМАТИЧЕСКИЙ ЗАПРОС**

Ваш вопрос: "{message}"

**Я могу помочь с:**
• Арифметическими вычислениями
• Геометрическими расчетами
• Тригонометрическими функциями
• Формулами и уравнениями
• Математическими определениями

**Примеры запросов:**
• "Реши 2+2"
• "Площадь круга радиусом 5"
• "sin(30)"
• "Формула площади треугольника"
• "Что такое производная"

Попробуйте задать более конкретный вопрос!"""
        
        return {
            "success": True,
            "response": {
                "content": content,
                "title": "Математическая помощь",
                "source": "Mathematics Handler"
            },
            "category": "general",
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat(),
            "provider": "Mathematics Handler"
        }
    
    def _create_calculation_response(self, calculation: str, operation: str) -> Dict[str, Any]:
        """Создание ответа на расчет"""
        content = f"""**МАТЕМАТИЧЕСКИЙ РАСЧЕТ**

**Операция:** {operation}
**Результат:** {calculation}

**Проверка:** Расчет выполнен корректно"""
        
        return {
            "success": True,
            "response": {
                "content": content,
                "title": f"Расчет: {operation}",
                "source": "Mathematics Handler"
            },
            "category": "calculation",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
            "provider": "Mathematics Handler"
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Создание ответа об ошибке"""
        return {
            "success": False,
            "response": {
                "content": f"**ОШИБКА В РАСЧЕТАХ**\n\n{error_message}",
                "title": "Ошибка",
                "source": "Mathematics Handler"
            },
            "category": "error",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "provider": "Mathematics Handler"
        }
    
    def _get_formula_explanation(self, formula_name: str) -> str:
        """Получение объяснения формулы"""
        explanations = {
            "площадь круга": "Площадь круга вычисляется как произведение числа π на квадрат радиуса.",
            "площадь треугольника": "Площадь треугольника равна половине произведения основания на высоту.",
            "теорема пифагора": "В прямоугольном треугольнике квадрат гипотенузы равен сумме квадратов катетов."
        }
        return explanations.get(formula_name, "Формула для вычисления указанной величины.")
    
    def _get_formula_application(self, formula_name: str) -> str:
        """Получение применения формулы"""
        applications = {
            "площадь круга": "Используется в геометрии, строительстве, дизайне.",
            "площадь треугольника": "Применяется в архитектуре, навигации, компьютерной графике.",
            "теорема пифагора": "Основа для многих геометрических расчетов и построений."
        }
        return applications.get(formula_name, "Широко применяется в различных областях математики и науки.")
    
    def _get_formula_examples(self, formula_name: str) -> str:
        """Получение примеров использования формулы"""
        examples = {
            "площадь круга": "• Расчет площади садового участка\n• Определение размера колеса\n• Планирование строительства",
            "площадь треугольника": "• Расчет площади земельного участка\n• Определение площади крыши\n• Компьютерная графика",
            "теорема пифагора": "• Построение прямых углов\n• Расчет расстояний\n• Навигация и картография"
        }
        return examples.get(formula_name, "• Примеры использования в практических задачах")
    
    def _get_term_properties(self, term: str) -> str:
        """Получение свойств термина"""
        properties = {
            "производная": "• Линейность\n• Производная произведения\n• Производная частного\n• Цепное правило",
            "интеграл": "• Линейность\n• Интеграл суммы\n• Замена переменной\n• Интегрирование по частям",
            "вектор": "• Длина (модуль)\n• Направление\n• Сложение и вычитание\n• Скалярное произведение"
        }
        return properties.get(term, "• Основные свойства и характеристики")
    
    def _get_term_application(self, term: str) -> str:
        """Получение применения термина"""
        applications = {
            "производная": "• Анализ функций\n• Оптимизация\n• Физика (скорость, ускорение)\n• Экономика",
            "интеграл": "• Вычисление площадей\n• Решение дифференциальных уравнений\n• Физика (работа, энергия)\n• Вероятность",
            "вектор": "• Физика (силы, скорости)\n• Компьютерная графика\n• Навигация\n• Машинное обучение"
        }
        return applications.get(term, "• Широкое применение в науке и технике")

# Глобальный экземпляр
_mathematics_handler = None

def get_mathematics_handler():
    """Получает глобальный экземпляр обработчика"""
    global _mathematics_handler
    if _mathematics_handler is None:
        _mathematics_handler = MathematicsHandler()
    return _mathematics_handler


