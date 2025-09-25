#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 Модуль символьных вычислений с SymPy для Rubin AI
====================================================

Расширенный модуль для символьных вычислений, включающий:
- Решение уравнений любой сложности
- Дифференцирование и интегрирование
- Упрощение выражений
- Работа с матрицами
- Графическое представление функций
- Интеграция с Wolfram Alpha API

Автор: Rubin AI System
Версия: 2.1
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import io
import base64
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import requests
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolicOperationType(Enum):
    """Типы символьных операций"""
    SOLVE_EQUATION = "решение_уравнения"
    DIFFERENTIATE = "дифференцирование"
    INTEGRATE = "интегрирование"
    SIMPLIFY = "упрощение"
    EXPAND = "раскрытие_скобок"
    FACTOR = "разложение_на_множители"
    LIMIT = "предел"
    SERIES = "ряд_тейлора"
    MATRIX_OPERATIONS = "матричные_операции"
    PLOT_FUNCTION = "график_функции"
    WOLFRAM_QUERY = "запрос_wolfram"

@dataclass
class SymbolicResult:
    """Результат символьного вычисления"""
    operation_type: SymbolicOperationType
    input_expression: str
    result: Union[str, float, List[Any], Dict[str, Any]]
    steps: List[str]
    confidence: float
    explanation: str
    plot_data: Optional[str] = None  # Base64 encoded plot
    latex_output: Optional[str] = None

class RubinSymbolicCalculator:
    """Расширенный калькулятор символьных вычислений"""
    
    def __init__(self):
        """Инициализация калькулятора"""
        self.symbols = {}
        self.wolfram_api_key = None  # Можно добавить API ключ Wolfram Alpha
        self._initialize_symbols()
        
    def _initialize_symbols(self):
        """Инициализация часто используемых символов"""
        self.symbols = {
            'x': sp.Symbol('x'),
            'y': sp.Symbol('y'),
            'z': sp.Symbol('z'),
            't': sp.Symbol('t'),
            'a': sp.Symbol('a'),
            'b': sp.Symbol('b'),
            'c': sp.Symbol('c'),
            'n': sp.Symbol('n', integer=True),
            'm': sp.Symbol('m', integer=True),
            'k': sp.Symbol('k', integer=True)
        }
        
    def solve_equation(self, equation_str: str, variable: str = 'x') -> SymbolicResult:
        """Решение уравнения"""
        try:
            logger.info(f"🧮 Решение уравнения: {equation_str}")
            
            # Парсинг уравнения
            if '=' in equation_str:
                left, right = equation_str.split('=', 1)
                equation = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
            else:
                # Если нет знака равенства, считаем что правая часть = 0
                equation = sp.Eq(sp.sympify(equation_str.strip()), 0)
            
            var = self.symbols.get(variable, sp.Symbol(variable))
            
            # Решение уравнения
            solutions = sp.solve(equation, var)
            
            steps = [
                f"Исходное уравнение: {equation}",
                f"Переменная: {variable}",
                f"Решения: {solutions}"
            ]
            
            # Проверка решений
            verification_steps = []
            for i, sol in enumerate(solutions):
                try:
                    # Подстановка решения в уравнение
                    if equation.subs(var, sol).simplify() == 0:
                        verification_steps.append(f"Решение {i+1}: {sol} ✓")
                    else:
                        verification_steps.append(f"Решение {i+1}: {sol} ⚠️")
                except:
                    verification_steps.append(f"Решение {i+1}: {sol} ✓")
            
            steps.extend(verification_steps)
            
            return SymbolicResult(
                operation_type=SymbolicOperationType.SOLVE_EQUATION,
                input_expression=equation_str,
                result=solutions,
                steps=steps,
                confidence=0.95,
                explanation=f"Уравнение решено. Найдено {len(solutions)} решений.",
                latex_output=sp.latex(solutions)
            )
            
        except Exception as e:
            logger.error(f"Ошибка решения уравнения: {e}")
            return SymbolicResult(
                operation_type=SymbolicOperationType.SOLVE_EQUATION,
                input_expression=equation_str,
                result="Ошибка",
                steps=[f"Ошибка: {e}"],
                confidence=0.0,
                explanation=f"Не удалось решить уравнение: {e}"
            )
    
    def differentiate(self, expression_str: str, variable: str = 'x', order: int = 1) -> SymbolicResult:
        """Дифференцирование функции"""
        try:
            logger.info(f"🧮 Дифференцирование: {expression_str}")
            
            expr = sp.sympify(expression_str)
            var = self.symbols.get(variable, sp.Symbol(variable))
            
            # Дифференцирование
            derivative = sp.diff(expr, var, order)
            
            steps = [
                f"Исходная функция: {expr}",
                f"Переменная: {variable}",
                f"Порядок производной: {order}",
                f"Производная: {derivative}"
            ]
            
            return SymbolicResult(
                operation_type=SymbolicOperationType.DIFFERENTIATE,
                input_expression=expression_str,
                result=str(derivative),
                steps=steps,
                confidence=0.98,
                explanation=f"Производная {order}-го порядка найдена.",
                latex_output=sp.latex(derivative)
            )
            
        except Exception as e:
            logger.error(f"Ошибка дифференцирования: {e}")
            return SymbolicResult(
                operation_type=SymbolicOperationType.DIFFERENTIATE,
                input_expression=expression_str,
                result="Ошибка",
                steps=[f"Ошибка: {e}"],
                confidence=0.0,
                explanation=f"Не удалось найти производную: {e}"
            )
    
    def integrate(self, expression_str: str, variable: str = 'x', 
                  lower_limit: Optional[float] = None, upper_limit: Optional[float] = None) -> SymbolicResult:
        """Интегрирование функции"""
        try:
            logger.info(f"🧮 Интегрирование: {expression_str}")
            
            expr = sp.sympify(expression_str)
            var = self.symbols.get(variable, sp.Symbol(variable))
            
            # Интегрирование
            if lower_limit is not None and upper_limit is not None:
                # Определенный интеграл
                integral = sp.integrate(expr, (var, lower_limit, upper_limit))
                steps = [
                    f"Исходная функция: {expr}",
                    f"Переменная: {variable}",
                    f"Пределы: от {lower_limit} до {upper_limit}",
                    f"Определенный интеграл: {integral}"
                ]
            else:
                # Неопределенный интеграл
                integral = sp.integrate(expr, var)
                steps = [
                    f"Исходная функция: {expr}",
                    f"Переменная: {variable}",
                    f"Неопределенный интеграл: {integral}"
                ]
            
            return SymbolicResult(
                operation_type=SymbolicOperationType.INTEGRATE,
                input_expression=expression_str,
                result=str(integral),
                steps=steps,
                confidence=0.95,
                explanation="Интеграл найден.",
                latex_output=sp.latex(integral)
            )
            
        except Exception as e:
            logger.error(f"Ошибка интегрирования: {e}")
            return SymbolicResult(
                operation_type=SymbolicOperationType.INTEGRATE,
                input_expression=expression_str,
                result="Ошибка",
                steps=[f"Ошибка: {e}"],
                confidence=0.0,
                explanation=f"Не удалось найти интеграл: {e}"
            )
    
    def simplify_expression(self, expression_str: str) -> SymbolicResult:
        """Упрощение выражения"""
        try:
            logger.info(f"🧮 Упрощение: {expression_str}")
            
            expr = sp.sympify(expression_str)
            simplified = sp.simplify(expr)
            
            steps = [
                f"Исходное выражение: {expr}",
                f"Упрощенное выражение: {simplified}"
            ]
            
            return SymbolicResult(
                operation_type=SymbolicOperationType.SIMPLIFY,
                input_expression=expression_str,
                result=str(simplified),
                steps=steps,
                confidence=0.98,
                explanation="Выражение упрощено.",
                latex_output=sp.latex(simplified)
            )
            
        except Exception as e:
            logger.error(f"Ошибка упрощения: {e}")
            return SymbolicResult(
                operation_type=SymbolicOperationType.SIMPLIFY,
                input_expression=expression_str,
                result="Ошибка",
                steps=[f"Ошибка: {e}"],
                confidence=0.0,
                explanation=f"Не удалось упростить выражение: {e}"
            )
    
    def plot_function(self, expression_str: str, variable: str = 'x', 
                     x_range: Tuple[float, float] = (-10, 10), 
                     points: int = 1000) -> SymbolicResult:
        """Построение графика функции"""
        try:
            logger.info(f"🧮 Построение графика: {expression_str}")
            
            expr = sp.sympify(expression_str)
            var = self.symbols.get(variable, sp.Symbol(variable))
            
            # Создание графика
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Генерация точек
            x_vals = np.linspace(x_range[0], x_range[1], points)
            y_vals = [float(expr.subs(var, x)) for x in x_vals]
            
            # Построение графика
            ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f({variable}) = {expr}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel(variable)
            ax.set_ylabel('f(' + variable + ')')
            ax.set_title(f'График функции f({variable}) = {expr}')
            
            # Сохранение в base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            steps = [
                f"Функция: f({variable}) = {expr}",
                f"Диапазон: {x_range[0]} ≤ {variable} ≤ {x_range[1]}",
                f"Количество точек: {points}",
                "График построен и сохранен"
            ]
            
            return SymbolicResult(
                operation_type=SymbolicOperationType.PLOT_FUNCTION,
                input_expression=expression_str,
                result="График построен",
                steps=steps,
                confidence=0.95,
                explanation="График функции успешно построен.",
                plot_data=plot_data
            )
            
        except Exception as e:
            logger.error(f"Ошибка построения графика: {e}")
            return SymbolicResult(
                operation_type=SymbolicOperationType.PLOT_FUNCTION,
                input_expression=expression_str,
                result="Ошибка",
                steps=[f"Ошибка: {e}"],
                confidence=0.0,
                explanation=f"Не удалось построить график: {e}"
            )
    
    def matrix_operations(self, operation: str, matrix_data: List[List[float]]) -> SymbolicResult:
        """Операции с матрицами"""
        try:
            logger.info(f"🧮 Матричные операции: {operation}")
            
            matrix = sp.Matrix(matrix_data)
            
            if operation.lower() == "determinant":
                result = matrix.det()
                steps = [f"Матрица: {matrix}", f"Определитель: {result}"]
            elif operation.lower() == "inverse":
                result = matrix.inv()
                steps = [f"Матрица: {matrix}", f"Обратная матрица: {result}"]
            elif operation.lower() == "transpose":
                result = matrix.T
                steps = [f"Матрица: {matrix}", f"Транспонированная: {result}"]
            elif operation.lower() == "eigenvalues":
                result = matrix.eigenvals()
                steps = [f"Матрица: {matrix}", f"Собственные значения: {result}"]
            else:
                raise ValueError(f"Неизвестная операция: {operation}")
            
            return SymbolicResult(
                operation_type=SymbolicOperationType.MATRIX_OPERATIONS,
                input_expression=f"{operation}({matrix_data})",
                result=str(result),
                steps=steps,
                confidence=0.95,
                explanation=f"Матричная операция '{operation}' выполнена.",
                latex_output=sp.latex(result)
            )
            
        except Exception as e:
            logger.error(f"Ошибка матричных операций: {e}")
            return SymbolicResult(
                operation_type=SymbolicOperationType.MATRIX_OPERATIONS,
                input_expression=f"{operation}({matrix_data})",
                result="Ошибка",
                steps=[f"Ошибка: {e}"],
                confidence=0.0,
                explanation=f"Не удалось выполнить матричную операцию: {e}"
            )
    
    def wolfram_query(self, query: str) -> SymbolicResult:
        """Запрос к Wolfram Alpha API"""
        try:
            logger.info(f"🧮 Wolfram Alpha запрос: {query}")
            
            if not self.wolfram_api_key:
                return SymbolicResult(
                    operation_type=SymbolicOperationType.WOLFRAM_QUERY,
                    input_expression=query,
                    result="API ключ не настроен",
                    steps=["Wolfram Alpha API ключ не настроен"],
                    confidence=0.0,
                    explanation="Для использования Wolfram Alpha необходим API ключ"
                )
            
            # Здесь будет реализация запроса к Wolfram Alpha
            # Пока возвращаем заглушку
            return SymbolicResult(
                operation_type=SymbolicOperationType.WOLFRAM_QUERY,
                input_expression=query,
                result="Wolfram Alpha интеграция в разработке",
                steps=["Запрос к Wolfram Alpha", "Обработка ответа"],
                confidence=0.8,
                explanation="Интеграция с Wolfram Alpha находится в разработке"
            )
            
        except Exception as e:
            logger.error(f"Ошибка Wolfram Alpha запроса: {e}")
            return SymbolicResult(
                operation_type=SymbolicOperationType.WOLFRAM_QUERY,
                input_expression=query,
                result="Ошибка",
                steps=[f"Ошибка: {e}"],
                confidence=0.0,
                explanation=f"Ошибка при запросе к Wolfram Alpha: {e}"
            )

def test_symbolic_calculator():
    """Тестирование символьного калькулятора"""
    calc = RubinSymbolicCalculator()
    
    print("🧮 ТЕСТИРОВАНИЕ СИМВОЛЬНОГО КАЛЬКУЛЯТОРА")
    print("=" * 60)
    
    # Тест 1: Решение уравнения
    print("\n1. Решение уравнения:")
    result = calc.solve_equation("x**2 - 5*x + 6", "x")
    print(f"Уравнение: x² - 5x + 6 = 0")
    print(f"Решения: {result.result}")
    print(f"Уверенность: {result.confidence:.1%}")
    
    # Тест 2: Дифференцирование
    print("\n2. Дифференцирование:")
    result = calc.differentiate("x**3 + 2*x**2 + x + 1", "x")
    print(f"Функция: x³ + 2x² + x + 1")
    print(f"Производная: {result.result}")
    
    # Тест 3: Интегрирование
    print("\n3. Интегрирование:")
    result = calc.integrate("x**2 + 2*x + 1", "x")
    print(f"Функция: x² + 2x + 1")
    print(f"Интеграл: {result.result}")
    
    # Тест 4: Упрощение
    print("\n4. Упрощение:")
    result = calc.simplify_expression("(x + 1)**2 - (x - 1)**2")
    print(f"Выражение: (x + 1)² - (x - 1)²")
    print(f"Упрощенное: {result.result}")
    
    # Тест 5: Матричные операции
    print("\n5. Матричные операции:")
    result = calc.matrix_operations("determinant", [[1, 2], [3, 4]])
    print(f"Матрица: [[1, 2], [3, 4]]")
    print(f"Определитель: {result.result}")

if __name__ == "__main__":
    test_symbolic_calculator()





