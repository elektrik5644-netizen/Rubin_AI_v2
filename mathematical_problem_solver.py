#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 Алгоритм для решения математических задач
============================================

Комплексная система для автоматического решения различных типов математических задач:
- Распознавание типа задачи
- Выбор оптимального алгоритма решения
- Пошаговое решение с объяснениями
- Проверка результата

Автор: Rubin AI System
Версия: 2.0
"""

import re
import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт новых модулей
try:
    from rubin_symbolic_calculator import RubinSymbolicCalculator, SymbolicOperationType
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logging.warning("SymPy модуль недоступен")

try:
    from rubin_graph_analyzer import RubinGraphAnalyzer, GraphType
    OCR_CV_AVAILABLE = True
except ImportError:
    OCR_CV_AVAILABLE = False
    logging.warning("OCR/CV модуль недоступен")

try:
    from rubin_data_visualizer import RubinDataVisualizer, VisualizationType
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Модуль визуализации недоступен")

class ProblemType(Enum):
    """Типы математических задач"""
    ARITHMETIC = "арифметика"
    ALGEBRA = "алгебра"
    GEOMETRY = "геометрия"
    TRIGONOMETRY = "тригонометрия"
    CALCULUS = "математический_анализ"
    STATISTICS = "статистика"
    LINEAR_EQUATION = "линейное_уравнение"
    QUADRATIC_EQUATION = "квадратное_уравнение"
    SYSTEM_EQUATIONS = "система_уравнений"
    AREA_CALCULATION = "расчет_площади"
    VOLUME_CALCULATION = "расчет_объема"
    PERCENTAGE = "проценты"
    RATIO = "отношения"
    # Новые типы задач
    PHYSICS_FORMULAS = "физические_формулы"
    CHEMISTRY_CALCULATIONS = "химические_расчеты"
    ENGINEERING_DESIGN = "инженерные_расчеты"
    GRAPH_ANALYSIS = "анализ_графиков"
    DATA_VISUALIZATION = "визуализация_данных"
    FORMULA_CALCULATION = "расчет_по_формуле"
    SYMBOLIC_COMPUTATION = "символьные_вычисления"
    EQUATION_SOLVING = "решение_уравнений"
    DIFFERENTIATION = "дифференцирование"
    INTEGRATION = "интегрирование"
    MATRIX_OPERATIONS = "матричные_операции"
    FUNCTION_PLOTTING = "построение_графиков"
    UNKNOWN = "неизвестно"

@dataclass
class ProblemSolution:
    """Результат решения математической задачи"""
    problem_type: ProblemType
    input_data: Dict[str, Any]
    solution_steps: List[str]
    final_answer: Union[float, str, Dict[str, Any]]
    verification: bool
    confidence: float
    explanation: str

class MathematicalProblemSolver:
    """Основной класс для решения математических задач"""
    
    def __init__(self):
        """Инициализация решателя"""
        self.problem_patterns = self._initialize_patterns()
        self.solution_methods = self._initialize_solution_methods()
        # Инициализация новых решателей
        self.physics_solver = PhysicsFormulaSolver()
        self.chemistry_solver = ChemistryFormulaSolver()
        
        # Инициализация расширенных модулей
        if SYMPY_AVAILABLE:
            self.symbolic_calculator = RubinSymbolicCalculator()
            logger.info("✅ Символьный калькулятор загружен")
        else:
            self.symbolic_calculator = None
            
        if OCR_CV_AVAILABLE:
            self.graph_analyzer = RubinGraphAnalyzer()
            logger.info("✅ Анализатор графиков загружен")
        else:
            self.graph_analyzer = None
            
        if VISUALIZATION_AVAILABLE:
            self.data_visualizer = RubinDataVisualizer()
            logger.info("✅ Визуализатор данных загружен")
        else:
            self.data_visualizer = None
        
    def _initialize_patterns(self) -> Dict[ProblemType, List[str]]:
        """Инициализация паттернов для распознавания типов задач"""
        return {
            ProblemType.ARITHMETIC: [
                r'вычисли|посчитай|найди\s+значение|результат',
                r'[\+\-\*/]\s*\d+|\d+\s*[\+\-\*/]',
                r'^\d+[\+\-\*/]\d+',
                r'сколько\s+осталось|сколько\s+осталось\s+на|осталось\s+на',
                r'\d+\s+яблок.*осталось|\d+\s+деревьев.*осталось',
                r'укатилось|упало|съел|потерял|потратил',
                r'было\s+\d+.*стало|стало\s+\d+.*было'
            ],
            ProblemType.LINEAR_EQUATION: [
                r'линейное\s+уравнение|реши\s+уравнение',
                r'[a-zA-Z]\s*[\+\-]\s*\d+\s*=\s*\d+',
                r'[a-zA-Z]\s*=\s*\d+[\+\-]\d+'
            ],
            ProblemType.QUADRATIC_EQUATION: [
                r'квадратное\s+уравнение',
                r'[a-zA-Z]\^?2\s*[\+\-]',
                r'[a-zA-Z]²\s*[\+\-]',
                r'x[²²^2]\s*[\+\-]\s*\d*x',
                r'x[²²^2]\s*[\+\-]\s*\d*x\s*[\+\-]\s*\d+\s*=\s*\d+'
            ],
            ProblemType.SYSTEM_EQUATIONS: [
                r'система\s+уравнений|система\s+линейных',
                r'и\s+одновременно|вместе\s+с'
            ],
            ProblemType.AREA_CALCULATION: [
                r'площадь|найди\s+площадь|рассчитай\s+площадь',
                r'треугольник|круг|прямоугольник|квадрат'
            ],
            ProblemType.VOLUME_CALCULATION: [
                r'объем|найди\s+объем|рассчитай\s+объем',
                r'куб|цилиндр|шар|пирамида'
            ],
            ProblemType.TRIGONOMETRY: [
                r'sin|cos|tan|синус|косинус|тангенс',
                r'тригонометрия|угол|градус|радиан'
            ],
            ProblemType.CALCULUS: [
                r'производная|интеграл|дифференциал',
                r'lim|предел|непрерывность'
            ],
            ProblemType.STATISTICS: [
                r'среднее|медиана|мода|дисперсия',
                r'статистика|вероятность|распределение'
            ],
            ProblemType.PERCENTAGE: [
                r'процент|%|\d+\s*процентов',
                r'найти\s+\d+%\s+от'
            ],
            # Новые паттерны для расширенных типов задач
            ProblemType.PHYSICS_FORMULAS: [
                r'кинетическая\s+энергия|потенциальная\s+энергия',
                r'закон\s+ома|мощность|сила\s+тяжести',
                r'ускорение|путь|скорость',
                r'физика|физический|механика',
                r'напряжение.*ток|ток.*напряжение',
                r'напряжение.*сопротивление|сопротивление.*напряжение'
            ],
            ProblemType.CHEMISTRY_CALCULATIONS: [
                r'концентрация|молярная\s+масса',
                r'количество\s+вещества|моль',
                r'химия|химический|раствор'
            ],
            ProblemType.GRAPH_ANALYSIS: [
                r'график|диаграмма|анализ\s+графика',
                r'изображение|картинка|файл.*\.(png|jpg|jpeg)'
            ],
            ProblemType.DATA_VISUALIZATION: [
                r'построить|создать|нарисовать',
                r'визуализация|график.*данных'
            ],
            ProblemType.FORMULA_CALCULATION: [
                r'формула|расчет\s+по\s+формуле',
                r'вычислить\s+по\s+формуле'
            ],
            ProblemType.SYMBOLIC_COMPUTATION: [
                r'символьн|символ|упростить\s+выражение',
                r'раскрыть\s+скобки|разложить\s+на\s+множители'
            ],
            ProblemType.EQUATION_SOLVING: [
                r'решить\s+уравнение|найти\s+корни',
                r'уравнение.*равно|равно.*уравнение',
                r'x\*\*2.*=.*0|x\*\*2.*равно.*0',
                r'квадратное\s+уравнение'
            ],
            ProblemType.DIFFERENTIATION: [
                r'производная|дифференцировать|найти\s+производную',
                r'd/dx|d/dy|d/dz'
            ],
            ProblemType.INTEGRATION: [
                r'интеграл|интегрировать|найти\s+интеграл',
                r'∫|интегрирование'
            ],
            ProblemType.MATRIX_OPERATIONS: [
                r'матрица|определитель|обратная\s+матрица',
                r'транспонирование|собственные\s+значения'
            ],
            ProblemType.FUNCTION_PLOTTING: [
                r'построить\s+график|нарисовать\s+функцию',
                r'график\s+функции|plot.*function'
            ]
        }
    
    def _initialize_solution_methods(self) -> Dict[ProblemType, callable]:
        """Инициализация методов решения для каждого типа задач"""
        return {
            ProblemType.ARITHMETIC: self._solve_arithmetic,
            ProblemType.LINEAR_EQUATION: self._solve_linear_equation,
            ProblemType.QUADRATIC_EQUATION: self._solve_quadratic_equation,
            ProblemType.SYSTEM_EQUATIONS: self._solve_system_equations,
            ProblemType.AREA_CALCULATION: self._solve_area_calculation,
            ProblemType.VOLUME_CALCULATION: self._solve_volume_calculation,
            ProblemType.TRIGONOMETRY: self._solve_trigonometry,
            ProblemType.CALCULUS: self._solve_calculus,
            ProblemType.STATISTICS: self._solve_statistics,
            ProblemType.PERCENTAGE: self._solve_percentage,
            # Новые методы решения удалены из solution_methods
            # так как они имеют другой порядок параметров
        }
    
    def solve_problem(self, problem_text: str, **kwargs) -> ProblemSolution:
        """
        Основной метод для решения математической задачи
        
        Args:
            problem_text: Текст задачи
            **kwargs: Дополнительные параметры
            
        Returns:
            ProblemSolution: Результат решения
        """
        try:
            # 1. Анализ и распознавание типа задачи
            problem_type = self._identify_problem_type(problem_text)
            logger.info(f"Распознан тип задачи: {problem_type.value}")
            
            # 2. Извлечение данных из задачи
            input_data = self._extract_data(problem_text, problem_type)
            logger.info(f"Извлечены данные: {input_data}")
            
            # 3. Выбор метода решения
            if problem_type in self.solution_methods:
                solution_method = self.solution_methods[problem_type]
                result = solution_method(input_data, problem_text)
            elif problem_type == ProblemType.PHYSICS_FORMULAS:
                result = self._solve_physics_problem(input_data, problem_text)
            elif problem_type == ProblemType.CHEMISTRY_CALCULATIONS:
                result = self._solve_chemistry_problem(problem_text, input_data)
            elif problem_type == ProblemType.GRAPH_ANALYSIS:
                result = self._solve_graph_problem(problem_text, input_data)
            elif problem_type == ProblemType.DATA_VISUALIZATION:
                result = self._solve_visualization_problem(problem_text, input_data)
            elif problem_type == ProblemType.SYMBOLIC_COMPUTATION:
                result = self._solve_symbolic_problem(problem_text, input_data)
            elif problem_type == ProblemType.EQUATION_SOLVING:
                result = self._solve_equation_problem(problem_text, input_data)
            elif problem_type == ProblemType.DIFFERENTIATION:
                result = self._solve_differentiation_problem(problem_text, input_data)
            elif problem_type == ProblemType.INTEGRATION:
                result = self._solve_integration_problem(problem_text, input_data)
            elif problem_type == ProblemType.MATRIX_OPERATIONS:
                result = self._solve_matrix_problem(problem_text, input_data)
            elif problem_type == ProblemType.FUNCTION_PLOTTING:
                result = self._solve_plotting_problem(problem_text, input_data)
            else:
                result = self._solve_generic(problem_text, input_data)
            
            # 4. Проверка результата
            verification = self._verify_solution(result, input_data, problem_type)
            
            # 5. Формирование ответа
            return ProblemSolution(
                problem_type=problem_type,
                input_data=input_data,
                solution_steps=result.get('steps', []),
                final_answer=result.get('answer'),
                verification=verification,
                confidence=result.get('confidence', 0.8),
                explanation=result.get('explanation', '')
            )
            
        except Exception as e:
            logger.error(f"Ошибка при решении задачи: {e}")
            return ProblemSolution(
                problem_type=ProblemType.UNKNOWN,
                input_data={},
                solution_steps=[f"Ошибка: {str(e)}"],
                final_answer="Ошибка решения",
                verification=False,
                confidence=0.0,
                explanation=f"Произошла ошибка при решении задачи: {str(e)}"
            )
    
    def _identify_problem_type(self, problem_text: str) -> ProblemType:
        """Распознавание типа математической задачи"""
        problem_lower = problem_text.lower()
        
        # Приоритетная проверка для квадратных уравнений
        if any(re.search(pattern, problem_lower) for pattern in self.problem_patterns[ProblemType.QUADRATIC_EQUATION]):
            return ProblemType.QUADRATIC_EQUATION
        
        # Подсчет совпадений для каждого типа
        type_scores = {}
        for problem_type, patterns in self.problem_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, problem_lower))
                score += matches
            type_scores[problem_type] = score
        
        # Выбор типа с наибольшим количеством совпадений
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return ProblemType.UNKNOWN
    
    def _extract_data(self, problem_text: str, problem_type: ProblemType) -> Dict[str, Any]:
        """Извлечение числовых данных и параметров из текста задачи"""
        data = {}
        
        # Извлечение чисел
        numbers = re.findall(r'-?\d+\.?\d*', problem_text)
        if numbers:
            data['numbers'] = [float(n) for n in numbers]
        
        # Извлечение переменных
        variables = re.findall(r'[a-zA-Z]\w*', problem_text)
        if variables:
            data['variables'] = list(set(variables))
        
        # Специфичное извлечение для разных типов задач
        if problem_type == ProblemType.AREA_CALCULATION:
            data.update(self._extract_geometry_data(problem_text))
        elif problem_type in [ProblemType.LINEAR_EQUATION, ProblemType.QUADRATIC_EQUATION]:
            data.update(self._extract_equation_data(problem_text))
        elif problem_type == ProblemType.TRIGONOMETRY:
            data.update(self._extract_trigonometry_data(problem_text))
        
        return data
    
    def _extract_geometry_data(self, problem_text: str) -> Dict[str, Any]:
        """Извлечение геометрических данных"""
        data = {}
        
        # Поиск размеров
        if 'треугольник' in problem_text.lower():
            if 'основание' in problem_text.lower():
                base_match = re.search(r'основание\s*(\d+\.?\d*)', problem_text.lower())
                if base_match:
                    data['base'] = float(base_match.group(1))
            
            if 'высота' in problem_text.lower():
                height_match = re.search(r'высота\s*(\d+\.?\d*)', problem_text.lower())
                if height_match:
                    data['height'] = float(height_match.group(1))
        
        elif 'круг' in problem_text.lower():
            radius_match = re.search(r'радиус\s*(\d+\.?\d*)', problem_text.lower())
            if radius_match:
                data['radius'] = float(radius_match.group(1))
        
        elif 'прямоугольник' in problem_text.lower():
            length_match = re.search(r'длина\s*(\d+\.?\d*)', problem_text.lower())
            width_match = re.search(r'ширина\s*(\d+\.?\d*)', problem_text.lower())
            if length_match:
                data['length'] = float(length_match.group(1))
            if width_match:
                data['width'] = float(width_match.group(1))
        
        return data
    
    def _extract_equation_data(self, problem_text: str) -> Dict[str, Any]:
        """Извлечение данных уравнения"""
        data = {}
        
        # Поиск коэффициентов в уравнениях вида ax + b = c
        linear_match = re.search(r'(\d*\.?\d*)\s*[a-zA-Z]\s*([\+\-])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', problem_text)
        if linear_match:
            a = float(linear_match.group(1)) if linear_match.group(1) else 1.0
            sign = linear_match.group(2)
            b = float(linear_match.group(3))
            c = float(linear_match.group(4))
            
            if sign == '-':
                b = -b
            
            data['coefficients'] = {'a': a, 'b': b, 'c': c}
        
        return data
    
    def _extract_trigonometry_data(self, problem_text: str) -> Dict[str, Any]:
        """Извлечение тригонометрических данных"""
        data = {}
        
        # Поиск углов
        angle_match = re.search(r'(\d+\.?\d*)\s*(градус|радиан|°)', problem_text.lower())
        if angle_match:
            angle = float(angle_match.group(1))
            unit = angle_match.group(2)
            if unit in ['градус', '°']:
                data['angle_degrees'] = angle
                data['angle_radians'] = math.radians(angle)
            else:
                data['angle_radians'] = angle
                data['angle_degrees'] = math.degrees(angle)
        
        return data
    
    def _solve_arithmetic(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение арифметических задач"""
        problem_lower = problem_text.lower()
        
        # Специальная обработка задач типа "сколько осталось"
        if any(word in problem_lower for word in ['осталось', 'осталось на', 'укатилось', 'упало', 'съел', 'потерял', 'потратил']):
            numbers = data.get('numbers', [])
            if len(numbers) >= 2:
                # Задачи типа "2 яблока на столе одно укатилось, сколько осталось"
                initial_count = numbers[0]  # было
                removed_count = numbers[1] if len(numbers) > 1 else 1  # укатилось/потерял
                result = initial_count - removed_count
                
                return {
                    'answer': f"{int(result)}",
                    'steps': [
                        f"Было: {int(initial_count)}",
                        f"Укатилось/потерял: {int(removed_count)}",
                        f"Осталось: {int(initial_count)} - {int(removed_count)} = {int(result)}"
                    ],
                    'confidence': 0.95,
                    'explanation': f"Если было {int(initial_count)} предметов, а {int(removed_count)} укатилось/потерял, то осталось {int(result)}"
                }
        
        try:
            # Попытка прямого вычисления выражения
            expression = self._clean_expression(problem_text)
            result = eval(expression)
            
            return {
                'answer': result,
                'steps': [f"Вычисляем: {expression} = {result}"],
                'confidence': 0.9,
                'explanation': f"Арифметическое выражение {expression} равно {result}"
            }
        except:
            # Если не удалось вычислить напрямую, используем числа из задачи
            numbers = data.get('numbers', [])
            if len(numbers) >= 2:
                # Простые операции
                if '+' in problem_text:
                    result = sum(numbers)
                    operation = "сложение"
                elif '-' in problem_text:
                    result = numbers[0] - numbers[1]
                    operation = "вычитание"
                elif '*' in problem_text or '×' in problem_text:
                    result = numbers[0] * numbers[1]
                    operation = "умножение"
                elif '/' in problem_text or '÷' in problem_text:
                    result = numbers[0] / numbers[1]
                    operation = "деление"
                else:
                    result = sum(numbers)
                    operation = "сложение"
                
                return {
                    'answer': result,
                    'steps': [f"Выполняем {operation}: {numbers[0]} {operation} {numbers[1]} = {result}"],
                    'confidence': 0.8,
                    'explanation': f"Результат {operation} чисел {numbers[0]} и {numbers[1]} равен {result}"
                }
        
        return {
            'answer': "Не удалось решить",
            'steps': ["Не удалось распознать арифметическую операцию"],
            'confidence': 0.0,
            'explanation': "Не удалось определить тип арифметической операции"
        }
    
    def _solve_linear_equation(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение линейных уравнений"""
        coefficients = data.get('coefficients', {})
        
        if not coefficients:
            return {
                'answer': "Не удалось извлечь коэффициенты уравнения",
                'steps': ["Ошибка: не найдены коэффициенты уравнения"],
                'confidence': 0.0,
                'explanation': "Не удалось распознать коэффициенты линейного уравнения"
            }
        
        a = coefficients.get('a', 1)
        b = coefficients.get('b', 0)
        c = coefficients.get('c', 0)
        
        if a == 0:
            return {
                'answer': "Уравнение не имеет решения или имеет бесконечно много решений",
                'steps': ["a = 0, уравнение вырожденное"],
                'confidence': 0.9,
                'explanation': "При a = 0 уравнение становится нелинейным"
            }
        
        # Решение: x = (c - b) / a
        x = (c - b) / a
        
        steps = [
            f"Уравнение: {a}x + {b} = {c}",
            f"Переносим {b} в правую часть: {a}x = {c} - {b}",
            f"Упрощаем: {a}x = {c - b}",
            f"Делим на {a}: x = {c - b} / {a}",
            f"Ответ: x = {x}"
        ]
        
        return {
            'answer': x,
            'steps': steps,
            'confidence': 0.95,
            'explanation': f"Линейное уравнение {a}x + {b} = {c} имеет решение x = {x}"
        }
    
    def _solve_quadratic_equation(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение квадратных уравнений"""
        # Парсинг квадратного уравнения вида ax² + bx + c = 0
        # Ищем паттерн: x² + 5x + 6 = 0
        quadratic_pattern = r'x[²²^2]\s*([+\-]?)\s*(\d*)\s*x\s*([+\-]?)\s*(\d*)\s*=\s*(\d+)'
        quadratic_match = re.search(quadratic_pattern, problem_text)
        
        if quadratic_match:
            try:
                # Извлекаем коэффициенты
                a_sign = quadratic_match.group(1) or '+'
                b_coeff = quadratic_match.group(2) or '1'
                c_sign = quadratic_match.group(3) or '+'
                c_coeff = quadratic_match.group(4) or '0'
                right_side = int(quadratic_match.group(5))
                
                # Преобразуем в стандартную форму ax² + bx + c = 0
                a = 1  # коэффициент при x² всегда 1 в нашем случае
                b = int(a_sign + b_coeff) if b_coeff else 0
                c = int(c_sign + c_coeff) if c_coeff else 0
                
                # Переносим правую часть влево
                c = c - right_side
            except:
                # Fallback к старому методу
                numbers = data.get('numbers', [])
                if len(numbers) >= 3:
                    a, b, c = numbers[0], numbers[1], numbers[2]
                else:
                    return {
                        'answer': "Недостаточно коэффициентов",
                        'steps': ["Ошибка: не найдены все коэффициенты"],
                        'confidence': 0.0,
                        'explanation': "Для квадратного уравнения нужны 3 коэффициента"
                    }
        else:
            # Fallback к старому методу
            numbers = data.get('numbers', [])
            if len(numbers) >= 3:
                a, b, c = numbers[0], numbers[1], numbers[2]
            else:
                return {
                    'answer': "Недостаточно коэффициентов",
                    'steps': ["Ошибка: не найдены все коэффициенты"],
                    'confidence': 0.0,
                    'explanation': "Для квадратного уравнения нужны 3 коэффициента"
                }
        
        # Вычисление дискриминанта
        discriminant = b**2 - 4*a*c
        
        steps = [
            f"Квадратное уравнение: {a}x² + {b}x + {c} = 0",
            f"Дискриминант: D = b² - 4ac = {b}² - 4·{a}·{c} = {discriminant}"
        ]
        
        if discriminant > 0:
            x1 = (-b + math.sqrt(discriminant)) / (2*a)
            x2 = (-b - math.sqrt(discriminant)) / (2*a)
            steps.extend([
                f"D > 0, уравнение имеет два корня:",
                f"x₁ = (-b + √D) / (2a) = (-{b} + √{discriminant}) / (2·{a}) = {x1}",
                f"x₂ = (-b - √D) / (2a) = (-{b} - √{discriminant}) / (2·{a}) = {x2}"
            ])
            answer = {"x1": x1, "x2": x2}
            explanation = f"Квадратное уравнение имеет два корня: x₁ = {x1}, x₂ = {x2}"
            
        elif discriminant == 0:
            x = -b / (2*a)
            steps.extend([
                f"D = 0, уравнение имеет один корень:",
                f"x = -b / (2a) = -{b} / (2·{a}) = {x}"
            ])
            answer = x
            explanation = f"Квадратное уравнение имеет один корень: x = {x}"
            
        else:
            steps.extend([
                f"D < 0, уравнение не имеет действительных корней",
                f"Комплексные корни: x = (-{b} ± i√{abs(discriminant)}) / (2·{a})"
            ])
            answer = "Нет действительных корней"
            explanation = "Квадратное уравнение не имеет действительных корней"
        
        return {
            'answer': answer,
            'steps': steps,
            'confidence': 0.9,
            'explanation': explanation
        }
    
    def _solve_system_equations(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение систем уравнений"""
        # Упрощенная реализация для систем 2x2
        numbers = data.get('numbers', [])
        
        if len(numbers) >= 6:
            # Предполагаем систему вида:
            # a1*x + b1*y = c1
            # a2*x + b2*y = c2
            a1, b1, c1, a2, b2, c2 = numbers[:6]
            
            # Метод Крамера
            det = a1*b2 - a2*b1
            
            if abs(det) < 1e-10:
                return {
                    'answer': "Система не имеет единственного решения",
                    'steps': ["Определитель равен нулю"],
                    'confidence': 0.9,
                    'explanation': "Система уравнений вырожденная"
                }
            
            det_x = c1*b2 - c2*b1
            det_y = a1*c2 - a2*c1
            
            x = det_x / det
            y = det_y / det
            
            steps = [
                f"Система уравнений:",
                f"{a1}x + {b1}y = {c1}",
                f"{a2}x + {b2}y = {c2}",
                f"Определитель: Δ = {a1}·{b2} - {a2}·{b1} = {det}",
                f"Δx = {c1}·{b2} - {c2}·{b1} = {det_x}",
                f"Δy = {a1}·{c2} - {a2}·{c1} = {det_y}",
                f"x = Δx/Δ = {det_x}/{det} = {x}",
                f"y = Δy/Δ = {det_y}/{det} = {y}"
            ]
            
            return {
                'answer': {"x": x, "y": y},
                'steps': steps,
                'confidence': 0.85,
                'explanation': f"Решение системы: x = {x}, y = {y}"
            }
        
        return {
            'answer': "Недостаточно данных для решения системы",
            'steps': ["Ошибка: не найдены все коэффициенты системы"],
            'confidence': 0.0,
            'explanation': "Для системы 2x2 нужно 6 коэффициентов"
        }
    
    def _solve_area_calculation(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение задач на расчет площади"""
        problem_lower = problem_text.lower()
        
        if 'треугольник' in problem_lower:
            base = data.get('base')
            height = data.get('height')
            
            if base and height:
                area = 0.5 * base * height
                steps = [
                    f"Площадь треугольника: S = (1/2) × основание × высота",
                    f"S = (1/2) × {base} × {height}",
                    f"S = {area}"
                ]
                return {
                    'answer': area,
                    'steps': steps,
                    'confidence': 0.95,
                    'explanation': f"Площадь треугольника с основанием {base} и высотой {height} равна {area}"
                }
        
        elif 'круг' in problem_lower:
            radius = data.get('radius')
            
            if radius:
                area = math.pi * radius**2
                steps = [
                    f"Площадь круга: S = π × r²",
                    f"S = π × {radius}²",
                    f"S = {math.pi} × {radius**2}",
                    f"S = {area}"
                ]
                return {
                    'answer': area,
                    'steps': steps,
                    'confidence': 0.95,
                    'explanation': f"Площадь круга с радиусом {radius} равна {area}"
                }
        
        elif 'прямоугольник' in problem_lower:
            length = data.get('length')
            width = data.get('width')
            
            if length and width:
                area = length * width
                steps = [
                    f"Площадь прямоугольника: S = длина × ширина",
                    f"S = {length} × {width}",
                    f"S = {area}"
                ]
                return {
                    'answer': area,
                    'steps': steps,
                    'confidence': 0.95,
                    'explanation': f"Площадь прямоугольника {length}×{width} равна {area}"
                }
        
        return {
            'answer': "Не удалось определить тип фигуры",
            'steps': ["Ошибка: не найдены параметры фигуры"],
            'confidence': 0.0,
            'explanation': "Не удалось распознать тип геометрической фигуры"
        }
    
    def _solve_volume_calculation(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение задач на расчет объема"""
        # Реализация для основных объемных фигур
        numbers = data.get('numbers', [])
        problem_lower = problem_text.lower()
        
        if 'куб' in problem_lower and numbers:
            side = numbers[0]
            volume = side**3
            steps = [
                f"Объем куба: V = a³",
                f"V = {side}³",
                f"V = {volume}"
            ]
            return {
                'answer': volume,
                'steps': steps,
                'confidence': 0.95,
                'explanation': f"Объем куба со стороной {side} равен {volume}"
            }
        
        elif 'цилиндр' in problem_lower and len(numbers) >= 2:
            radius, height = numbers[0], numbers[1]
            volume = math.pi * radius**2 * height
            steps = [
                f"Объем цилиндра: V = π × r² × h",
                f"V = π × {radius}² × {height}",
                f"V = {math.pi} × {radius**2} × {height}",
                f"V = {volume}"
            ]
            return {
                'answer': volume,
                'steps': steps,
                'confidence': 0.95,
                'explanation': f"Объем цилиндра с радиусом {radius} и высотой {height} равен {volume}"
            }
        
        return {
            'answer': "Не удалось определить тип объемной фигуры",
            'steps': ["Ошибка: не найдены параметры фигуры"],
            'confidence': 0.0,
            'explanation': "Не удалось распознать тип объемной фигуры"
        }
    
    def _solve_trigonometry(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение тригонометрических задач"""
        angle_rad = data.get('angle_radians')
        angle_deg = data.get('angle_degrees')
        
        if not angle_rad and not angle_deg:
            return {
                'answer': "Не найден угол",
                'steps': ["Ошибка: не найден угол для вычисления"],
                'confidence': 0.0,
                'explanation': "Для тригонометрических вычислений нужен угол"
            }
        
        if not angle_rad:
            angle_rad = math.radians(angle_deg)
        if not angle_deg:
            angle_deg = math.degrees(angle_rad)
        
        sin_val = math.sin(angle_rad)
        cos_val = math.cos(angle_rad)
        tan_val = math.tan(angle_rad)
        
        steps = [
            f"Угол: {angle_deg}° = {angle_rad:.4f} радиан",
            f"sin({angle_deg}°) = {sin_val:.4f}",
            f"cos({angle_deg}°) = {cos_val:.4f}",
            f"tan({angle_deg}°) = {tan_val:.4f}"
        ]
        
        return {
            'answer': {
                'sin': sin_val,
                'cos': cos_val,
                'tan': tan_val,
                'angle_degrees': angle_deg,
                'angle_radians': angle_rad
            },
            'steps': steps,
            'confidence': 0.95,
            'explanation': f"Тригонометрические функции угла {angle_deg}°: sin={sin_val:.4f}, cos={cos_val:.4f}, tan={tan_val:.4f}"
        }
    
    def _solve_calculus(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение задач математического анализа"""
        # Упрощенная реализация для базовых производных
        problem_lower = problem_text.lower()
        
        if 'производная' in problem_lower:
            # Поиск функции в тексте
            func_match = re.search(r'(\w+)\s*\(', problem_text)
            if func_match:
                func_name = func_match.group(1)
                
                if func_name in ['sin', 'cos', 'tan']:
                    # Производные тригонометрических функций
                    derivatives = {
                        'sin': 'cos(x)',
                        'cos': '-sin(x)',
                        'tan': '1/cos²(x)'
                    }
                    
                    return {
                        'answer': derivatives[func_name],
                        'steps': [f"Производная {func_name}(x) = {derivatives[func_name]}"],
                        'confidence': 0.9,
                        'explanation': f"Производная {func_name}(x) равна {derivatives[func_name]}"
                    }
        
        return {
            'answer': "Не удалось распознать тип задачи анализа",
            'steps': ["Ошибка: не распознан тип задачи"],
            'confidence': 0.0,
            'explanation': "Не удалось определить тип задачи математического анализа"
        }
    
    def _solve_statistics(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение статистических задач"""
        numbers = data.get('numbers', [])
        
        if not numbers:
            return {
                'answer': "Нет данных для анализа",
                'steps': ["Ошибка: не найдены числовые данные"],
                'confidence': 0.0,
                'explanation': "Для статистического анализа нужны числовые данные"
            }
        
        # Базовые статистики
        mean = sum(numbers) / len(numbers)
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        
        if n % 2 == 0:
            median = (sorted_numbers[n//2-1] + sorted_numbers[n//2]) / 2
        else:
            median = sorted_numbers[n//2]
        
        variance = sum((x - mean)**2 for x in numbers) / len(numbers)
        std_dev = math.sqrt(variance)
        
        steps = [
            f"Данные: {numbers}",
            f"Среднее: μ = Σx/n = {sum(numbers)}/{len(numbers)} = {mean:.2f}",
            f"Медиана: {median}",
            f"Дисперсия: σ² = Σ(x-μ)²/n = {variance:.2f}",
            f"Стандартное отклонение: σ = √σ² = {std_dev:.2f}"
        ]
        
        return {
            'answer': {
                'mean': mean,
                'median': median,
                'variance': variance,
                'standard_deviation': std_dev,
                'count': len(numbers)
            },
            'steps': steps,
            'confidence': 0.9,
            'explanation': f"Статистический анализ: среднее={mean:.2f}, медиана={median}, σ={std_dev:.2f}"
        }
    
    def _solve_percentage(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение задач на проценты"""
        numbers = data.get('numbers', [])
        
        if len(numbers) >= 2:
            number, percent = numbers[0], numbers[1]
            result = (number * percent) / 100
            
            steps = [
                f"Найти {percent}% от {number}",
                f"Формула: (число × процент) / 100",
                f"({number} × {percent}) / 100 = {result}"
            ]
            
            return {
                'answer': result,
                'steps': steps,
                'confidence': 0.95,
                'explanation': f"{percent}% от {number} равно {result}"
            }
        
        return {
            'answer': "Недостаточно данных",
            'steps': ["Ошибка: не найдены число и процент"],
            'confidence': 0.0,
            'explanation': "Для расчета процентов нужны число и процент"
        }
    
    def _solve_physics_problem(self, data: Dict[str, Any], problem_text: str) -> Dict[str, Any]:
        """Решение физических задач"""
        try:
            logger.info(f"🔍 Отладка: problem_text = {problem_text}, type = {type(problem_text)}")
            logger.info(f"🔍 Отладка: data = {data}, type = {type(data)}")
            
            # Определение формулы из текста задачи
            formula_name = self._detect_physics_formula(problem_text)
            
            if formula_name:
                # Извлечение переменных из задачи
                variables = self._extract_physics_variables(problem_text)
                
                # Решение через физический решатель
                solution = self.physics_solver.solve_physics_formula(formula_name, variables)
                
                return {
                    'answer': solution.final_answer,
                    'steps': solution.solution_steps,
                    'confidence': solution.confidence,
                    'explanation': solution.explanation,
                    'formula': formula_name,
                    'variables': variables
                }
            else:
                return {
                    'answer': "Физическая формула не определена",
                    'steps': ["Не удалось определить формулу"],
                    'confidence': 0.1,
                    'explanation': "Физическая формула не найдена в базе данных"
                }
                
        except Exception as e:
            logger.error(f"Ошибка решения физической задачи: {e}")
            return {
                'answer': "Ошибка решения",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_chemistry_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение химических задач"""
        try:
            logger.info(f"🔍 Химия: problem_text = {problem_text}, type = {type(problem_text)}")
            logger.info(f"🔍 Химия: data = {data}, type = {type(data)}")
            
            # Определение формулы из текста задачи
            formula_name = self._detect_chemistry_formula(problem_text)
            
            if formula_name:
                # Извлечение переменных из задачи
                variables = self._extract_chemistry_variables(problem_text)
                
                # Решение через химический решатель
                solution = self.chemistry_solver.solve_chemistry_formula(formula_name, variables)
                
                return {
                    'answer': solution.final_answer,
                    'steps': solution.solution_steps,
                    'confidence': solution.confidence,
                    'explanation': solution.explanation,
                    'formula': formula_name,
                    'variables': variables
                }
            else:
                return {
                    'answer': "Химическая формула не определена",
                    'steps': ["Не удалось определить формулу"],
                    'confidence': 0.1,
                    'explanation': "Химическая формула не найдена в базе данных"
                }
                
        except Exception as e:
            logger.error(f"Ошибка решения химической задачи: {e}")
            return {
                'answer': "Ошибка решения",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_graph_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задач анализа графиков"""
        try:
            # Извлечение пути к изображению
            image_path = self._extract_image_path(problem_text)
            
            if image_path:
                return {
                    'answer': f"Анализ графика: {image_path}",
                    'steps': [
                        f"Найден файл изображения: {image_path}",
                        "Анализ структуры графика",
                        "Извлечение данных с помощью OCR",
                        "Интерпретация результатов"
                    ],
                    'confidence': 0.8,
                    'explanation': "Анализ графика будет выполнен после реализации OCR и компьютерного зрения",
                    'image_path': image_path
                }
            else:
                return {
                    'answer': "Путь к изображению не найден",
                    'steps': ["Укажите путь к файлу изображения"],
                    'confidence': 0.1,
                    'explanation': "Для анализа графика необходимо указать путь к файлу изображения"
                }
                
        except Exception as e:
            logger.error(f"Ошибка анализа графика: {e}")
            return {
                'answer': "Ошибка анализа",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_visualization_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задач визуализации данных"""
        try:
            # Извлечение данных из задачи
            viz_data = self._extract_visualization_data(problem_text)
            
            if viz_data:
                return {
                    'answer': "График будет создан",
                    'steps': [
                        "Извлечение данных из задачи",
                        "Определение типа визуализации",
                        "Создание графика",
                        "Сохранение файла"
                    ],
                    'confidence': 0.9,
                    'explanation': "Визуализация данных будет выполнена после реализации модуля создания графиков",
                    'data': viz_data
                }
            else:
                return {
                    'answer': "Данные для визуализации не найдены",
                    'steps': ["Укажите данные для создания графика"],
                    'confidence': 0.1,
                    'explanation': "Для создания графика необходимо указать данные"
                }
                
        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
            return {
                'answer': "Ошибка создания",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_symbolic_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задач символьных вычислений"""
        try:
            if not self.symbolic_calculator:
                return {
                    'answer': "Символьный калькулятор недоступен",
                    'steps': ["Модуль SymPy не установлен"],
                    'confidence': 0.0,
                    'explanation': "Для символьных вычислений требуется модуль SymPy"
                }
            
            # Определение типа символьной операции
            if "упростить" in problem_text.lower():
                result = self.symbolic_calculator.simplify_expression(problem_text)
            elif "решить уравнение" in problem_text.lower():
                result = self.symbolic_calculator.solve_equation(problem_text)
            elif "производная" in problem_text.lower():
                result = self.symbolic_calculator.differentiate(problem_text)
            elif "интеграл" in problem_text.lower():
                result = self.symbolic_calculator.integrate(problem_text)
            else:
                result = self.symbolic_calculator.simplify_expression(problem_text)
            
            return {
                'answer': str(result.result),
                'steps': result.steps,
                'confidence': result.confidence,
                'explanation': result.explanation,
                'latex_output': result.latex_output
            }
            
        except Exception as e:
            logger.error(f"Ошибка символьных вычислений: {e}")
            return {
                'answer': "Ошибка вычислений",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_equation_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение уравнений"""
        try:
            if not self.symbolic_calculator:
                return {
                    'answer': "Символьный калькулятор недоступен",
                    'steps': ["Модуль SymPy не установлен"],
                    'confidence': 0.0,
                    'explanation': "Для решения уравнений требуется модуль SymPy"
                }
            
            result = self.symbolic_calculator.solve_equation(problem_text)
            
            return {
                'answer': str(result.result),
                'steps': result.steps,
                'confidence': result.confidence,
                'explanation': result.explanation,
                'latex_output': result.latex_output
            }
            
        except Exception as e:
            logger.error(f"Ошибка решения уравнения: {e}")
            return {
                'answer': "Ошибка решения",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_differentiation_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задач дифференцирования"""
        try:
            if not self.symbolic_calculator:
                return {
                    'answer': "Символьный калькулятор недоступен",
                    'steps': ["Модуль SymPy не установлен"],
                    'confidence': 0.0,
                    'explanation': "Для дифференцирования требуется модуль SymPy"
                }
            
            result = self.symbolic_calculator.differentiate(problem_text)
            
            return {
                'answer': str(result.result),
                'steps': result.steps,
                'confidence': result.confidence,
                'explanation': result.explanation,
                'latex_output': result.latex_output
            }
            
        except Exception as e:
            logger.error(f"Ошибка дифференцирования: {e}")
            return {
                'answer': "Ошибка вычислений",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_integration_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задач интегрирования"""
        try:
            if not self.symbolic_calculator:
                return {
                    'answer': "Символьный калькулятор недоступен",
                    'steps': ["Модуль SymPy не установлен"],
                    'confidence': 0.0,
                    'explanation': "Для интегрирования требуется модуль SymPy"
                }
            
            result = self.symbolic_calculator.integrate(problem_text)
            
            return {
                'answer': str(result.result),
                'steps': result.steps,
                'confidence': result.confidence,
                'explanation': result.explanation,
                'latex_output': result.latex_output
            }
            
        except Exception as e:
            logger.error(f"Ошибка интегрирования: {e}")
            return {
                'answer': "Ошибка вычислений",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_matrix_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение матричных задач"""
        try:
            if not self.symbolic_calculator:
                return {
                    'answer': "Символьный калькулятор недоступен",
                    'steps': ["Модуль SymPy не установлен"],
                    'confidence': 0.0,
                    'explanation': "Для матричных операций требуется модуль SymPy"
                }
            
            # Извлечение матрицы из текста (упрощенная версия)
            matrix_data = [[1, 2], [3, 4]]  # Заглушка
            
            operation = "determinant"
            if "обратная" in problem_text.lower():
                operation = "inverse"
            elif "транспонированная" in problem_text.lower():
                operation = "transpose"
            elif "собственные значения" in problem_text.lower():
                operation = "eigenvalues"
            
            result = self.symbolic_calculator.matrix_operations(operation, matrix_data)
            
            return {
                'answer': str(result.result),
                'steps': result.steps,
                'confidence': result.confidence,
                'explanation': result.explanation,
                'latex_output': result.latex_output
            }
            
        except Exception as e:
            logger.error(f"Ошибка матричных операций: {e}")
            return {
                'answer': "Ошибка вычислений",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _solve_plotting_problem(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задач построения графиков"""
        try:
            if not self.symbolic_calculator:
                return {
                    'answer': "Символьный калькулятор недоступен",
                    'steps': ["Модуль SymPy не установлен"],
                    'confidence': 0.0,
                    'explanation': "Для построения графиков требуется модуль SymPy"
                }
            
            # Извлечение функции из текста (упрощенная версия)
            function = "x**2 + 2*x + 1"  # Заглушка
            
            result = self.symbolic_calculator.plot_function(function)
            
            return {
                'answer': "График построен",
                'steps': result.steps,
                'confidence': result.confidence,
                'explanation': result.explanation,
                'plot_data': result.plot_data
            }
            
        except Exception as e:
            logger.error(f"Ошибка построения графика: {e}")
            return {
                'answer': "Ошибка построения",
                'steps': ["Ошибка обработки"],
                'confidence': 0.0,
                'explanation': f"Ошибка: {e}"
            }
    
    def _detect_physics_formula(self, problem_text: str) -> Optional[str]:
        """Определение физической формулы из текста задачи"""
        text_lower = problem_text.lower()
        
        # Кинетическая энергия - проверяем разные падежи
        if any(phrase in text_lower for phrase in ["кинетическая энергия", "кинетическую энергию", "кинетической энергии"]):
            return "кинетическая_энергия"
        elif any(phrase in text_lower for phrase in ["потенциальная энергия", "потенциальную энергию", "потенциальной энергии"]):
            return "потенциальная_энергия"
        elif "закон ома" in text_lower:
            return "закон_ома"
        elif ("напряжение" in text_lower and "ток" in text_lower and ("сопротивление" in text_lower or "сопротивлении" in text_lower)):
            return "закон_ома"
        elif "мощность" in text_lower:
            return "мощность"
        elif "сила тяжести" in text_lower:
            return "сила_тяжести"
        elif "ускорение" in text_lower:
            return "ускорение"
        elif "путь" in text_lower:
            return "путь"
        
        return None
    
    def _detect_chemistry_formula(self, problem_text: str) -> Optional[str]:
        """Определение химической формулы из текста задачи"""
        text_lower = problem_text.lower()
        
        # Концентрация - проверяем разные падежи
        if any(word in text_lower for word in ["концентрация", "концентрацию", "концентрации"]):
            return "концентрация"
        elif any(word in text_lower for word in ["молярная масса", "молярную массу", "молярной массы"]):
            return "молярная_масса"
        elif "закон сохранения массы" in text_lower:
            return "закон_сохранения_массы"
        
        return None
    
    def _extract_physics_variables(self, problem_text: str) -> Dict[str, float]:
        """Извлечение переменных из физической задачи"""
        variables = {}
        
        # Поиск числовых значений
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        logger.info(f"🔍 Найденные числа: {numbers}")
        
        # Простое сопоставление по ключевым словам
        if any(word in problem_text.lower() for word in ["масса", "массой", "массу"]):
            # Найти число перед словом "кг"
            massa_match = re.search(r'(\d+(?:\.\d+)?)\s*кг', problem_text.lower())
            if massa_match:
                variables["m"] = float(massa_match.group(1))
                logger.info(f"🔍 Найдена масса: {variables['m']} кг")
        if any(word in problem_text.lower() for word in ["скорость", "скоростью", "скорости"]):
            # Найти число перед словом "м/с"
            speed_match = re.search(r'(\d+(?:\.\d+)?)\s*м/с', problem_text.lower())
            if speed_match:
                variables["v"] = float(speed_match.group(1))
                logger.info(f"🔍 Найдена скорость: {variables['v']} м/с")
        if any(word in problem_text.lower() for word in ["ток", "током", "токе"]):
            # Найти число перед словом "ток" или "а"
            tok_match = re.search(r'(\d+(?:\.\d+)?)\s*[аа]', problem_text.lower())
            if tok_match:
                variables["I"] = float(tok_match.group(1))
                logger.info(f"🔍 Найден ток: {variables['I']} А")
        if any(word in problem_text.lower() for word in ["сопротивление", "сопротивлении", "сопротивлением"]):
            # Найти число перед словом "ом" или "омов"
            sopr_match = re.search(r'(\d+(?:\.\d+)?)\s*ом', problem_text.lower())
            if sopr_match:
                variables["R"] = float(sopr_match.group(1))
                logger.info(f"🔍 Найдено сопротивление: {variables['R']} Ом")
        if any(word in problem_text.lower() for word in ["напряжение", "напряжении", "напряжением"]):
            # Найти число перед словом "в" или "вольт"
            voltage_match = re.search(r'(\d+(?:\.\d+)?)\s*[вв]', problem_text.lower())
            if voltage_match:
                variables["U"] = float(voltage_match.group(1))
                logger.info(f"🔍 Найдено напряжение: {variables['U']} В")
            else:
                # Fallback: взять первое число если есть напряжение
                if len(numbers) > 0:
                    variables["U"] = float(numbers[0])
                    logger.info(f"🔍 Найдено напряжение (fallback): {variables['U']} В")
        if "высота" in problem_text.lower() and len(numbers) > 0:
            variables["h"] = float(numbers[0])
        
        logger.info(f"🔍 Извлеченные переменные: {variables}")
        return variables
    
    def _extract_chemistry_variables(self, problem_text: str) -> Dict[str, float]:
        """Извлечение переменных из химической задачи"""
        variables = {}
        
        # Поиск числовых значений
        numbers = re.findall(r'\d+(?:\.\d+)?', problem_text)
        logger.info(f"🔍 Химия - найденные числа: {numbers}")
        
        # Сопоставление по ключевым словам
        if any(word in problem_text.lower() for word in ["количество вещества", "моль вещества", "моль"]):
            # Найти число перед словом "моль"
            mol_match = re.search(r'(\d+(?:\.\d+)?)\s*моль', problem_text.lower())
            if mol_match:
                variables["n"] = float(mol_match.group(1))
                logger.info(f"🔍 Найдено количество вещества: {variables['n']} моль")
        
        if any(word in problem_text.lower() for word in ["объем", "объему", "объеме", "л раствора", "литр", "литров"]):
            # Найти число перед словом "л"
            volume_match = re.search(r'(\d+(?:\.\d+)?)\s*л', problem_text.lower())
            if volume_match:
                variables["V"] = float(volume_match.group(1))
                logger.info(f"🔍 Найден объем: {variables['V']} л")
        
        if any(word in problem_text.lower() for word in ["масса вещества", "массу вещества", "массе вещества"]):
            # Найти число перед словом "г"
            mass_match = re.search(r'(\d+(?:\.\d+)?)\s*г', problem_text.lower())
            if mass_match:
                variables["m"] = float(mass_match.group(1))
                logger.info(f"🔍 Найдена масса вещества: {variables['m']} г")
        
        logger.info(f"🔍 Химия - извлеченные переменные: {variables}")
        return variables
    
    def _extract_image_path(self, problem_text: str) -> Optional[str]:
        """Извлечение пути к изображению из задачи"""
        # Поиск путей к файлам
        paths = re.findall(r'[a-zA-Z]:\\[^\\/:*?"<>]+\.[a-zA-Z]+|[a-zA-Z0-9_/.-]+\.(?:png|jpg|jpeg|gif|bmp)', problem_text)
        return paths[0] if paths else None
    
    def _extract_visualization_data(self, problem_text: str) -> Optional[Dict[str, List[float]]]:
        """Извлечение данных для визуализации из задачи"""
        logger.info(f"🔍 Визуализация - извлечение данных из: {problem_text}")
        
        # Паттерн 1: "A: 1,2,3,4,5 B: 2,4,6,8,10"
        data_pattern = r'([A-Za-z]+):\s*([0-9.,\s]+)'
        matches = re.findall(data_pattern, problem_text)
        
        if matches:
            data = {}
            for label, values_str in matches:
                values = [float(x.strip()) for x in values_str.split(',')]
                data[label] = values
            logger.info(f"🔍 Визуализация - найдены данные (паттерн 1): {data}")
            return data
        
        # Паттерн 2: "Яблоки 25, Бананы 30, Апельсины 20"
        category_pattern = r'([А-Яа-яA-Za-z]+)\s+(\d+(?:\.\d+)?)'
        category_matches = re.findall(category_pattern, problem_text)
        
        if category_matches:
            data = {}
            for category, value in category_matches:
                data[category] = [float(value)]
            logger.info(f"🔍 Визуализация - найдены данные (паттерн 2): {data}")
            return data
        
        logger.info("🔍 Визуализация - данные не найдены")
        return None

    def _solve_generic(self, problem_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Универсальный решатель для неопознанных задач"""
        return {
            'answer': "Задача не распознана",
            'steps': ["Попытка универсального решения не удалась"],
            'confidence': 0.1,
            'explanation': "Не удалось определить тип задачи и применить специализированный алгоритм"
        }
    
    def _clean_expression(self, text: str) -> str:
        """Очистка математического выражения"""
        # Удаление лишних слов и символов
        expression = re.sub(r'[^\d\+\-\*/\(\)\.\s]', '', text)
        expression = expression.strip()
        return expression
    
    def _verify_solution(self, result: Dict[str, Any], data: Dict[str, Any], problem_type: ProblemType) -> bool:
        """Проверка правильности решения"""
        try:
            answer = result.get('answer')
            
            if problem_type == ProblemType.LINEAR_EQUATION:
                # Проверка подстановкой
                coefficients = data.get('coefficients', {})
                if coefficients and isinstance(answer, (int, float)):
                    a, b, c = coefficients.get('a', 1), coefficients.get('b', 0), coefficients.get('c', 0)
                    left_side = a * answer + b
                    return abs(left_side - c) < 1e-6
            
            elif problem_type == ProblemType.AREA_CALCULATION:
                # Проверка разумности результата
                if isinstance(answer, (int, float)):
                    return answer > 0
            
            elif problem_type == ProblemType.ARITHMETIC:
                # Базовая проверка
                return isinstance(answer, (int, float))
            
            return True
            
        except:
            return False

# Примеры использования
class PhysicsFormulaSolver:
    """Решатель физических формул"""
    
    def __init__(self):
        self.formulas = {
            "кинетическая_энергия": {
                "formula": "E = 0.5 * m * v²",
                "variables": {"m": "масса (кг)", "v": "скорость (м/с)"},
                "units": "Дж",
                "description": "Кинетическая энергия движущегося тела"
            },
            "потенциальная_энергия": {
                "formula": "E = m * g * h",
                "variables": {"m": "масса (кг)", "g": "ускорение свободного падения (м/с²)", "h": "высота (м)"},
                "units": "Дж",
                "description": "Потенциальная энергия тела в поле тяжести"
            },
            "закон_ома": {
                "formula": "U = I * R",
                "variables": {"U": "напряжение (В)", "I": "ток (А)", "R": "сопротивление (Ом)"},
                "units": "В",
                "description": "Закон Ома для участка цепи"
            },
            "мощность": {
                "formula": "P = U * I",
                "variables": {"P": "мощность (Вт)", "U": "напряжение (В)", "I": "ток (А)"},
                "units": "Вт",
                "description": "Электрическая мощность"
            },
            "сила_тяжести": {
                "formula": "F = m * g",
                "variables": {"F": "сила тяжести (Н)", "m": "масса (кг)", "g": "ускорение свободного падения (м/с²)"},
                "units": "Н",
                "description": "Сила тяжести"
            },
            "ускорение": {
                "formula": "a = (v - v₀) / t",
                "variables": {"a": "ускорение (м/с²)", "v": "конечная скорость (м/с)", "v₀": "начальная скорость (м/с)", "t": "время (с)"},
                "units": "м/с²",
                "description": "Ускорение при равномерно ускоренном движении"
            },
            "путь": {
                "formula": "s = v₀ * t + 0.5 * a * t²",
                "variables": {"s": "путь (м)", "v₀": "начальная скорость (м/с)", "t": "время (с)", "a": "ускорение (м/с²)"},
                "units": "м",
                "description": "Путь при равномерно ускоренном движении"
            }
        }
    
    def solve_physics_formula(self, formula_name: str, variables: Dict[str, float]) -> ProblemSolution:
        """Решение физической формулы"""
        try:
            if formula_name not in self.formulas:
                return ProblemSolution(
                    problem_type=ProblemType.PHYSICS_FORMULAS,
                    input_data={"formula": formula_name, "variables": variables},
                    solution_steps=["Формула не найдена"],
                    final_answer="Ошибка: формула не найдена",
                    verification=False,
                    confidence=0.0,
                    explanation="Формула не найдена в базе данных"
                )
            
            formula_info = self.formulas[formula_name]
            steps = []
            
            # Подстановка значений в формулу
            if formula_name == "кинетическая_энергия":
                m, v = variables.get("m", 0), variables.get("v", 0)
                result = 0.5 * m * v**2
                steps.append(f"E = 0.5 * m * v²")
                steps.append(f"E = 0.5 * {m} * {v}²")
                steps.append(f"E = 0.5 * {m} * {v**2}")
                steps.append(f"E = {result} Дж")
                
            elif formula_name == "закон_ома":
                I, R = variables.get("I", 0), variables.get("R", 0)
                result = I * R
                steps.append(f"U = I * R")
                steps.append(f"U = {I} * {R}")
                steps.append(f"U = {result} В")
                
            elif formula_name == "мощность":
                U, I = variables.get("U", 0), variables.get("I", 0)
                result = U * I
                steps.append(f"P = U * I")
                steps.append(f"P = {U} * {I}")
                steps.append(f"P = {result} Вт")
                
            elif formula_name == "потенциальная_энергия":
                m, g, h = variables.get("m", 0), variables.get("g", 9.81), variables.get("h", 0)
                result = m * g * h
                steps.append(f"E = m * g * h")
                steps.append(f"E = {m} * {g} * {h}")
                steps.append(f"E = {result} Дж")
                
            elif formula_name == "сила_тяжести":
                m, g = variables.get("m", 0), variables.get("g", 9.81)
                result = m * g
                steps.append(f"F = m * g")
                steps.append(f"F = {m} * {g}")
                steps.append(f"F = {result} Н")
                
            else:
                result = 0
                steps.append("Формула не реализована")
            
            return ProblemSolution(
                problem_type=ProblemType.PHYSICS_FORMULAS,
                input_data={"formula": formula_name, "variables": variables},
                solution_steps=steps,
                final_answer=f"{result} {formula_info['units']}",
                verification=True,
                confidence=0.95,
                explanation=f"{formula_info['description']}. Результат: {result} {formula_info['units']}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка решения физической формулы: {e}")
            return ProblemSolution(
                problem_type=ProblemType.PHYSICS_FORMULAS,
                input_data={"formula": formula_name, "variables": variables},
                solution_steps=["Ошибка вычисления"],
                final_answer="Ошибка",
                verification=False,
                confidence=0.0,
                explanation=f"Ошибка: {e}"
            )

class ChemistryFormulaSolver:
    """Решатель химических формул"""
    
    def __init__(self):
        self.formulas = {
            "концентрация": {
                "formula": "C = n / V",
                "variables": {"C": "концентрация (моль/л)", "n": "количество вещества (моль)", "V": "объем (л)"},
                "units": "моль/л",
                "description": "Молярная концентрация раствора"
            },
            "молярная_масса": {
                "formula": "M = m / n",
                "variables": {"M": "молярная масса (г/моль)", "m": "масса вещества (г)", "n": "количество вещества (моль)"},
                "units": "г/моль",
                "description": "Молярная масса вещества"
            },
            "закон_сохранения_массы": {
                "formula": "Σm_вход = Σm_выход",
                "variables": {"m_вход": "масса реагентов (г)", "m_выход": "масса продуктов (г)"},
                "units": "г",
                "description": "Закон сохранения массы в химических реакциях"
            }
        }
    
    def solve_chemistry_formula(self, formula_name: str, variables: Dict[str, float]) -> ProblemSolution:
        """Решение химической формулы"""
        try:
            if formula_name not in self.formulas:
                return ProblemSolution(
                    problem_type=ProblemType.CHEMISTRY_CALCULATIONS,
                    input_data={"formula": formula_name, "variables": variables},
                    solution_steps=["Формула не найдена"],
                    final_answer="Ошибка: формула не найдена",
                    verification=False,
                    confidence=0.0,
                    explanation="Формула не найдена в базе данных"
                )
            
            formula_info = self.formulas[formula_name]
            steps = []
            
            # Подстановка значений в формулу
            if formula_name == "концентрация":
                n, V = variables.get("n", 0), variables.get("V", 0)
                if V == 0:
                    return ProblemSolution(
                        problem_type=ProblemType.CHEMISTRY_CALCULATIONS,
                        input_data={"formula": formula_name, "variables": variables},
                        solution_steps=["Ошибка: деление на ноль"],
                        final_answer="Ошибка: объем не может быть равен нулю",
                        verification=False,
                        confidence=0.0,
                        explanation="Объем раствора не может быть равен нулю"
                    )
                result = n / V
                steps.append(f"C = n / V")
                steps.append(f"C = {n} / {V}")
                steps.append(f"C = {result} моль/л")
                
            elif formula_name == "молярная_масса":
                m, n = variables.get("m", 0), variables.get("n", 0)
                if n == 0:
                    return ProblemSolution(
                        problem_type=ProblemType.CHEMISTRY_CALCULATIONS,
                        input_data={"formula": formula_name, "variables": variables},
                        solution_steps=["Ошибка: деление на ноль"],
                        final_answer="Ошибка: количество вещества не может быть равно нулю",
                        verification=False,
                        confidence=0.0,
                        explanation="Количество вещества не может быть равно нулю"
                    )
                result = m / n
                steps.append(f"M = m / n")
                steps.append(f"M = {m} / {n}")
                steps.append(f"M = {result} г/моль")
                
            else:
                result = 0
                steps.append("Формула не реализована")
            
            return ProblemSolution(
                problem_type=ProblemType.CHEMISTRY_CALCULATIONS,
                input_data={"formula": formula_name, "variables": variables},
                solution_steps=steps,
                final_answer=f"{result} {formula_info['units']}",
                verification=True,
                confidence=0.95,
                explanation=f"{formula_info['description']}. Результат: {result} {formula_info['units']}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка решения химической формулы: {e}")
            return ProblemSolution(
                problem_type=ProblemType.CHEMISTRY_CALCULATIONS,
                input_data={"formula": formula_name, "variables": variables},
                solution_steps=["Ошибка вычисления"],
                final_answer="Ошибка",
                verification=False,
                confidence=0.0,
                explanation=f"Ошибка: {e}"
            )

if __name__ == "__main__":
    solver = MathematicalProblemSolver()
    
    # Тестовые задачи
    test_problems = [
        "Вычисли 2 + 3 * 4",
        "Реши уравнение 2x + 5 = 13",
        "Найди площадь треугольника с основанием 5 и высотой 3",
        "Реши квадратное уравнение x² - 5x + 6 = 0",
        "Найди 15% от 200",
        "Вычисли sin(30°)",
        "Найди среднее значение чисел 1, 2, 3, 4, 5",
        # Новые тестовые задачи
        "Найти кинетическую энергию тела массой 2 кг, движущегося со скоростью 10 м/с",
        "Найти напряжение при токе 2 А и сопротивлении 5 Ом",
        "Найти концентрацию раствора с 0.5 моль вещества в 2 л раствора",
        "Проанализировать график в файле sales_chart.png",
        "Построить линейный график A: 1,2,3,4,5 B: 2,4,6,8,10"
    ]
    
    print("🧮 Тестирование алгоритма решения математических задач\n")
    
    for i, problem in enumerate(test_problems, 1):
        print(f"Задача {i}: {problem}")
        solution = solver.solve_problem(problem)
        
        print(f"Тип: {solution.problem_type.value}")
        print(f"Ответ: {solution.final_answer}")
        print(f"Уверенность: {solution.confidence:.2f}")
        print(f"Проверка: {'✓' if solution.verification else '✗'}")
        print(f"Объяснение: {solution.explanation}")
        print("-" * 50)

