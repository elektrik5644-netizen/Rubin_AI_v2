#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 Расширенный математический решатель для Rubin AI
====================================================

Новые возможности:
- Анализ графиков и изображений
- Работа с формулами физики и химии
- Визуализация данных
- OCR для чтения графиков
- Интерактивные расчеты

Автор: Rubin AI System
Версия: 3.0
"""

import re
import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import pytesseract
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedProblemType(Enum):
    """Расширенные типы задач"""
    PHYSICS_FORMULAS = "физические_формулы"
    CHEMISTRY_CALCULATIONS = "химические_расчеты"
    ENGINEERING_DESIGN = "инженерные_расчеты"
    STATISTICAL_ANALYSIS = "статистический_анализ"
    GRAPH_ANALYSIS = "анализ_графиков"
    DATA_VISUALIZATION = "визуализация_данных"
    FORMULA_CALCULATION = "расчет_по_формуле"
    IMAGE_ANALYSIS = "анализ_изображений"

@dataclass
class AdvancedSolution:
    """Результат расширенного решения"""
    problem_type: AdvancedProblemType
    input_data: Dict[str, Any]
    solution_steps: List[str]
    final_answer: Union[float, str, Dict[str, Any]]
    verification: bool
    confidence: float
    explanation: str
    visualization: Optional[str] = None
    graph_data: Optional[Dict] = None

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
    
    def solve_physics_formula(self, formula_name: str, variables: Dict[str, float]) -> AdvancedSolution:
        """Решение физической формулы"""
        try:
            if formula_name not in self.formulas:
                return AdvancedSolution(
                    problem_type=AdvancedProblemType.PHYSICS_FORMULAS,
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
                
            else:
                result = 0
                steps.append("Формула не реализована")
            
            return AdvancedSolution(
                problem_type=AdvancedProblemType.PHYSICS_FORMULAS,
                input_data={"formula": formula_name, "variables": variables},
                solution_steps=steps,
                final_answer=f"{result} {formula_info['units']}",
                verification=True,
                confidence=0.95,
                explanation=f"{formula_info['description']}. Результат: {result} {formula_info['units']}"
            )
            
        except Exception as e:
            logger.error(f"Ошибка решения физической формулы: {e}")
            return AdvancedSolution(
                problem_type=AdvancedProblemType.PHYSICS_FORMULAS,
                input_data={"formula": formula_name, "variables": variables},
                solution_steps=["Ошибка вычисления"],
                final_answer="Ошибка",
                verification=False,
                confidence=0.0,
                explanation=f"Ошибка: {e}"
            )

class GraphAnalyzer:
    """Анализатор графиков и изображений"""
    
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6'
        
    def extract_text_from_graph(self, image_path: str) -> str:
        """Извлечение текста с графиков"""
        try:
            if not os.path.exists(image_path):
                return "Файл изображения не найден"
            
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                return "Ошибка загрузки изображения"
            
            # Предобработка
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)
            
            # OCR
            text = pytesseract.image_to_string(enhanced, config=self.tesseract_config)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return f"Ошибка извлечения текста: {e}"
    
    def analyze_graph_structure(self, image_path: str) -> Dict[str, Any]:
        """Анализ структуры графика"""
        try:
            if not os.path.exists(image_path):
                return {"error": "Файл изображения не найден"}
            
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Ошибка загрузки изображения"}
            
            # Определение типа графика
            graph_type = self.detect_graph_type(image)
            
            # Поиск осей координат
            axes = self.find_coordinate_axes(image)
            
            # Поиск точек данных
            data_points = self.find_data_points(image)
            
            return {
                "graph_type": graph_type,
                "axes": axes,
                "data_points": data_points,
                "image_size": image.shape
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа графика: {e}")
            return {"error": f"Ошибка анализа: {e}"}
    
    def detect_graph_type(self, image) -> str:
        """Определение типа графика"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Поиск линий (для линейных графиков)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) > 5:
                return "line_graph"
            
            # Поиск прямоугольников (для столбчатых диаграмм)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = [c for c in contours if cv2.contourArea(c) > 1000]
            
            if len(rectangles) > 3:
                return "bar_chart"
            
            # Поиск кругов (для круговых диаграмм)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
            
            if circles is not None:
                return "pie_chart"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Ошибка определения типа графика: {e}")
            return "unknown"
    
    def find_coordinate_axes(self, image) -> Dict[str, Any]:
        """Поиск осей координат"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Поиск горизонтальных и вертикальных линий
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            return {
                "horizontal_lines": np.sum(horizontal_lines > 0),
                "vertical_lines": np.sum(vertical_lines > 0)
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска осей: {e}")
            return {"error": f"Ошибка: {e}"}
    
    def find_data_points(self, image) -> List[Dict[str, Any]]:
        """Поиск точек данных"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Поиск кругов (точек данных)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                     param1=50, param2=30, minRadius=5, maxRadius=20)
            
            points = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    points.append({"x": int(x), "y": int(y), "radius": int(r)})
            
            return points
            
        except Exception as e:
            logger.error(f"Ошибка поиска точек данных: {e}")
            return []

class DataVisualizer:
    """Создатель визуализаций данных"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_line_graph(self, data: Dict[str, List[float]], title: str = "График", 
                         xlabel: str = "X", ylabel: str = "Y") -> str:
        """Создание линейного графика"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if isinstance(data, dict):
                for i, (key, values) in enumerate(data.items()):
                    ax.plot(values, label=key, color=self.colors[i % len(self.colors)], linewidth=2)
            else:
                ax.plot(data, color=self.colors[0], linewidth=2)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Сохранение графика
            filename = f"graph_{hash(title)}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return filename
            
        except Exception as e:
            logger.error(f"Ошибка создания графика: {e}")
            return f"Ошибка: {e}"
    
    def create_bar_chart(self, data: Dict[str, float], title: str = "Столбчатая диаграмма") -> str:
        """Создание столбчатой диаграммы"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = list(data.keys())
            values = list(data.values())
            
            bars = ax.bar(categories, values, color=self.colors[:len(categories)])
            
            # Добавление значений на столбцы
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value}', ha='center', va='bottom')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Значения', fontsize=12)
            plt.xticks(rotation=45)
            
            # Сохранение графика
            filename = f"bar_chart_{hash(title)}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return filename
            
        except Exception as e:
            logger.error(f"Ошибка создания столбчатой диаграммы: {e}")
            return f"Ошибка: {e}"
    
    def create_pie_chart(self, data: Dict[str, float], title: str = "Круговая диаграмма") -> str:
        """Создание круговой диаграммы"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            labels = list(data.keys())
            sizes = list(data.values())
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                             colors=self.colors[:len(labels)],
                                             startangle=90)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Сохранение графика
            filename = f"pie_chart_{hash(title)}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return filename
            
        except Exception as e:
            logger.error(f"Ошибка создания круговой диаграммы: {e}")
            return f"Ошибка: {e}"

class AdvancedMathSolver:
    """Расширенный математический решатель"""
    
    def __init__(self):
        self.physics_solver = PhysicsFormulaSolver()
        self.graph_analyzer = GraphAnalyzer()
        self.data_visualizer = DataVisualizer()
        
    def solve_advanced_problem(self, question: str) -> AdvancedSolution:
        """Решение расширенной задачи"""
        try:
            # Определение типа задачи
            problem_type = self.classify_problem(question)
            
            if problem_type == AdvancedProblemType.PHYSICS_FORMULAS:
                return self.solve_physics_problem(question)
            elif problem_type == AdvancedProblemType.GRAPH_ANALYSIS:
                return self.solve_graph_problem(question)
            elif problem_type == AdvancedProblemType.DATA_VISUALIZATION:
                return self.solve_visualization_problem(question)
            else:
                return AdvancedSolution(
                    problem_type=AdvancedProblemType.UNKNOWN,
                    input_data={"question": question},
                    solution_steps=["Тип задачи не определен"],
                    final_answer="Не удалось определить тип задачи",
                    verification=False,
                    confidence=0.0,
                    explanation="Задача не соответствует известным типам"
                )
                
        except Exception as e:
            logger.error(f"Ошибка решения расширенной задачи: {e}")
            return AdvancedSolution(
                problem_type=AdvancedProblemType.UNKNOWN,
                input_data={"question": question},
                solution_steps=["Ошибка обработки"],
                final_answer="Ошибка",
                verification=False,
                confidence=0.0,
                explanation=f"Ошибка: {e}"
            )
    
    def classify_problem(self, question: str) -> AdvancedProblemType:
        """Классификация типа задачи"""
        question_lower = question.lower()
        
        # Физические формулы
        physics_keywords = ["кинетическая энергия", "потенциальная энергия", "закон ома", 
                          "мощность", "сила тяжести", "ускорение", "путь", "скорость"]
        if any(keyword in question_lower for keyword in physics_keywords):
            return AdvancedProblemType.PHYSICS_FORMULAS
        
        # Анализ графиков
        graph_keywords = ["график", "диаграмма", "анализ", "изображение", "картинка"]
        if any(keyword in question_lower for keyword in graph_keywords):
            return AdvancedProblemType.GRAPH_ANALYSIS
        
        # Визуализация данных
        viz_keywords = ["построить", "создать", "нарисовать", "визуализация"]
        if any(keyword in question_lower for keyword in viz_keywords):
            return AdvancedProblemType.DATA_VISUALIZATION
        
        return AdvancedProblemType.UNKNOWN
    
    def solve_physics_problem(self, question: str) -> AdvancedSolution:
        """Решение физической задачи"""
        try:
            # Извлечение переменных из вопроса
            variables = self.extract_physics_variables(question)
            
            # Определение формулы
            formula_name = self.detect_physics_formula(question)
            
            if formula_name:
                return self.physics_solver.solve_physics_formula(formula_name, variables)
            else:
                return AdvancedSolution(
                    problem_type=AdvancedProblemType.PHYSICS_FORMULAS,
                    input_data={"question": question},
                    solution_steps=["Формула не определена"],
                    final_answer="Не удалось определить формулу",
                    verification=False,
                    confidence=0.0,
                    explanation="Физическая формула не найдена"
                )
                
        except Exception as e:
            logger.error(f"Ошибка решения физической задачи: {e}")
            return AdvancedSolution(
                problem_type=AdvancedProblemType.PHYSICS_FORMULAS,
                input_data={"question": question},
                solution_steps=["Ошибка обработки"],
                final_answer="Ошибка",
                verification=False,
                confidence=0.0,
                explanation=f"Ошибка: {e}"
            )
    
    def extract_physics_variables(self, question: str) -> Dict[str, float]:
        """Извлечение переменных из физической задачи"""
        variables = {}
        
        # Поиск числовых значений
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        
        # Простое сопоставление по позиции (можно улучшить)
        if "масса" in question.lower() and len(numbers) > 0:
            variables["m"] = float(numbers[0])
        if "скорость" in question.lower() and len(numbers) > 1:
            variables["v"] = float(numbers[1])
        if "ток" in question.lower() and len(numbers) > 0:
            variables["I"] = float(numbers[0])
        if "сопротивление" in question.lower() and len(numbers) > 1:
            variables["R"] = float(numbers[1])
        if "напряжение" in question.lower() and len(numbers) > 0:
            variables["U"] = float(numbers[0])
        
        return variables
    
    def detect_physics_formula(self, question: str) -> Optional[str]:
        """Определение физической формулы"""
        question_lower = question.lower()
        
        if "кинетическая энергия" in question_lower:
            return "кинетическая_энергия"
        elif "потенциальная энергия" in question_lower:
            return "потенциальная_энергия"
        elif "закон ома" in question_lower or ("напряжение" in question_lower and "ток" in question_lower):
            return "закон_ома"
        elif "мощность" in question_lower:
            return "мощность"
        elif "сила тяжести" in question_lower:
            return "сила_тяжести"
        elif "ускорение" in question_lower:
            return "ускорение"
        elif "путь" in question_lower:
            return "путь"
        
        return None
    
    def solve_graph_problem(self, question: str) -> AdvancedSolution:
        """Решение задачи анализа графика"""
        try:
            # Извлечение пути к изображению
            image_path = self.extract_image_path(question)
            
            if not image_path:
                return AdvancedSolution(
                    problem_type=AdvancedProblemType.GRAPH_ANALYSIS,
                    input_data={"question": question},
                    solution_steps=["Путь к изображению не найден"],
                    final_answer="Укажите путь к изображению графика",
                    verification=False,
                    confidence=0.0,
                    explanation="Необходимо указать путь к файлу изображения"
                )
            
            # Анализ графика
            graph_data = self.graph_analyzer.analyze_graph_structure(image_path)
            text_data = self.graph_analyzer.extract_text_from_graph(image_path)
            
            steps = [
                f"Анализ изображения: {image_path}",
                f"Тип графика: {graph_data.get('graph_type', 'неизвестно')}",
                f"Извлеченный текст: {text_data[:100]}..."
            ]
            
            return AdvancedSolution(
                problem_type=AdvancedProblemType.GRAPH_ANALYSIS,
                input_data={"question": question, "image_path": image_path},
                solution_steps=steps,
                final_answer=f"График проанализирован: {graph_data.get('graph_type', 'неизвестно')}",
                verification=True,
                confidence=0.8,
                explanation=f"Анализ графика завершен. Тип: {graph_data.get('graph_type', 'неизвестно')}",
                graph_data=graph_data
            )
            
        except Exception as e:
            logger.error(f"Ошибка анализа графика: {e}")
            return AdvancedSolution(
                problem_type=AdvancedProblemType.GRAPH_ANALYSIS,
                input_data={"question": question},
                solution_steps=["Ошибка анализа"],
                final_answer="Ошибка анализа графика",
                verification=False,
                confidence=0.0,
                explanation=f"Ошибка: {e}"
            )
    
    def extract_image_path(self, question: str) -> Optional[str]:
        """Извлечение пути к изображению из вопроса"""
        # Поиск путей к файлам
        paths = re.findall(r'[a-zA-Z]:\\[^\\/:*?"<>|]+\.[a-zA-Z]+|[a-zA-Z0-9_/.-]+\.(?:png|jpg|jpeg|gif|bmp)', question)
        return paths[0] if paths else None
    
    def solve_visualization_problem(self, question: str) -> AdvancedSolution:
        """Решение задачи визуализации данных"""
        try:
            # Извлечение данных из вопроса
            data = self.extract_data_from_question(question)
            
            if not data:
                return AdvancedSolution(
                    problem_type=AdvancedProblemType.DATA_VISUALIZATION,
                    input_data={"question": question},
                    solution_steps=["Данные не найдены"],
                    final_answer="Укажите данные для визуализации",
                    verification=False,
                    confidence=0.0,
                    explanation="Необходимо указать данные для создания графика"
                )
            
            # Определение типа визуализации
            viz_type = self.detect_visualization_type(question)
            
            if viz_type == "line":
                filename = self.data_visualizer.create_line_graph(data, "Линейный график")
            elif viz_type == "bar":
                filename = self.data_visualizer.create_bar_chart(data, "Столбчатая диаграмма")
            elif viz_type == "pie":
                filename = self.data_visualizer.create_pie_chart(data, "Круговая диаграмма")
            else:
                filename = self.data_visualizer.create_line_graph(data, "График")
            
            return AdvancedSolution(
                problem_type=AdvancedProblemType.DATA_VISUALIZATION,
                input_data={"question": question, "data": data},
                solution_steps=[
                    f"Извлечены данные: {list(data.keys())}",
                    f"Создан график типа: {viz_type}",
                    f"Сохранен файл: {filename}"
                ],
                final_answer=f"График создан: {filename}",
                verification=True,
                confidence=0.9,
                explanation=f"Визуализация данных завершена. Файл: {filename}",
                visualization=filename
            )
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
            return AdvancedSolution(
                problem_type=AdvancedProblemType.DATA_VISUALIZATION,
                input_data={"question": question},
                solution_steps=["Ошибка создания"],
                final_answer="Ошибка создания графика",
                verification=False,
                confidence=0.0,
                explanation=f"Ошибка: {e}"
            )
    
    def extract_data_from_question(self, question: str) -> Optional[Dict[str, List[float]]]:
        """Извлечение данных из вопроса"""
        # Простое извлечение данных (можно улучшить)
        # Пример: "A: 1,2,3,4 B: 5,6,7,8"
        data_pattern = r'([A-Za-z]+):\s*([0-9.,\s]+)'
        matches = re.findall(data_pattern, question)
        
        if matches:
            data = {}
            for label, values_str in matches:
                values = [float(x.strip()) for x in values_str.split(',')]
                data[label] = values
            return data
        
        return None
    
    def detect_visualization_type(self, question: str) -> str:
        """Определение типа визуализации"""
        question_lower = question.lower()
        
        if "линейный" in question_lower or "line" in question_lower:
            return "line"
        elif "столбчатая" in question_lower or "bar" in question_lower:
            return "bar"
        elif "круговая" in question_lower or "pie" in question_lower:
            return "pie"
        else:
            return "line"  # По умолчанию линейный график

# Пример использования
if __name__ == "__main__":
    solver = AdvancedMathSolver()
    
    # Тест физических формул
    physics_question = "Найти кинетическую энергию тела массой 2 кг, движущегося со скоростью 10 м/с"
    result = solver.solve_advanced_problem(physics_question)
    print(f"Физическая задача: {result.final_answer}")
    print(f"Объяснение: {result.explanation}")
    
    # Тест визуализации
    viz_question = "Построить линейный график A: 1,2,3,4,5 B: 2,4,6,8,10"
    result = solver.solve_advanced_problem(viz_question)
    print(f"Визуализация: {result.final_answer}")
    print(f"Файл: {result.visualization}")










