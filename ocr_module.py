#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Module для Rubin AI
Модуль оптического распознавания символов для анализа графиков и изображений
"""

import logging
import json
import re
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import base64
import io

# Попытка импорта OCR библиотек с fallback
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    # Mock классы для работы без OCR библиотек
    class cv2:
        @staticmethod
        def imread(path):
            return None
        
        @staticmethod
        def cvtColor(img, code):
            return img
        
        @staticmethod
        def threshold(img, thresh, maxval, type):
            return None, img
        
        @staticmethod
        def imwrite(path, img):
            return True
    
    class np:
        @staticmethod
        def array(data):
            return data
    
    class Image:
        @staticmethod
        def open(path):
            return None
    
    class pytesseract:
        @staticmethod
        def image_to_string(img, lang='rus'):
            return "Mock OCR text"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubinOCRModule:
    """Модуль OCR для Rubin AI"""
    
    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        self.supported_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        self.languages = ['rus', 'eng']
        
        logger.info(f"🔍 OCR Module инициализирован")
        logger.info(f"🔧 OCR библиотеки доступны: {self.ocr_available}")
    
    def extract_text_from_image(self, image_path: str, language: str = 'rus') -> Dict[str, Any]:
        """Извлечение текста из изображения"""
        try:
            if not self.ocr_available:
                return self._mock_text_extraction(image_path)
            
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Не удалось загрузить изображение',
                    'image_path': image_path
                }
            
            # Преобразуем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Применяем пороговую обработку
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Извлекаем текст
            text = pytesseract.image_to_string(thresh, lang=language)
            
            # Очищаем текст
            cleaned_text = self._clean_extracted_text(text)
            
            return {
                'success': True,
                'text': cleaned_text,
                'original_text': text,
                'image_path': image_path,
                'language': language,
                'confidence': self._calculate_confidence(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения текста: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def _mock_text_extraction(self, image_path: str) -> Dict[str, Any]:
        """Mock извлечение текста для тестирования"""
        mock_texts = {
            'graph': "График функции y = x^2\nТочки: (0,0), (1,1), (2,4), (3,9)",
            'chart': "Диаграмма продаж\nЯнварь: 100\nФевраль: 150\nМарт: 200",
            'formula': "E = mc^2\nF = ma\nU = IR",
            'circuit': "Схема электрической цепи\nR1 = 10 Ом\nR2 = 20 Ом\nU = 220 В"
        }
        
        # Определяем тип изображения по имени файла
        filename = image_path.lower()
        for key, text in mock_texts.items():
            if key in filename:
                return {
                    'success': True,
                    'text': text,
                    'original_text': text,
                    'image_path': image_path,
                    'language': 'rus',
                    'confidence': 0.8,
                    'mock': True
                }
        
        return {
            'success': True,
            'text': "Извлеченный текст из изображения",
            'original_text': "Extracted text from image",
            'image_path': image_path,
            'language': 'rus',
            'confidence': 0.5,
            'mock': True
        }
    
    def _clean_extracted_text(self, text: str) -> str:
        """Очистка извлеченного текста"""
        # Удаляем лишние пробелы и переносы строк
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Исправляем распространенные OCR ошибки
        corrections = {
            '0': 'O',  # Ноль может быть распознан как буква O
            '1': 'I',  # Единица может быть распознана как буква I
            '5': 'S',  # Пятерка может быть распознана как буква S
            '8': 'B',  # Восьмерка может быть распознана как буква B
        }
        
        # Применяем коррекции только в контексте формул
        if any(char in cleaned for char in ['=', '+', '-', '*', '/', '^']):
            for wrong, correct in corrections.items():
                cleaned = cleaned.replace(wrong, correct)
        
        return cleaned
    
    def _calculate_confidence(self, text: str) -> float:
        """Расчет уверенности в извлеченном тексте"""
        if not text:
            return 0.0
        
        # Простая эвристика для расчета уверенности
        confidence = 0.5
        
        # Увеличиваем уверенность для математических выражений
        if re.search(r'[0-9]', text):
            confidence += 0.2
        
        if re.search(r'[+\-*/=]', text):
            confidence += 0.2
        
        if re.search(r'[a-zA-Z]', text):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def analyze_graph(self, image_path: str) -> Dict[str, Any]:
        """Анализ графика на изображении"""
        try:
            # Извлекаем текст из изображения
            text_result = self.extract_text_from_image(image_path)
            
            if not text_result['success']:
                return text_result
            
            text = text_result['text']
            
            # Анализируем график
            analysis = {
                'graph_type': self._detect_graph_type(text),
                'function': self._extract_function(text),
                'data_points': self._extract_data_points(text),
                'axes_labels': self._extract_axes_labels(text),
                'title': self._extract_title(text),
                'mathematical_content': self._extract_mathematical_content(text)
            }
            
            return {
                'success': True,
                'image_path': image_path,
                'extracted_text': text,
                'analysis': analysis,
                'confidence': text_result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа графика: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def _detect_graph_type(self, text: str) -> str:
        """Определение типа графика"""
        text_lower = text.lower()
        
        if 'sin' in text_lower or 'cos' in text_lower or 'tan' in text_lower:
            return 'trigonometric'
        elif '^' in text or '**' in text or 'степень' in text_lower:
            return 'polynomial'
        elif 'log' in text_lower or 'ln' in text_lower:
            return 'logarithmic'
        elif 'exp' in text_lower or 'e^' in text:
            return 'exponential'
        elif 'диаграмма' in text_lower or 'chart' in text_lower:
            return 'chart'
        elif 'схема' in text_lower or 'circuit' in text_lower:
            return 'circuit'
        else:
            return 'unknown'
    
    def _extract_function(self, text: str) -> Optional[str]:
        """Извлечение функции из текста"""
        # Поиск функций вида y = f(x)
        function_patterns = [
            r'y\s*=\s*([^,\n]+)',
            r'f\(x\)\s*=\s*([^,\n]+)',
            r'функция\s*:\s*([^,\n]+)',
            r'function\s*:\s*([^,\n]+)'
        ]
        
        for pattern in function_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_data_points(self, text: str) -> List[Tuple[float, float]]:
        """Извлечение точек данных"""
        points = []
        
        # Поиск точек в формате (x, y)
        point_pattern = r'\(([0-9.-]+),\s*([0-9.-]+)\)'
        matches = re.findall(point_pattern, text)
        
        for x_str, y_str in matches:
            try:
                x = float(x_str)
                y = float(y_str)
                points.append((x, y))
            except ValueError:
                continue
        
        return points
    
    def _extract_axes_labels(self, text: str) -> Dict[str, str]:
        """Извлечение подписей осей"""
        labels = {}
        
        # Поиск подписей осей
        x_pattern = r'x\s*[=:]\s*([^,\n]+)'
        y_pattern = r'y\s*[=:]\s*([^,\n]+)'
        
        x_match = re.search(x_pattern, text, re.IGNORECASE)
        y_match = re.search(y_pattern, text, re.IGNORECASE)
        
        if x_match:
            labels['x'] = x_match.group(1).strip()
        if y_match:
            labels['y'] = y_match.group(1).strip()
        
        return labels
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Извлечение заголовка"""
        # Поиск заголовка в первой строке
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) > 3 and not re.match(r'^[0-9\s\-=]+$', first_line):
                return first_line
        
        return None
    
    def _extract_mathematical_content(self, text: str) -> Dict[str, Any]:
        """Извлечение математического содержимого"""
        content = {
            'formulas': [],
            'equations': [],
            'numbers': [],
            'variables': []
        }
        
        # Поиск формул
        formula_pattern = r'([A-Za-z]\s*[=]\s*[^,\n]+)'
        formulas = re.findall(formula_pattern, text)
        content['formulas'] = formulas
        
        # Поиск уравнений
        equation_pattern = r'([^,\n]*[=][^,\n]*)'
        equations = re.findall(equation_pattern, text)
        content['equations'] = equations
        
        # Поиск чисел
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, text)
        content['numbers'] = [float(n) for n in numbers]
        
        # Поиск переменных
        variable_pattern = r'\b[A-Za-z]\b'
        variables = re.findall(variable_pattern, text)
        content['variables'] = list(set(variables))
        
        return content
    
    def analyze_circuit_diagram(self, image_path: str) -> Dict[str, Any]:
        """Анализ схемы электрической цепи"""
        try:
            text_result = self.extract_text_from_image(image_path)
            
            if not text_result['success']:
                return text_result
            
            text = text_result['text']
            
            # Анализ схемы
            analysis = {
                'components': self._extract_circuit_components(text),
                'connections': self._extract_circuit_connections(text),
                'values': self._extract_circuit_values(text),
                'circuit_type': self._detect_circuit_type(text)
            }
            
            return {
                'success': True,
                'image_path': image_path,
                'extracted_text': text,
                'analysis': analysis,
                'confidence': text_result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа схемы: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def _extract_circuit_components(self, text: str) -> List[str]:
        """Извлечение компонентов схемы"""
        components = []
        
        component_patterns = [
            r'R\d+\s*=\s*([0-9.]+)\s*Ом',
            r'C\d+\s*=\s*([0-9.]+)\s*Ф',
            r'L\d+\s*=\s*([0-9.]+)\s*Гн',
            r'U\d+\s*=\s*([0-9.]+)\s*В',
            r'I\d+\s*=\s*([0-9.]+)\s*А'
        ]
        
        for pattern in component_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            components.extend(matches)
        
        return components
    
    def _extract_circuit_connections(self, text: str) -> List[str]:
        """Извлечение соединений схемы"""
        connections = []
        
        connection_patterns = [
            r'последовательно',
            r'параллельно',
            r'series',
            r'parallel'
        ]
        
        for pattern in connection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                connections.append(pattern)
        
        return connections
    
    def _extract_circuit_values(self, text: str) -> Dict[str, float]:
        """Извлечение значений схемы"""
        values = {}
        
        value_patterns = {
            'resistance': r'R\s*=\s*([0-9.]+)\s*Ом',
            'voltage': r'U\s*=\s*([0-9.]+)\s*В',
            'current': r'I\s*=\s*([0-9.]+)\s*А',
            'power': r'P\s*=\s*([0-9.]+)\s*Вт'
        }
        
        for key, pattern in value_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values[key] = float(match.group(1))
        
        return values
    
    def _detect_circuit_type(self, text: str) -> str:
        """Определение типа схемы"""
        text_lower = text.lower()
        
        if 'последовательно' in text_lower or 'series' in text_lower:
            return 'series'
        elif 'параллельно' in text_lower or 'parallel' in text_lower:
            return 'parallel'
        elif 'мост' in text_lower or 'bridge' in text_lower:
            return 'bridge'
        else:
            return 'unknown'
    
    def get_module_info(self) -> Dict[str, Any]:
        """Получение информации о модуле"""
        return {
            'name': 'Rubin OCR Module',
            'version': '1.0',
            'ocr_available': self.ocr_available,
            'supported_formats': self.supported_formats,
            'languages': self.languages,
            'features': [
                'text_extraction',
                'graph_analysis',
                'circuit_analysis',
                'mathematical_content_extraction'
            ]
        }

def main():
    """Основная функция для тестирования OCR модуля"""
    print("🔍 ТЕСТИРОВАНИЕ OCR MODULE")
    print("=" * 40)
    
    ocr = RubinOCRModule()
    
    # Информация о модуле
    info = ocr.get_module_info()
    print(f"📊 Модуль: {info['name']} v{info['version']}")
    print(f"🔧 OCR доступен: {'✅' if info['ocr_available'] else '❌'}")
    print(f"📁 Поддерживаемые форматы: {', '.join(info['supported_formats'])}")
    print(f"🌐 Языки: {', '.join(info['languages'])}")
    
    # Тестирование извлечения текста
    print("\n🧪 ТЕСТИРОВАНИЕ ИЗВЛЕЧЕНИЯ ТЕКСТА:")
    print("-" * 35)
    
    test_images = [
        "test_graph.png",
        "test_chart.jpg",
        "test_formula.png",
        "test_circuit.bmp"
    ]
    
    for image_path in test_images:
        print(f"\n📷 Тестирование: {image_path}")
        result = ocr.extract_text_from_image(image_path)
        
        if result['success']:
            print(f"✅ Текст извлечен (уверенность: {result['confidence']:.2f})")
            print(f"📝 Текст: {result['text'][:100]}...")
        else:
            print(f"❌ Ошибка: {result['error']}")
    
    # Тестирование анализа графиков
    print("\n📊 ТЕСТИРОВАНИЕ АНАЛИЗА ГРАФИКОВ:")
    print("-" * 35)
    
    for image_path in test_images:
        print(f"\n📈 Анализ графика: {image_path}")
        result = ocr.analyze_graph(image_path)
        
        if result['success']:
            analysis = result['analysis']
            print(f"✅ Тип графика: {analysis['graph_type']}")
            if analysis['function']:
                print(f"📐 Функция: {analysis['function']}")
            if analysis['data_points']:
                print(f"📍 Точки данных: {len(analysis['data_points'])}")
            if analysis['title']:
                print(f"🏷️ Заголовок: {analysis['title']}")
        else:
            print(f"❌ Ошибка: {result['error']}")
    
    # Тестирование анализа схем
    print("\n⚡ ТЕСТИРОВАНИЕ АНАЛИЗА СХЕМ:")
    print("-" * 30)
    
    circuit_image = "test_circuit.bmp"
    print(f"\n🔌 Анализ схемы: {circuit_image}")
    result = ocr.analyze_circuit_diagram(circuit_image)
    
    if result['success']:
        analysis = result['analysis']
        print(f"✅ Тип схемы: {analysis['circuit_type']}")
        print(f"🔧 Компоненты: {len(analysis['components'])}")
        print(f"🔗 Соединения: {len(analysis['connections'])}")
        print(f"📊 Значения: {analysis['values']}")
    else:
        print(f"❌ Ошибка: {result['error']}")
    
    print("\n🎉 ТЕСТИРОВАНИЕ OCR MODULE ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()





