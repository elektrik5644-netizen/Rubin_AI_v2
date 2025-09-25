#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный категоризатор запросов для Rubin AI
Правильно определяет тип вопроса для корректной маршрутизации
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProcessedRequest:
    """Структура обработанного запроса"""
    original_message: str
    language: str  # 'ru' или 'en'
    category: str
    confidence: float
    keywords: List[str]
    timestamp: datetime

class EnhancedRequestCategorizer:
    """Улучшенный категоризатор запросов с поддержкой русского и английского языков"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Определение категорий с ключевыми словами и паттернами
        self.categories = {
            'programming': {
                'keywords_ru': [
                    'c++', 'python', 'программирование', 'код', 'алгоритм', 'сравни', 'сравнить',
                    'язык программирования', 'разработка', 'автоматизация', 'plc программирование',
                    'промышленное программирование', 'встраиваемые системы', 'реальное время',
                    'библиотеки', 'фреймворк', 'компилятор', 'интерпретатор', 'производительность'
                ],
                'keywords_en': [
                    'c++', 'python', 'programming', 'code', 'algorithm', 'compare', 'comparison',
                    'programming language', 'development', 'automation', 'plc programming',
                    'industrial programming', 'embedded systems', 'real time', 'libraries',
                    'framework', 'compiler', 'interpreter', 'performance'
                ],
                'patterns': [
                    r'сравни.*python.*c\+\+',
                    r'сравни.*c\+\+.*python', 
                    r'compare.*python.*c\+\+',
                    r'compare.*c\+\+.*python',
                    r'язык.*программирования.*для.*автоматизации',
                    r'programming.*language.*for.*automation',
                    r'какой.*язык.*лучше',
                    r'which.*language.*better'
                ],
                'priority': 10  # Высокий приоритет для исправления проблемы
            },
            
            'electrical': {
                'keywords_ru': [
                    'защита', 'короткое замыкание', 'цепи', 'электрические', 'электротехника',
                    'предохранитель', 'автомат', 'узо', 'реле', 'трансформатор', 'двигатель',
                    'ток', 'напряжение', 'сопротивление', 'мощность', 'коэффициент мощности',
                    'диод', 'транзистор', 'резистор', 'конденсатор', 'индуктивность'
                ],
                'keywords_en': [
                    'protection', 'short circuit', 'circuits', 'electrical', 'electronics',
                    'fuse', 'breaker', 'relay', 'transformer', 'motor', 'current', 'voltage',
                    'resistance', 'power', 'power factor', 'diode', 'transistor', 'resistor',
                    'capacitor', 'inductance'
                ],
                'patterns': [
                    r'как защитить.*цепи',
                    r'how to protect.*circuits',
                    r'короткое замыкание',
                    r'short circuit',
                    r'защита.*от.*кз',
                    r'protection.*from.*short',
                    r'электрические.*цепи',
                    r'electrical.*circuits'
                ],
                'priority': 9  # Высокий приоритет для исправления проблемы
            },
            
            'physics': {
                'keywords_ru': [
                    'физика', 'квантовая', 'теория', 'относительность', 'механика',
                    'термодинамика', 'оптика', 'астрофизика', 'космос', 'реактор',
                    'энергия', 'материя', 'волна', 'частица', 'поле', 'закон',
                    'гравитация', 'электричество', 'магнетизм', 'ядерная физика'
                ],
                'keywords_en': [
                    'physics', 'quantum', 'theory', 'relativity', 'mechanics',
                    'thermodynamics', 'optics', 'astrophysics', 'space', 'reactor',
                    'energy', 'matter', 'wave', 'particle', 'field', 'law',
                    'gravity', 'electricity', 'magnetism', 'nuclear physics'
                ],
                'patterns': [
                    r'что\s+такое\s+квантовая\s+физика',
                    r'what\s+is\s+quantum\s+physics',
                    r'какие\s+бывают\s+виды\s+реакторов',
                    r'what\s+are\s+types\s+of\s+reactors'
                ],
                'priority': 8
            },
            
            'mathematics': {
                'keywords_ru': [
                    'решить', 'вычислить', 'найти', 'угол', 'градус', 'уравнение', 'формула',
                    'арифметика', 'геометрия', 'алгебра', 'тригонометрия', 'интеграл',
                    'производная', 'матрица', 'вектор', 'функция'
                ],
                'keywords_en': [
                    'solve', 'calculate', 'find', 'angle', 'degree', 'equation', 'formula',
                    'arithmetic', 'geometry', 'algebra', 'trigonometry', 'integral',
                    'derivative', 'matrix', 'vector', 'function'
                ],
                'patterns': [
                    r'\d+\s*[+\-*/]\s*\d+',  # Арифметические выражения
                    r'угол.*градус',
                    r'angle.*degree',
                    r'решить.*уравнение',
                    r'solve.*equation',
                    r'\d+\s*яблок',  # Задачи с яблоками
                    r'луч.*делит.*угол',
                    r'смежные.*углы'
                ],
                'priority': 5  # Средний приоритет
            },
            
            'controllers': {
                'keywords_ru': [
                    'пид', 'pid', 'регулятор', 'контроллер', 'plc', 'scada', 'hmi',
                    'автоматизация', 'управление', 'датчик', 'привод', 'сервопривод',
                    'частотный преобразователь', 'modbus', 'profibus', 'ethernet'
                ],
                'keywords_en': [
                    'pid', 'controller', 'plc', 'scada', 'hmi', 'automation', 'control',
                    'sensor', 'actuator', 'servo', 'frequency converter', 'modbus',
                    'profibus', 'ethernet'
                ],
                'patterns': [
                    r'пид.*регулятор',
                    r'pid.*controller',
                    r'plc.*программирование',
                    r'plc.*programming',
                    r'система.*управления',
                    r'control.*system'
                ],
                'priority': 7
            },
            
            'radiomechanics': {
                'keywords_ru': [
                    'антенна', 'сигнал', 'радио', 'модуляция', 'частота', 'передатчик',
                    'приемник', 'усилитель', 'фильтр', 'генератор', 'осциллограф',
                    'спектр', 'децибел', 'шум', 'помехи'
                ],
                'keywords_en': [
                    'antenna', 'signal', 'radio', 'modulation', 'frequency', 'transmitter',
                    'receiver', 'amplifier', 'filter', 'generator', 'oscilloscope',
                    'spectrum', 'decibel', 'noise', 'interference'
                ],
                'patterns': [
                    r'радио.*сигнал',
                    r'radio.*signal',
                    r'антенна.*частота',
                    r'antenna.*frequency'
                ],
                'priority': 6
            },
            
            'general': {
                'keywords_ru': [
                    'привет', 'здравствуй', 'как дела', 'помощь', 'что умеешь',
                    'спасибо', 'пока', 'до свидания'
                ],
                'keywords_en': [
                    'hello', 'hi', 'how are you', 'help', 'what can you do',
                    'thanks', 'bye', 'goodbye'
                ],
                'patterns': [
                    r'^привет$',
                    r'^hello$',
                    r'^hi$',
                    r'как дела',
                    r'how are you'
                ],
                'priority': 1  # Низкий приоритет
            },
            
            'time_series': {
                'keywords_ru': [
                    'временные ряды', 'прогноз', 'прогнозирование', 'тренд', 'анализ времени',
                    'прогнозировать', 'будущее', 'динамика', 'предсказать', 'временной ряд',
                    'период', 'бар', 'цена закрытия', 'котировки', 'график'
                ],
                'keywords_en': [
                    'time series', 'forecast', 'forecasting', 'trend', 'time analysis',
                    'predict', 'future', 'dynamics', 'predict', 'time series',
                    'period', 'bar', 'close price', 'quotes', 'chart'
                ],
                'patterns': [
                    r'прогноз.*временных.*рядов',
                    r'forecast.*time.*series',
                    r'прогнозировать.*тренд',
                    r'predict.*trend',
                    r'временные.*ряды.*анализ',
                    r'time.*series.*analysis'
                ],
                'priority': 12 # Высокий приоритет
            }
        }
        
        # Кэш для ускорения повторных запросов
        self.categorization_cache = {}
        
        self.logger.info("Улучшенный категоризатор запросов инициализирован")
    
    def detect_language(self, message: str) -> str:
        """Определение языка сообщения"""
        # Простая эвристика на основе кириллицы
        cyrillic_chars = len(re.findall(r'[а-яё]', message.lower()))
        latin_chars = len(re.findall(r'[a-z]', message.lower()))
        
        if cyrillic_chars > latin_chars:
            return 'ru'
        elif latin_chars > 0:
            return 'en'
        else:
            return 'ru'  # По умолчанию русский
    
    def extract_keywords(self, message: str, language: str) -> List[str]:
        """Извлечение ключевых слов из сообщения"""
        message_lower = message.lower()
        found_keywords = []
        
        for category, config in self.categories.items():
            keywords_key = f'keywords_{language}'
            if keywords_key in config:
                for keyword in config[keywords_key]:
                    if keyword in message_lower:
                        found_keywords.append(keyword)
        
        return found_keywords
    
    def calculate_category_score(self, message: str, category: str, language: str) -> float:
        """Вычисление score для категории"""
        message_lower = message.lower()
        config = self.categories[category]
        score = 0.0
        
        # Проверка ключевых слов
        keywords_key = f'keywords_{language}'
        if keywords_key in config:
            for keyword in config[keywords_key]:
                if keyword in message_lower:
                    # Вес зависит от длины ключевого слова
                    weight = len(keyword.split()) * 1.0
                    score += weight
        
        # Проверка паттернов (более высокий вес)
        if 'patterns' in config:
            for pattern in config['patterns']:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    score += 5.0  # Паттерны имеют высокий вес
        
        # Применение приоритета категории
        priority_multiplier = config.get('priority', 1) / 10.0
        score *= priority_multiplier
        
        return score
    
    def categorize(self, message: str) -> str:
        """Основной метод категоризации"""
        # Проверка кэша
        if message in self.categorization_cache:
            cached_result = self.categorization_cache[message]
            self.logger.debug(f"Использован кэш для: {message[:50]}... -> {cached_result}")
            return cached_result
        
        # Определение языка
        language = self.detect_language(message)
        
        # Вычисление score для каждой категории
        scores = {}
        for category in self.categories.keys():
            scores[category] = self.calculate_category_score(message, category, language)
        
        # Выбор категории с максимальным score
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        # Если score слишком низкий, используем общую категорию
        if best_score < 1.0:
            best_category = 'general'
        
        # Кэширование результата
        self.categorization_cache[message] = best_category
        
        self.logger.info(f"Категоризация: '{message[:50]}...' -> {best_category} (score: {best_score:.2f})")
        
        return best_category
    
    def get_confidence(self, message: str, category: str) -> float:
        """Получение уверенности в категоризации"""
        language = self.detect_language(message)
        
        # Вычисляем score для всех категорий
        scores = {}
        for cat in self.categories.keys():
            scores[cat] = self.calculate_category_score(message, cat, language)
        
        # Нормализация confidence
        total_score = sum(scores.values())
        if total_score == 0:
            return 0.0
        
        category_score = scores.get(category, 0)
        confidence = category_score / total_score
        
        return min(confidence, 1.0)  # Ограничиваем максимум 1.0
    
    def process_request(self, message: str) -> ProcessedRequest:
        """Полная обработка запроса"""
        language = self.detect_language(message)
        category = self.categorize(message)
        confidence = self.get_confidence(message, category)
        keywords = self.extract_keywords(message, language)
        
        return ProcessedRequest(
            original_message=message,
            language=language,
            category=category,
            confidence=confidence,
            keywords=keywords,
            timestamp=datetime.now()
        )
    
    def get_statistics(self) -> Dict:
        """Получение статистики работы категоризатора"""
        return {
            'cache_size': len(self.categorization_cache),
            'categories_count': len(self.categories),
            'supported_languages': ['ru', 'en']
        }
    
    def clear_cache(self):
        """Очистка кэша категоризации"""
        self.categorization_cache.clear()
        self.logger.info("Кэш категоризации очищен")

# Глобальный экземпляр для использования в других модулях
_categorizer_instance = None

def get_enhanced_categorizer() -> EnhancedRequestCategorizer:
    """Получение глобального экземпляра категоризатора"""
    global _categorizer_instance
    if _categorizer_instance is None:
        _categorizer_instance = EnhancedRequestCategorizer()
    return _categorizer_instance

# Тестирование категоризатора
if __name__ == "__main__":
    # Настройка логирования для тестирования
    logging.basicConfig(level=logging.INFO)
    
    categorizer = EnhancedRequestCategorizer()
    
    # Тестовые запросы
    test_requests = [
        "Сравни C++ и Python для задач промышленной автоматизации",
        "Как защитить электрические цепи от короткого замыкания?",
        "2 + 3 = ?",
        "Что такое ПИД-регулятор?",
        "Привет, как дела?",
        "Compare C++ and Python for automation",
        "How to protect circuits from short circuit?",
        "Solve equation x + 5 = 10"
    ]
    
    print("=== ТЕСТИРОВАНИЕ КАТЕГОРИЗАТОРА ===")
    for request in test_requests:
        processed = categorizer.process_request(request)
        print(f"Запрос: {request}")
        print(f"  Язык: {processed.language}")
        print(f"  Категория: {processed.category}")
        print(f"  Уверенность: {processed.confidence:.2f}")
        print(f"  Ключевые слова: {processed.keywords}")
        print()
    
    print("Статистика:", categorizer.get_statistics())