#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль обработки данных для Rubin AI v2.0
Сортировка, анализ и очистка информации от диспетчера
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import chardet
import langdetect
from langdetect import detect, DetectorFactory
from rubin_text_preprocessor import RubinTextPreprocessor # Добавляем для централизованной очистки текста

# Устанавливаем детерминированный режим для langdetect
DetectorFactory.seed = 0

@dataclass
class ProcessedContent:
    """Обработанный контент"""
    original_content: str
    cleaned_content: str
    language: str
    format_type: str
    quality_score: float
    readability_score: float
    metadata: Dict[str, Any]
    filtered_sections: List[str]
    valid_sections: List[str]

class DataProcessor:
    """Процессор данных для очистки и анализа контента"""
    
    def __init__(self):
        self.setup_logging()
        self.text_preprocessor = RubinTextPreprocessor() # Инициализация препроцессора текста
        
        # Паттерны для фильтрации
        self.noise_patterns = [
            r'^[\s\-_=*]+$',  # Только разделители
            r'^\d+\.?\s*$',   # Только номера
            r'^[;#]+.*$',     # Комментарии
            r'^[\s]*$',       # Пустые строки
            r'^.*\.(txt|pdf|docx):.*$',  # Названия файлов
            r'^📄.*$',        # Эмодзи файлов
            r'^.*\*\*.*\.txt\*\*.*$',    # Markdown названия файлов
        ]
        
        # Паттерны для определения формата
        self.format_patterns = {
            'code': [
                r'^\s*[{}()\[\]]',  # Скобки в начале строки
                r'^\s*(if|for|while|def|class|import|from)\s',  # Ключевые слова Python
                r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=<>!]',  # Присваивания
            ],
            'markdown': [
                r'^#{1,6}\s+',  # Заголовки
                r'^\*\*.*\*\*$',  # Жирный текст
                r'^\*.*\*$',  # Курсив
                r'^\- ',  # Списки
            ],
            'technical': [
                r'^\s*[A-Z][A-Z_\s]+$',  # Заголовки в верхнем регистре
                r'^\s*\d+\.\d+\.\d+',  # Версии
                r'^\s*[A-Z][a-z]+\s*:',  # Параметры
            ],
            'plain_text': []  # По умолчанию
        }
        
        # Ключевые слова для определения качества
        self.quality_indicators = {
            'high': ['принцип', 'метод', 'алгоритм', 'функция', 'процесс', 'система'],
            'medium': ['описание', 'характеристика', 'параметр', 'настройка'],
            'low': ['комментарий', 'примечание', 'заметка', 'вспомогательный']
        }
    
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_search_results(self, search_results: List[Dict]) -> ProcessedContent:
        """Основной метод обработки результатов поиска"""
        self.logger.info(f"🔄 Обработка {len(search_results)} результатов поиска")
        
        # Объединение всего контента
        all_content = self._extract_content(search_results)
        
        # Очистка контента
        cleaned_content = self._clean_content(all_content)
        
        # Определение языка и формата
        language = self._detect_language(cleaned_content)
        format_type = self._detect_format(cleaned_content)
        
        # Оценка качества
        quality_score = self._assess_quality(cleaned_content)
        readability_score = self._assess_readability(cleaned_content)
        
        # Фильтрация секций
        filtered_sections, valid_sections = self._filter_sections(cleaned_content)
        
        # Создание метаданных
        metadata = {
            'original_sections': len(search_results),
            'filtered_sections': len(filtered_sections),
            'valid_sections': len(valid_sections),
            'processing_time': 0,  # Будет заполнено
            'confidence': quality_score
        }
        
        processed_content = ProcessedContent(
            original_content=all_content,
            cleaned_content=cleaned_content,
            language=language,
            format_type=format_type,
            quality_score=quality_score,
            readability_score=readability_score,
            metadata=metadata,
            filtered_sections=filtered_sections,
            valid_sections=valid_sections
        )
        
        self.logger.info(f"✅ Обработка завершена: качество {quality_score:.2f}, читаемость {readability_score:.2f}")
        
        return processed_content
    
    def _extract_content(self, search_results: List[Dict]) -> str:
        """Извлечение контента из результатов поиска"""
        content_parts = []
        
        for result in search_results:
            # Извлекаем содержимое (приоритет полному контенту)
            content = result.get('content', '')
            if not content:
                content = result.get('content_preview', '')
            
            if content:
                # Добавляем информацию о источнике (скрыто)
                source_info = f"[Источник: {result.get('file_name', 'Unknown')}]"
                content_parts.append(f"{content}\n{source_info}")
        
        return '\n\n'.join(content_parts)
    
    def _clean_content(self, content: str) -> str:
        """Очистка контента от лишней информации"""
        if not content:
            return ""
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Пропускаем пустые строки
            if not line:
                continue
            
            # Проверяем паттерны шума
            is_noise = False
            for pattern in self.noise_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_noise = True
                    break
            
            if not is_noise:
                # Дополнительная очистка
                cleaned_line = self._clean_line(line)
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
        
        # Объединяем строки
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Удаляем множественные пробелы и переносы
        cleaned_content = self.text_preprocessor.remove_extra_spaces(cleaned_content) # Используем новый препроцессор
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        
        return cleaned_content.strip()
    
    def _clean_line(self, line: str) -> str:
        """Очистка отдельной строки"""
        # Удаляем лишние символы в начале
        line = re.sub(r'^[\s\-_=*]+', '', line)
        
        # Удаляем лишние символы в конце
        line = re.sub(r'[\s\-_=*]+$', '', line)
        
        # Удаляем множественные пробелы
        line = re.sub(r'\s+', ' ', line)
        
        # Удаляем информацию об источнике (скрыто)
        line = re.sub(r'\[Источник:.*?\]', '', line)
        
        return line.strip()
    
    def _detect_language(self, content: str) -> str:
        """Определение языка контента"""
        try:
            if not content or len(content.strip()) < 10:
                return 'unknown'
            
            # Берем первые 1000 символов для анализа
            sample = content[:1000]
            language = detect(sample)
            return language
        except Exception as e:
            self.logger.warning(f"Не удалось определить язык: {e}")
            return 'unknown'
    
    def _detect_format(self, content: str) -> str:
        """Определение формата контента"""
        lines = content.split('\n')
        format_scores = {format_name: 0 for format_name in self.format_patterns.keys()}
        
        for line in lines[:50]:  # Анализируем первые 50 строк
            for format_name, patterns in self.format_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line):
                        format_scores[format_name] += 1
        
        # Возвращаем формат с наибольшим количеством совпадений
        if format_scores:
            return max(format_scores, key=format_scores.get)
        return 'plain_text'
    
    def _assess_quality(self, content: str) -> float:
        """Оценка качества контента"""
        if not content:
            return 0.0
        
        quality_score = 0.5  # Базовый score
        
        # Анализ длины контента
        content_length = len(content)
        if content_length > 5000:
            quality_score += 0.2
        elif content_length > 2000:
            quality_score += 0.1
        
        # Анализ ключевых слов
        content_lower = content.lower()
        for level, keywords in self.quality_indicators.items():
            for keyword in keywords:
                if keyword in content_lower:
                    if level == 'high':
                        quality_score += 0.1
                    elif level == 'medium':
                        quality_score += 0.05
                    else:
                        quality_score += 0.02
        
        # Анализ структуры
        lines = content.split('\n')
        if len(lines) > 5:
            quality_score += 0.1
        
        # Проверка на наличие технической информации
        technical_indicators = ['параметр', 'настройка', 'конфигурация', 'алгоритм', 'функция']
        for indicator in technical_indicators:
            if indicator in content_lower:
                quality_score += 0.05
                break
        
        return min(1.0, quality_score)
    
    def _assess_readability(self, content: str) -> float:
        """Оценка читаемости контента"""
        if not content:
            return 0.0
        
        lines = content.split('\n')
        readability_score = 1.0  # Базовый score
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Штраф за слишком длинные строки
            if len(line) > 100:
                readability_score -= 0.05
            
            # Штраф за строки только из символов
            if re.match(r'^[^\w\s]+$', line):
                readability_score -= 0.1
            
            # Штраф за строки с множественными специальными символами
            special_chars = len(re.findall(r'[^\w\s]', line))
            if special_chars > len(line) * 0.3:
                readability_score -= 0.05
        
        return max(0.0, readability_score)
    
    def _filter_sections(self, content: str) -> Tuple[List[str], List[str]]:
        """Фильтрация секций контента"""
        sections = content.split('\n\n')
        filtered_sections = []
        valid_sections = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Проверяем качество секции
            section_quality = self._assess_quality(section)
            section_readability = self._assess_readability(section)
            
            if section_quality >= 0.3 and section_readability >= 0.5:
                valid_sections.append(section)
            else:
                filtered_sections.append(section)
        
        return filtered_sections, valid_sections
    
    def prepare_for_llm(self, processed_content: ProcessedContent) -> Dict[str, Any]:
        """Подготовка данных для передачи в локальную LLM"""
        return {
            'content': processed_content.cleaned_content,
            'language': processed_content.language,
            'format': processed_content.format_type,
            'quality_score': processed_content.quality_score,
            'readability_score': processed_content.readability_score,
            'valid_sections_count': len(processed_content.valid_sections),
            'metadata': processed_content.metadata
        }
    
    def validate_llm_response(self, response: str) -> Dict[str, Any]:
        """Проверка качества ответа от LLM"""
        if not response:
            return {
                'valid': False,
                'reason': 'Пустой ответ',
                'quality_score': 0.0
            }
        
        # Проверка длины
        if len(response) < 50:
            return {
                'valid': False,
                'reason': 'Слишком короткий ответ',
                'quality_score': 0.3
            }
        
        # Проверка на наличие технической информации
        technical_indicators = ['принцип', 'метод', 'функция', 'алгоритм', 'система']
        has_technical_info = any(indicator in response.lower() for indicator in technical_indicators)
        
        # Проверка структуры
        has_structure = bool(re.search(r'\n|•|\*|\d+\.', response))
        
        # Оценка качества
        quality_score = 0.5
        if has_technical_info:
            quality_score += 0.3
        if has_structure:
            quality_score += 0.2
        if len(response) > 200:
            quality_score += 0.1
        
        return {
            'valid': quality_score >= 0.6,
            'reason': 'Качество ответа' if quality_score >= 0.6 else 'Низкое качество',
            'quality_score': min(1.0, quality_score),
            'has_technical_info': has_technical_info,
            'has_structure': has_structure,
            'length': len(response)
        }

# Глобальный экземпляр процессора
data_processor = DataProcessor()

def get_data_processor() -> DataProcessor:
    """Получение глобального экземпляра процессора данных"""
    return data_processor

if __name__ == "__main__":
    # Тестирование процессора данных
    test_results = [
        {
            'file_name': 'test.txt',
            'content_preview': 'Это тестовый контент для проверки работы процессора данных.',
            'category': 'test'
        },
        {
            'file_name': 'test2.txt',
            'content_preview': 'Дополнительная информация о принципах работы системы.',
            'category': 'test'
        }
    ]
    
    processor = DataProcessor()
    processed = processor.process_search_results(test_results)
    
    print(f"Язык: {processed.language}")
    print(f"Формат: {processed.format_type}")
    print(f"Качество: {processed.quality_score:.2f}")
    print(f"Читаемость: {processed.readability_score:.2f}")
    print(f"Очищенный контент:\n{processed.cleaned_content}")
