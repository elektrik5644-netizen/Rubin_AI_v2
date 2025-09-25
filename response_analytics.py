#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система аналитики ответов Rubin AI
Проверяет качество ответов и автоматически улучшает их
"""

import logging
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import requests
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResponseAnalysis:
    """Результат анализа ответа"""
    is_logical: bool
    is_accurate: bool
    is_complete: bool
    quality_score: float
    issues: List[str]
    suggestions: List[str]
    corrected_response: Optional[str] = None

class ResponseAnalytics:
    """Система аналитики ответов"""
    
    def __init__(self):
        self.quality_patterns = {
            'logical_indicators': [
                r'поэтому', r'следовательно', r'таким образом', r'в результате',
                r'во-первых', r'во-вторых', r'в-третьих', r'сначала', r'затем',
                r'если.*то', r'при условии', r'в случае', r'когда'
            ],
            'accuracy_indicators': [
                r'точное значение', r'точный ответ', r'правильно', r'корректно',
                r'проверено', r'верифицировано', r'подтверждено', r'стандарт'
            ],
            'completeness_indicators': [
                r'полное решение', r'подробное объяснение', r'все аспекты',
                r'детальный анализ', r'исчерпывающий ответ', r'включает'
            ],
            'error_indicators': [
                r'ошибка', r'неправильно', r'некорректно', r'неверно',
                r'не удалось', r'проблема', r'сбой', r'недоступен'
            ]
        }
        
        self.subject_patterns = {
            'math': {
                'keywords': ['уравнение', 'формула', 'вычисление', 'математика', 'число'],
                'required_elements': ['решение', 'ответ', 'проверка'],
                'quality_checks': ['логичность_решения', 'правильность_расчета', 'полнота_объяснения']
            },
            'electrical': {
                'keywords': ['ток', 'напряжение', 'сопротивление', 'электричество', 'схема'],
                'required_elements': ['закон', 'формула', 'расчет', 'единицы'],
                'quality_checks': ['соответствие_законам', 'правильность_формул', 'безопасность']
            },
            'programming': {
                'keywords': ['код', 'программа', 'алгоритм', 'функция', 'класс'],
                'required_elements': ['код', 'объяснение', 'пример', 'тестирование'],
                'quality_checks': ['синтаксис', 'логика_программы', 'оптимизация']
            },
            'controllers': {
                'keywords': ['плка', 'чпу', 'контроллер', 'автоматизация', 'управление'],
                'required_elements': ['принцип_работы', 'алгоритм', 'настройка'],
                'quality_checks': ['техническая_корректность', 'практичность', 'стандарты']
            }
        }
        
        self.correction_templates = {
            'incomplete': "Ответ неполный. Добавлю недостающую информацию:",
            'illogical': "Обнаружена логическая ошибка. Исправляю:",
            'inaccurate': "Найдена неточность. Предоставляю корректную информацию:",
            'unclear': "Ответ неясен. Уточняю:",
            'missing_examples': "Добавлю практические примеры:"
        }
    
    def analyze_response(self, question: str, response: str, server_type: str) -> ResponseAnalysis:
        """Анализирует качество ответа"""
        try:
            logger.info(f"🔍 Анализирую ответ от {server_type}: {response[:50]}...")
            
            # Базовые проверки
            is_logical = self._check_logical_structure(response)
            is_accurate = self._check_accuracy(response, question, server_type)
            is_complete = self._check_completeness(response, question, server_type)
            
            # Выявление проблем
            issues = self._identify_issues(response, question, server_type)
            
            # Генерация предложений
            suggestions = self._generate_suggestions(issues, server_type)
            
            # Расчет качества
            quality_score = self._calculate_quality_score(is_logical, is_accurate, is_complete, issues)
            
            # Автоматическое исправление
            corrected_response = self._auto_correct_response(response, issues, question, server_type)
            
            return ResponseAnalysis(
                is_logical=is_logical,
                is_accurate=is_accurate,
                is_complete=is_complete,
                quality_score=quality_score,
                issues=issues,
                suggestions=suggestions,
                corrected_response=corrected_response
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа ответа: {e}")
            return ResponseAnalysis(
                is_logical=False,
                is_accurate=False,
                is_complete=False,
                quality_score=0.0,
                issues=[f"Ошибка анализа: {str(e)}"],
                suggestions=["Повторить анализ"]
            )
    
    def _check_logical_structure(self, response: str) -> bool:
        """Проверяет логическую структуру ответа"""
        response_lower = response.lower()
        
        # Проверяем наличие логических связок
        logical_connectors = sum(1 for pattern in self.quality_patterns['logical_indicators'] 
                               if re.search(pattern, response_lower))
        
        # Проверяем структуру (наличие введения, основной части, заключения)
        has_intro = any(word in response_lower for word in ['анализ', 'рассмотрим', 'изучим'])
        has_main = any(word in response_lower for word in ['решение', 'ответ', 'результат'])
        has_conclusion = any(word in response_lower for word in ['заключение', 'вывод', 'итог'])
        
        # Проверяем последовательность изложения
        has_sequence = any(word in response_lower for word in ['сначала', 'затем', 'далее', 'наконец'])
        
        # Оценка логичности
        logical_score = (logical_connectors * 0.3 + 
                        (has_intro + has_main + has_conclusion) * 0.2 + 
                        has_sequence * 0.1)
        
        return logical_score >= 0.6
    
    def _check_accuracy(self, response: str, question: str, server_type: str) -> bool:
        """Проверяет точность ответа"""
        response_lower = response.lower()
        question_lower = question.lower()
        
        # Проверяем наличие индикаторов точности
        accuracy_indicators = sum(1 for pattern in self.quality_patterns['accuracy_indicators'] 
                                if re.search(pattern, response_lower))
        
        # Проверяем наличие ошибок
        error_indicators = sum(1 for pattern in self.quality_patterns['error_indicators'] 
                             if re.search(pattern, response_lower))
        
        # Проверяем соответствие предметной области
        subject_accuracy = self._check_subject_accuracy(response, question, server_type)
        
        # Оценка точности
        accuracy_score = (accuracy_indicators * 0.4 + 
                         subject_accuracy * 0.4 - 
                         error_indicators * 0.2)
        
        return accuracy_score >= 0.5
    
    def _check_subject_accuracy(self, response: str, question: str, server_type: str) -> float:
        """Проверяет точность в контексте предметной области"""
        if server_type not in self.subject_patterns:
            return 0.5  # Нейтральная оценка для неизвестных типов
        
        subject_info = self.subject_patterns[server_type]
        response_lower = response.lower()
        
        # Проверяем наличие ключевых слов предметной области
        keyword_matches = sum(1 for keyword in subject_info['keywords'] 
                            if keyword in response_lower)
        
        # Проверяем наличие обязательных элементов
        element_matches = sum(1 for element in subject_info['required_elements'] 
                            if element in response_lower)
        
        # Проверяем качественные критерии
        quality_matches = sum(1 for check in subject_info['quality_checks'] 
                            if self._check_quality_criterion(response, check))
        
        # Расчет оценки
        total_possible = len(subject_info['keywords']) + len(subject_info['required_elements']) + len(subject_info['quality_checks'])
        actual_matches = keyword_matches + element_matches + quality_matches
        
        return actual_matches / total_possible if total_possible > 0 else 0.5
    
    def _check_quality_criterion(self, response: str, criterion: str) -> bool:
        """Проверяет конкретный качественный критерий"""
        response_lower = response.lower()
        
        criterion_checks = {
            'логичность_решения': ['пошагово', 'последовательно', 'логично'],
            'правильность_расчета': ['формула', 'расчет', 'проверка'],
            'полнота_объяснения': ['подробно', 'детально', 'полностью'],
            'соответствие_законам': ['закон ома', 'закон кирхгофа', 'физические законы'],
            'правильность_формул': ['формула', 'уравнение', 'математически'],
            'безопасность': ['безопасность', 'нормы', 'стандарты'],
            'синтаксис': ['синтаксис', 'грамматика', 'правильно написан'],
            'логика_программы': ['алгоритм', 'логика', 'структура'],
            'оптимизация': ['оптимизация', 'эффективность', 'производительность'],
            'техническая_корректность': ['технически', 'корректно', 'правильно'],
            'практичность': ['практически', 'применимо', 'реально'],
            'стандарты': ['стандарт', 'норма', 'требование']
        }
        
        if criterion in criterion_checks:
            return any(check in response_lower for check in criterion_checks[criterion])
        
        return False
    
    def _check_completeness(self, response: str, question: str, server_type: str) -> bool:
        """Проверяет полноту ответа"""
        response_lower = response.lower()
        question_lower = question.lower()
        
        # Проверяем наличие индикаторов полноты
        completeness_indicators = sum(1 for pattern in self.quality_patterns['completeness_indicators'] 
                                    if re.search(pattern, response_lower))
        
        # Проверяем длину ответа (достаточно ли подробный)
        length_score = min(len(response) / 5000, 1.0)  # Нормализуем к 5000 символам
        
        # Проверяем, отвечает ли на все части вопроса
        question_parts = self._extract_question_parts(question_lower)
        answered_parts = sum(1 for part in question_parts 
                           if any(word in response_lower for word in part.split()))
        
        completeness_score = (completeness_indicators * 0.3 + 
                            length_score * 0.4 + 
                            (answered_parts / len(question_parts)) * 0.3 if question_parts else 0.5)
        
        return completeness_score >= 0.6
    
    def _extract_question_parts(self, question: str) -> List[str]:
        """Извлекает части вопроса для проверки полноты ответа"""
        # Простое разделение по союзам и знакам препинания
        parts = re.split(r'[и,а,но,или,а также]', question)
        return [part.strip() for part in parts if part.strip()]
    
    def _identify_issues(self, response: str, question: str, server_type: str) -> List[str]:
        """Выявляет проблемы в ответе"""
        issues = []
        
        # Проверяем логические проблемы
        if not self._check_logical_structure(response):
            issues.append("Логическая структура ответа неясна")
        
        # Проверяем проблемы точности
        if not self._check_accuracy(response, question, server_type):
            issues.append("Ответ содержит неточности")
        
        # Проверяем проблемы полноты
        if not self._check_completeness(response, question, server_type):
            issues.append("Ответ неполный")
        
        # Проверяем наличие ошибок
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in ['ошибка', 'неправильно', 'некорректно']):
            issues.append("Ответ содержит ошибки")
        
        # Проверяем ясность
        if len(response) < 100:
            issues.append("Ответ слишком краткий")
        
        # Проверяем наличие примеров
        if server_type in ['math', 'programming', 'electrical'] and 'пример' not in response_lower:
            issues.append("Отсутствуют практические примеры")
        
        return issues
    
    def _generate_suggestions(self, issues: List[str], server_type: str) -> List[str]:
        """Генерирует предложения по улучшению"""
        suggestions = []
        
        for issue in issues:
            if "логическая структура" in issue.lower():
                suggestions.append("Добавить логические связки и структурировать ответ")
            elif "неточности" in issue.lower():
                suggestions.append("Проверить факты и добавить проверенную информацию")
            elif "неполный" in issue.lower():
                suggestions.append("Расширить ответ дополнительными деталями")
            elif "ошибки" in issue.lower():
                suggestions.append("Исправить ошибки и предоставить корректную информацию")
            elif "краткий" in issue.lower():
                suggestions.append("Добавить подробные объяснения")
            elif "примеры" in issue.lower():
                suggestions.append("Добавить практические примеры")
        
        # Общие предложения по типу сервера
        if server_type == 'math':
            suggestions.append("Добавить пошаговое решение с проверкой")
        elif server_type == 'electrical':
            suggestions.append("Включить формулы и единицы измерения")
        elif server_type == 'programming':
            suggestions.append("Предоставить рабочий код с комментариями")
        elif server_type == 'controllers':
            suggestions.append("Добавить технические детали и стандарты")
        
        return list(set(suggestions))  # Убираем дубликаты
    
    def _calculate_quality_score(self, is_logical: bool, is_accurate: bool, 
                                is_complete: bool, issues: List[str]) -> float:
        """Рассчитывает общий балл качества"""
        base_score = (is_logical * 0.4 + is_accurate * 0.4 + is_complete * 0.2)
        
        # Штрафы за проблемы
        issue_penalty = len(issues) * 0.1
        
        # Итоговый балл
        final_score = max(0.0, base_score - issue_penalty)
        
        return round(final_score, 2)
    
    def _auto_correct_response(self, response: str, issues: List[str], 
                             question: str, server_type: str) -> Optional[str]:
        """Автоматически исправляет ответ"""
        if not issues:
            return None
        
        corrected_parts = [response]
        
        for issue in issues:
            if "логическая структура" in issue.lower():
                corrected_parts.append(self._add_logical_structure())
            elif "неполный" in issue.lower():
                corrected_parts.append(self._add_completeness(question, server_type))
            elif "неточности" in issue.lower():
                corrected_parts.append(self._add_accuracy_corrections(server_type))
            elif "примеры" in issue.lower():
                corrected_parts.append(self._add_examples(server_type))
        
        if len(corrected_parts) > 1:
            return "\n\n".join(corrected_parts)
        
        return None
    
    def _add_logical_structure(self) -> str:
        """Добавляет логическую структуру"""
        return """
**🔧 УЛУЧШЕНИЕ ЛОГИЧЕСКОЙ СТРУКТУРЫ:**

Для лучшего понимания структурирую ответ:

1. **Введение:** Определение основных понятий
2. **Основная часть:** Пошаговое решение/объяснение  
3. **Заключение:** Итоговый результат и выводы
4. **Проверка:** Верификация правильности решения
"""
    
    def _add_completeness(self, question: str, server_type: str) -> str:
        """Добавляет недостающую информацию"""
        completeness_additions = {
            'math': """
**📚 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:**

• **Альтернативные методы решения**
• **Проверка результата**
• **Практическое применение**
• **Связанные темы**
""",
            'electrical': """
**⚡ ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:**

• **Физические принципы**
• **Безопасность и нормы**
• **Практические рекомендации**
• **Связанные компоненты**
""",
            'programming': """
**💻 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:**

• **Альтернативные подходы**
• **Оптимизация кода**
• **Обработка ошибок**
• **Тестирование**
""",
            'controllers': """
**🎛️ ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:**

• **Технические характеристики**
• **Настройка и калибровка**
• **Диагностика и обслуживание**
• **Соответствие стандартам**
"""
        }
        
        return completeness_additions.get(server_type, """
**📖 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:**

• **Подробное объяснение**
• **Практические примеры**
• **Связанные темы**
• **Дополнительные ресурсы**
""")
    
    def _add_accuracy_corrections(self, server_type: str) -> str:
        """Добавляет исправления точности"""
        return """
**✅ ИСПРАВЛЕНИЯ ТОЧНОСТИ:**

• **Проверенные факты и данные**
• **Соответствие стандартам**
• **Верифицированная информация**
• **Корректные формулы и расчеты**
"""
    
    def _add_examples(self, server_type: str) -> str:
        """Добавляет практические примеры"""
        examples = {
            'math': """
**📝 ПРАКТИЧЕСКИЕ ПРИМЕРЫ:**

• **Пример 1:** Базовое применение
• **Пример 2:** Усложненная задача
• **Пример 3:** Проверка решения
""",
            'electrical': """
**⚡ ПРАКТИЧЕСКИЕ ПРИМЕРЫ:**

• **Пример 1:** Простая схема
• **Пример 2:** Расчет параметров
• **Пример 3:** Практическое применение
""",
            'programming': """
**💻 ПРАКТИЧЕСКИЕ ПРИМЕРЫ:**

• **Пример 1:** Базовый код
• **Пример 2:** Оптимизированная версия
• **Пример 3:** Тестирование
""",
            'controllers': """
**🎛️ ПРАКТИЧЕСКИЕ ПРИМЕРЫ:**

• **Пример 1:** Базовая настройка
• **Пример 2:** Расширенная конфигурация
• **Пример 3:** Диагностика проблем
"""
        }
        
        return examples.get(server_type, """
**📝 ПРАКТИЧЕСКИЕ ПРИМЕРЫ:**

• **Пример 1:** Базовое применение
• **Пример 2:** Расширенное использование
• **Пример 3:** Практические советы
""")

class ResponseQualityController:
    """Контроллер качества ответов"""
    
    def __init__(self):
        self.analytics = ResponseAnalytics()
        self.quality_threshold = 0.7  # Порог качества для автоматического исправления
        self.auto_correction_enabled = True
    
    def process_response(self, question: str, response: str, server_type: str) -> Dict[str, Any]:
        """Обрабатывает ответ с проверкой качества"""
        try:
            logger.info(f"🔍 Обрабатываю ответ для анализа качества...")
            
            # Анализируем ответ
            analysis = self.analytics.analyze_response(question, response, server_type)
            
            # Определяем, нужна ли коррекция
            needs_correction = (analysis.quality_score < self.quality_threshold or 
                              len(analysis.issues) > 0)
            
            # Выбираем финальный ответ
            if needs_correction and self.auto_correction_enabled and analysis.corrected_response:
                final_response = analysis.corrected_response
                correction_applied = True
            else:
                final_response = response
                correction_applied = False
            
            # Формируем результат
            result = {
                'original_response': response,
                'final_response': final_response,
                'analysis': {
                    'is_logical': analysis.is_logical,
                    'is_accurate': analysis.is_accurate,
                    'is_complete': analysis.is_complete,
                    'quality_score': analysis.quality_score,
                    'issues': analysis.issues,
                    'suggestions': analysis.suggestions
                },
                'correction_applied': correction_applied,
                'quality_status': self._get_quality_status(analysis.quality_score),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Анализ завершен. Качество: {analysis.quality_score}, Исправлений: {correction_applied}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки ответа: {e}")
            return {
                'original_response': response,
                'final_response': response,
                'analysis': {
                    'is_logical': False,
                    'is_accurate': False,
                    'is_complete': False,
                    'quality_score': 0.0,
                    'issues': [f"Ошибка анализа: {str(e)}"],
                    'suggestions': ["Повторить анализ"]
                },
                'correction_applied': False,
                'quality_status': 'error',
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_quality_status(self, score: float) -> str:
        """Определяет статус качества"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def configure_quality_settings(self, threshold: float = None, 
                                  auto_correction: bool = None):
        """Настраивает параметры качества"""
        if threshold is not None:
            self.quality_threshold = threshold
        if auto_correction is not None:
            self.auto_correction_enabled = auto_correction
        
        logger.info(f"⚙️ Настройки качества обновлены: threshold={self.quality_threshold}, auto_correction={self.auto_correction_enabled}")

# Глобальный экземпляр
quality_controller = None

def get_quality_controller():
    """Получает глобальный экземпляр контроллера качества"""
    global quality_controller
    if quality_controller is None:
        quality_controller = ResponseQualityController()
    return quality_controller

if __name__ == "__main__":
    print("🚀 Тестирование системы аналитики ответов")
    
    controller = get_quality_controller()
    
    # Тестовые сценарии
    test_cases = [
        {
            'question': 'Реши уравнение x^2 + 5x + 6 = 0',
            'response': 'Ответ: x = -2 или x = -3',
            'server_type': 'math'
        },
        {
            'question': 'Что такое закон Ома?',
            'response': 'Закон Ома связывает ток, напряжение и сопротивление.',
            'server_type': 'electrical'
        },
        {
            'question': 'Напиши программу на Python',
            'response': 'def hello(): print("Hello")',
            'server_type': 'programming'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Тест {i}: {test_case['question']}")
        print(f"{'='*80}")
        
        result = controller.process_response(
            test_case['question'],
            test_case['response'],
            test_case['server_type']
        )
        
        print(f"📊 Качество: {result['analysis']['quality_score']}")
        print(f"🎯 Статус: {result['quality_status']}")
        print(f"✅ Логичный: {result['analysis']['is_logical']}")
        print(f"🎯 Точный: {result['analysis']['is_accurate']}")
        print(f"📝 Полный: {result['analysis']['is_complete']}")
        print(f"🔧 Исправлен: {result['correction_applied']}")
        
        if result['analysis']['issues']:
            print(f"⚠️ Проблемы: {', '.join(result['analysis']['issues'])}")
        
        if result['analysis']['suggestions']:
            print(f"💡 Предложения: {', '.join(result['analysis']['suggestions'])}")
        
        print(f"\n📝 Финальный ответ:")
        print(result['final_response'][:200] + "..." if len(result['final_response']) > 200 else result['final_response'])
    
    print(f"\n✅ Тестирование завершено!")



