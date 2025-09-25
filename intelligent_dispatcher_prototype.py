#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Прототип улучшенного интеллектуального диспетчера
Реализует архитектуру с поисковиком, анализатором и системой обратной связи
"""

import re
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Типы намерений пользователя"""
    ELECTRICAL_ANALYSIS = "electrical_analysis"
    PROGRAMMING = "programming"
    CONTROLLERS = "controllers"
    RADIOMECHANICS = "radiomechanics"
    SCIENCE = "science"
    GENERAL = "general"

class QualityLevel(Enum):
    """Уровни качества ответа"""
    EXCELLENT = "excellent"  # >= 0.9
    GOOD = "good"          # >= 0.7
    FAIR = "fair"          # >= 0.5
    POOR = "poor"          # < 0.5

@dataclass
class QueryContext:
    """Контекст запроса пользователя"""
    intent: QueryIntent
    entities: List[str]
    complexity: str
    domain: str
    requires_examples: bool
    requires_code: bool
    original_message: str

@dataclass
class SearchResult:
    """Результат поиска"""
    content: str
    source: str
    quality_score: float
    relevance_score: float
    metadata: Dict[str, Any]

@dataclass
class QualityScore:
    """Оценка качества ответа"""
    completeness: float
    accuracy: float
    relevance: float
    clarity: float
    overall_score: float

class QueryAnalyzer:
    """Анализатор запросов пользователя"""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.ELECTRICAL_ANALYSIS: [
                r'электрическ', r'электротехник', r'резистор', r'диод', r'транзистор',
                r'конденсатор', r'модбас', r'modbus', r'rtu', r'протокол', r'схема',
                r'цепь', r'напряжение', r'ток', r'мощность'
            ],
            QueryIntent.PROGRAMMING: [
                r'программирован', r'код', r'алгоритм', r'python', r'c\+\+', r'java',
                r'функция', r'класс', r'переменная', r'цикл', r'условие'
            ],
            QueryIntent.CONTROLLERS: [
                r'контроллер', r'plc', r'pmac', r'пид', r'pid', r'scada', r'автоматизац',
                r'датчик', r'привод', r'регулятор'
            ],
            QueryIntent.RADIOMECHANICS: [
                r'радио', r'антенн', r'передатчик', r'приемник', r'частота', r'сигнал',
                r'модуляц', r'демодуляц', r'усилитель'
            ],
            QueryIntent.SCIENCE: [
                r'физик', r'математик', r'формул', r'расчет', r'теорема', r'закон'
            ]
        }
        
        self.complexity_indicators = {
            'high': [r'подробно', r'детально', r'полное руководство', r'все аспекты'],
            'medium': [r'объясни', r'как работает', r'принцип'],
            'low': [r'что такое', r'определение', r'кратко']
        }
    
    def analyze_query(self, message: str) -> QueryContext:
        """Анализ запроса пользователя"""
        message_lower = message.lower()
        
        # Определение намерения
        intent = self._detect_intent(message_lower)
        
        # Извлечение сущностей
        entities = self._extract_entities(message_lower)
        
        # Оценка сложности
        complexity = self._assess_complexity(message_lower)
        
        # Определение домена
        domain = self._determine_domain(intent)
        
        # Анализ требований
        requires_examples = self._requires_examples(message_lower)
        requires_code = self._requires_code(message_lower)
        
        return QueryContext(
            intent=intent,
            entities=entities,
            complexity=complexity,
            domain=domain,
            requires_examples=requires_examples,
            requires_code=requires_code,
            original_message=message
        )
    
    def _detect_intent(self, message: str) -> QueryIntent:
        """Определение намерения пользователя"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, message)
                score += len(matches)
            intent_scores[intent] = score
        
        # Возвращаем намерение с наибольшим счетом
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0] if best_intent[1] > 0 else QueryIntent.GENERAL
    
    def _extract_entities(self, message: str) -> List[str]:
        """Извлечение ключевых сущностей"""
        entities = []
        
        # Технические термины
        technical_terms = [
            'modbus', 'rtu', 'tcp', 'rs485', 'rs232', 'plc', 'pmac', 'pid',
            'scada', 'hmi', 'opc', 'ethernet', 'протокол', 'контроллер',
            'датчик', 'привод', 'регулятор', 'автоматизация'
        ]
        
        for term in technical_terms:
            if term in message:
                entities.append(term)
        
        return entities
    
    def _assess_complexity(self, message: str) -> str:
        """Оценка сложности запроса"""
        for level, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if re.search(indicator, message):
                    return level
        return 'medium'
    
    def _determine_domain(self, intent: QueryIntent) -> str:
        """Определение домена"""
        domain_mapping = {
            QueryIntent.ELECTRICAL_ANALYSIS: 'electrical_engineering',
            QueryIntent.PROGRAMMING: 'software_development',
            QueryIntent.CONTROLLERS: 'industrial_automation',
            QueryIntent.RADIOMECHANICS: 'radio_engineering',
            QueryIntent.SCIENCE: 'science_education',
            QueryIntent.GENERAL: 'general_knowledge'
        }
        return domain_mapping.get(intent, 'general_knowledge')
    
    def _requires_examples(self, message: str) -> bool:
        """Проверка необходимости примеров"""
        example_indicators = [r'пример', r'как', r'покажи', r'демонстрац']
        return any(re.search(indicator, message) for indicator in example_indicators)
    
    def _requires_code(self, message: str) -> bool:
        """Проверка необходимости кода"""
        code_indicators = [r'код', r'программ', r'функция', r'класс', r'алгоритм']
        return any(re.search(indicator, message) for indicator in code_indicators)

class DatabaseSearchEngine:
    """Поисковик базы данных"""
    
    def __init__(self):
        self.knowledge_base = {
            'modbus_rtu': {
                'content': """🔌 **ПОЛНОЕ РУКОВОДСТВО ПО ПРОТОКОЛУ MODBUS RTU**

## 📋 **ОСНОВЫ MODBUS RTU**

### **1. ЧТО ТАКОЕ MODBUS RTU?**
Modbus RTU (Remote Terminal Unit) - это промышленный протокол связи, используемый для обмена данными между устройствами автоматизации.

**Основные характеристики:**
• **Стандарт:** Modbus over Serial Line (RS-485/RS-232)
• **Тип:** Master-Slave протокол
• **Скорость:** 1200-115200 бод
• **Расстояние:** до 1200 метров
• **Устройства:** до 247 устройств в сети

### **2. АРХИТЕКТУРА ПРОТОКОЛА**

**Структура сети:**
• **Master (Ведущий)** - инициирует запросы
• **Slave (Ведомый)** - отвечает на запросы
• **Адресация:** 1-247 (0 - broadcast, 248-255 - зарезервированы)

**Физический уровень:**
• **RS-485** - дифференциальная передача, 2 провода
• **RS-232** - последовательная связь, 3 провода
• **Терминаторы** - 120 Ом на концах линии

### **3. СТРУКТУРА КАДРА MODBUS RTU**

**Формат кадра:**
```
[Адрес] [Функция] [Данные] [CRC]
  1 байт   1 байт   N байт   2 байта
```

**Компоненты кадра:**
• **Адрес устройства** (1 байт) - 1-247
• **Код функции** (1 байт) - тип операции
• **Данные** (N байт) - параметры запроса
• **CRC** (2 байта) - контрольная сумма

### **4. ОСНОВНЫЕ ФУНКЦИИ MODBUS**

**Чтение данных:**
• **01 (0x01)** - Read Coils - чтение дискретных выходов
• **02 (0x02)** - Read Discrete Inputs - чтение дискретных входов
• **03 (0x03)** - Read Holding Registers - чтение регистров хранения
• **04 (0x04)** - Read Input Registers - чтение входных регистров

**Запись данных:**
• **05 (0x05)** - Write Single Coil - запись одного выхода
• **06 (0x06)** - Write Single Register - запись одного регистра
• **15 (0x0F)** - Write Multiple Coils - запись нескольких выходов
• **16 (0x10)** - Write Multiple Registers - запись нескольких регистров

**Какие конкретные аспекты Modbus RTU вас интересуют?** Я могу предоставить детальную информацию по любому разделу!""",
                'quality_score': 0.9,
                'relevance_score': 0.95,
                'metadata': {
                    'category': 'electrical_engineering',
                    'subcategory': 'industrial_protocols',
                    'difficulty': 'intermediate',
                    'last_updated': '2025-09-14'
                }
            }
        }
    
    def search(self, query: str, context: QueryContext) -> Optional[SearchResult]:
        """Поиск в базе данных"""
        query_lower = query.lower()
        
        # Поиск по ключевым словам
        for key, data in self.knowledge_base.items():
            if any(entity in query_lower for entity in context.entities):
                if key in query_lower or any(entity in key for entity in context.entities):
                    return SearchResult(
                        content=data['content'],
                        source='database',
                        quality_score=data['quality_score'],
                        relevance_score=data['relevance_score'],
                        metadata=data['metadata']
                    )
        
        return None

class ResponseAnalyzer:
    """Анализатор качества ответов"""
    
    def analyze_quality(self, response: str, query: str) -> QualityScore:
        """Анализ качества ответа"""
        
        # Оценка полноты
        completeness = self._assess_completeness(response, query)
        
        # Оценка точности
        accuracy = self._assess_accuracy(response)
        
        # Оценка релевантности
        relevance = self._assess_relevance(response, query)
        
        # Оценка ясности
        clarity = self._assess_clarity(response)
        
        # Общая оценка
        overall_score = (completeness + accuracy + relevance + clarity) / 4
        
        return QualityScore(
            completeness=completeness,
            accuracy=accuracy,
            relevance=relevance,
            clarity=clarity,
            overall_score=overall_score
        )
    
    def _assess_completeness(self, response: str, query: str) -> float:
        """Оценка полноты ответа"""
        # Проверяем наличие ключевых элементов
        completeness_indicators = [
            'что такое', 'основы', 'принцип работы', 'типы', 'применение',
            'примеры', 'схемы', 'расчеты', 'настройки'
        ]
        
        found_indicators = sum(1 for indicator in completeness_indicators 
                             if indicator in response.lower())
        
        return min(found_indicators / len(completeness_indicators), 1.0)
    
    def _assess_accuracy(self, response: str) -> float:
        """Оценка точности ответа"""
        # Проверяем наличие технических терминов и их корректность
        technical_terms = [
            'modbus', 'rtu', 'rs485', 'crc', 'master', 'slave',
            'регистр', 'функция', 'адрес', 'протокол'
        ]
        
        found_terms = sum(1 for term in technical_terms 
                         if term in response.lower())
        
        return min(found_terms / len(technical_terms), 1.0)
    
    def _assess_relevance(self, response: str, query: str) -> float:
        """Оценка релевантности ответа"""
        query_entities = query.lower().split()
        response_entities = response.lower().split()
        
        # Подсчитываем пересечение сущностей
        common_entities = set(query_entities) & set(response_entities)
        
        if not query_entities:
            return 0.0
        
        return len(common_entities) / len(query_entities)
    
    def _assess_clarity(self, response: str) -> float:
        """Оценка ясности ответа"""
        # Проверяем структурированность ответа
        clarity_indicators = [
            '##', '###', '•', '**', '```', 'таблица', 'схема'
        ]
        
        found_indicators = sum(1 for indicator in clarity_indicators 
                             if indicator in response)
        
        return min(found_indicators / len(clarity_indicators), 1.0)

class InternetSearchEngine:
    """Поисковик в интернете"""
    
    def search(self, query: str, context: QueryContext) -> SearchResult:
        """Поиск в интернете (заглушка)"""
        # В реальной реализации здесь был бы поиск в интернете
        internet_content = f"""
🌐 **ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ИЗ ИНТЕРНЕТА**

**Расширенная информация по {query}:**

• **Официальная документация:** Modbus.org
• **Стандарты:** IEC 61158, IEC 61784
• **Совместимость:** Широкая поддержка производителями
• **Безопасность:** Встроенные механизмы проверки CRC
• **Производительность:** Оптимизирован для промышленных сетей

**Современные тенденции:**
• Modbus TCP/IP для Ethernet сетей
• Modbus RTU over TCP для гибридных решений
• Интеграция с IoT платформами
• Облачные решения для удаленного мониторинга

**Практические рекомендации:**
• Используйте качественные терминаторы
• Обеспечьте правильное экранирование кабелей
• Регулярно проверяйте целостность сети
• Документируйте адресацию устройств
"""
        
        return SearchResult(
            content=internet_content,
            source='internet',
            quality_score=0.8,
            relevance_score=0.85,
            metadata={
                'search_engine': 'google',
                'timestamp': time.time(),
                'results_count': 1000
            }
        )

class FeedbackSystem:
    """Система обратной связи"""
    
    def __init__(self):
        self.feedback_history = []
    
    def process_feedback(self, response_id: str, feedback: str, user_rating: Optional[int] = None):
        """Обработка обратной связи от пользователя"""
        feedback_entry = {
            'response_id': response_id,
            'feedback': feedback,
            'rating': user_rating,
            'timestamp': time.time()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Анализ обратной связи
        if 'не то' in feedback.lower() or 'неверно' in feedback.lower():
            logger.info(f"❌ Пользователь указал на ошибку в ответе {response_id}")
            return {'action': 'search_alternative', 'reason': 'incorrect_answer'}
        elif 'больше информации' in feedback.lower() or 'подробнее' in feedback.lower():
            logger.info(f"📚 Пользователь запросил дополнительную информацию для ответа {response_id}")
            return {'action': 'search_additional', 'reason': 'needs_more_info'}
        elif user_rating and user_rating >= 4:
            logger.info(f"✅ Пользователь высоко оценил ответ {response_id}")
            return {'action': 'reinforce', 'reason': 'positive_feedback'}
        else:
            logger.info(f"🔄 Получена обратная связь для ответа {response_id}: {feedback}")
            return {'action': 'analyze', 'reason': 'general_feedback'}

class SmartDispatcher:
    """Улучшенный интеллектуальный диспетчер"""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.db_search_engine = DatabaseSearchEngine()
        self.response_analyzer = ResponseAnalyzer()
        self.internet_search_engine = InternetSearchEngine()
        self.feedback_system = FeedbackSystem()
        self.quality_threshold = 0.8
    
    def process_query(self, message: str) -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        logger.info(f"🔍 Обрабатываем запрос: {message[:100]}...")
        
        # Этап 1: Анализ запроса
        context = self.query_analyzer.analyze_query(message)
        logger.info(f"📋 Контекст: {context.intent.value}, сложность: {context.complexity}")
        
        # Этап 2: Поиск в базе данных
        db_result = self.db_search_engine.search(message, context)
        
        if db_result:
            logger.info(f"✅ Найден ответ в базе данных, качество: {db_result.quality_score}")
            
            # Этап 3: Анализ качества
            quality = self.response_analyzer.analyze_quality(db_result.content, message)
            logger.info(f"📊 Оценка качества: {quality.overall_score:.2f}")
            
            # Этап 4: Принятие решения
            if quality.overall_score >= self.quality_threshold:
                logger.info("✅ Качество достаточное, отправляем ответ из БД")
                return self._format_response(db_result, quality, context)
            else:
                logger.info("⚠️ Качество недостаточное, ищем в интернете")
                internet_result = self.internet_search_engine.search(message, context)
                internet_quality = self.response_analyzer.analyze_quality(internet_result.content, message)
                
                # Выбираем лучший ответ
                if internet_quality.overall_score > quality.overall_score:
                    logger.info("🌐 Интернет-ответ лучше, используем его")
                    return self._format_response(internet_result, internet_quality, context)
                else:
                    logger.info("💾 Ответ из БД лучше, используем его")
                    return self._format_response(db_result, quality, context)
        else:
            logger.info("❌ Ответ не найден в БД, ищем в интернете")
            internet_result = self.internet_search_engine.search(message, context)
            internet_quality = self.response_analyzer.analyze_quality(internet_result.content, message)
            
            return self._format_response(internet_result, internet_quality, context)
    
    def process_feedback(self, response_id: str, feedback: str, rating: Optional[int] = None):
        """Обработка обратной связи"""
        return self.feedback_system.process_feedback(response_id, feedback, rating)
    
    def _format_response(self, result: SearchResult, quality: QualityScore, context: QueryContext) -> Dict[str, Any]:
        """Форматирование ответа"""
        return {
            'response': result.content,
            'provider': 'Smart Dispatcher',
            'category': context.intent.value,
            'quality_score': quality.overall_score,
            'source': result.source,
            'metadata': {
                'intent': context.intent.value,
                'complexity': context.complexity,
                'domain': context.domain,
                'entities': context.entities,
                'quality_breakdown': {
                    'completeness': quality.completeness,
                    'accuracy': quality.accuracy,
                    'relevance': quality.relevance,
                    'clarity': quality.clarity
                }
            },
            'thinking_process': [
                f"Анализ запроса: {context.intent.value}",
                f"Поиск в источнике: {result.source}",
                f"Оценка качества: {quality.overall_score:.2f}",
                f"Формирование ответа"
            ],
            'timestamp': time.time(),
            'success': True
        }

# Пример использования
if __name__ == "__main__":
    dispatcher = SmartDispatcher()
    
    # Тестовый запрос
    test_query = "Опиши протокол Modbus RTU"
    
    print("🚀 Тестирование улучшенного диспетчера")
    print("=" * 50)
    
    # Обработка запроса
    result = dispatcher.process_query(test_query)
    
    print(f"📋 Категория: {result['category']}")
    print(f"📊 Качество: {result['quality_score']:.2f}")
    print(f"🔍 Источник: {result['source']}")
    print(f"💭 Процесс мышления: {result['thinking_process']}")
    print("\n📝 Ответ:")
    print(result['response'][:500] + "...")
    
    # Тестирование обратной связи
    print("\n🔄 Тестирование обратной связи")
    print("=" * 50)
    
    feedback_result = dispatcher.process_feedback("test_123", "Нужно больше информации")
    print(f"Действие: {feedback_result['action']}")
    print(f"Причина: {feedback_result['reason']}")


















