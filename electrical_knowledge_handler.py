#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработчик знаний по электротехнике для Rubin AI
Специализированные ответы на вопросы о защите цепей, электронике и электротехнике
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

class ElectricalKnowledgeHandler:
    """Обработчик знаний по электротехнике"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # База знаний по электротехнике
        self.knowledge_base = {
            'short_circuit_protection': {
                'keywords': [
                    'защита', 'короткое замыкание', 'кз', 'цепи', 'электрические',
                    'protection', 'short circuit', 'circuits', 'electrical',
                    'предохранитель', 'автомат', 'узо', 'fuse', 'breaker'
                ],
                'response_ru': """🛡️ **Защита электрических цепей от короткого замыкания:**

**Что такое короткое замыкание:**
Соединение двух точек цепи с разными потенциалами через очень малое сопротивление, что приводит к резкому увеличению тока.

**Опасности КЗ:**
• **Перегрев проводников** - возможен пожар
• **Повреждение оборудования** - выход из строя
• **Поражение электрическим током** - опасность для людей
• **Дуговые разряды** - взрывоопасность

**Средства защиты:**

**1. Предохранители:**
• **Плавкие вставки** - одноразовые, точные
• **Время срабатывания** - мгновенное при КЗ
• **Номиналы** - от мА до кА
• **Применение** - бытовые и промышленные сети

**2. Автоматические выключатели:**
• **Тепловая защита** - от перегрузки
• **Электромагнитная защита** - от КЗ
• **Многократное использование** - можно включать заново
• **Характеристики** - B, C, D (время-токовые)

**3. Дифференциальные автоматы (УЗО):**
• **Защита от утечек** - ток утечки на землю
• **Защита людей** - от поражения током
• **Чувствительность** - 10мА, 30мА, 100мА, 300мА
• **Время срабатывания** - < 30мс

**Расчет защиты:**
• **Iном.защиты ≥ Iраб.макс** (рабочий ток)
• **Iном.защиты ≤ Iдоп.провода** (допустимый ток провода)

✅ **Правильная защита = безопасность и надежность!**"""  
          },
            
            'power_factor': {
                'keywords': [
                    'коэффициент мощности', 'cos φ', 'cos phi', 'power factor',
                    'реактивная мощность', 'reactive power', 'компенсация'
                ],
                'response_ru': """⚡ **Коэффициент мощности (cos φ):**

**Что это:**
Коэффициент мощности показывает эффективность использования электроэнергии.

**Формула:** cos φ = P / S
• P - активная мощность (Вт)  
• S - полная мощность (ВА)
• φ - угол сдвига фаз между током и напряжением

**Типы мощности:**
• **Активная (P)** - полезная мощность, Вт
• **Реактивная (Q)** - "бесполезная" мощность, ВАр
• **Полная (S)** - общая мощность, ВА
• **S² = P² + Q²** (треугольник мощностей)

**Значения cos φ:**
• **cos φ = 1** - идеальная нагрузка (только активная)
• **cos φ = 0.9-0.95** - хорошо
• **cos φ = 0.7-0.9** - удовлетворительно
• **cos φ < 0.7** - плохо, требует коррекции

**Как улучшить cos φ:**
1. **Конденсаторные батареи** - компенсация реактивной мощности
2. **Синхронные двигатели** - работа с опережающим cos φ
3. **Активные фильтры** - для нелинейных нагрузок
4. **Правильный выбор оборудования**

**Преимущества высокого cos φ:**
• Снижение потерь в сети
• Уменьшение тока в проводах
• Экономия электроэнергии
• Улучшение качества напряжения

✅ **Хороший cos φ = экономия электроэнергии!**"""
            },
            
            'electronic_components': {
                'keywords': [
                    'транзистор', 'диод', 'резистор', 'конденсатор', 'индуктивность',
                    'transistor', 'diode', 'resistor', 'capacitor', 'inductor',
                    'компоненты', 'элементы', 'components'
                ],
                'response_ru': """🔌 **Основные электронные компоненты:**

**1. Резистор:**
• **Функция:** Ограничение тока, деление напряжения
• **Закон Ома:** U = I × R
• **Мощность:** P = I² × R = U²/R
• **Типы:** Постоянные, переменные, подстроечные
• **Применение:** Токоограничение, делители напряжения

**2. Конденсатор:**
• **Функция:** Накопление электрической энергии
• **Формула:** Q = C × U, E = ½CU²
• **Типы:** Электролитические, керамические, пленочные
• **Применение:** Фильтрация, развязка, времязадающие цепи

**3. Индуктивность (катушка):**
• **Функция:** Накопление магнитной энергии
• **Формула:** E = ½LI², U = L × dI/dt
• **Применение:** Фильтры, трансформаторы, дроссели

**4. Диод:**
• **Функция:** Пропускает ток только в одном направлении
• **Типы:** Выпрямительные, стабилитроны, светодиоды
• **Применение:** Выпрямление, стабилизация, защита

**5. Транзистор:**
• **Функция:** Усиление и переключение сигналов
• **Типы:** Биполярные (NPN, PNP), полевые (MOSFET, JFET)
• **Применение:** Усилители, ключи, генераторы

✅ **Основа всей современной электроники!**"""
            },
            
            'electrical_laws': {
                'keywords': [
                    'закон ома', 'кирхгоф', 'ohm law', 'kirchhoff',
                    'напряжение', 'ток', 'сопротивление', 'мощность',
                    'voltage', 'current', 'resistance', 'power'
                ],
                'response_ru': """⚡ **Основные законы электротехники:**

**Закон Ома:**
• **U = I × R** (напряжение = ток × сопротивление)
• **I = U / R** (ток = напряжение / сопротивление)
• **R = U / I** (сопротивление = напряжение / ток)

**Мощность:**
• **P = U × I** (мощность = напряжение × ток)
• **P = I² × R** (через ток и сопротивление)
• **P = U² / R** (через напряжение и сопротивление)

**Законы Кирхгофа:**
• **1-й закон (токов):** Сумма токов в узле = 0
  ΣI = 0 (алгебраическая сумма)
• **2-й закон (напряжений):** Сумма напряжений в контуре = 0
  ΣU = 0 (алгебраическая сумма)

**Соединения резисторов:**
• **Последовательное:** R = R1 + R2 + R3...
• **Параллельное:** 1/R = 1/R1 + 1/R2 + 1/R3...

**Единицы измерения:**
• Напряжение: Вольт (В)
• Ток: Ампер (А)
• Сопротивление: Ом (Ω)
• Мощность: Ватт (Вт)
• Энергия: Джоуль (Дж), кВт·ч

✅ **Основа всей электротехники!**"""
            },
            
            'motors': {
                'keywords': [
                    'двигатель', 'мотор', 'асинхронный', 'синхронный',
                    'motor', 'asynchronous', 'synchronous', 'servo',
                    'привод', 'частотный преобразователь'
                ],
                'response_ru': """⚙️ **Электрические двигатели:**

**Асинхронный двигатель:**
• **Принцип:** Ротор "догоняет" вращающееся магнитное поле статора
• **Скольжение:** s = (n₀ - n) / n₀
• **Преимущества:** Простота, надежность, низкая стоимость
• **Недостатки:** Сложность регулирования скорости
• **Применение:** Насосы, вентиляторы, конвейеры

**Синхронный двигатель:**
• **Принцип:** Ротор вращается синхронно с полем статора
• **Преимущества:** Постоянная скорость, высокий КПД
• **Недостатки:** Сложность пуска, необходимость возбуждения
• **Применение:** Генераторы, точные приводы

**Сервопривод:**
• **Принцип:** Двигатель с обратной связью по положению
• **Компоненты:** Двигатель + энкодер + контроллер
• **Преимущества:** Высокая точность позиционирования
• **Применение:** Станки ЧПУ, роботы, автоматизация

**Частотный преобразователь:**
• **Функция:** Плавное регулирование скорости АД
• **Принцип:** Изменение частоты и амплитуды питающего напряжения
• **Преимущества:** Энергосбережение, плавный пуск
• **Применение:** Насосы, вентиляторы, конвейеры

✅ **Правильный выбор двигателя = эффективная система!**"""
            }
        }
        
        self.logger.info("Обработчик знаний по электротехнике инициализирован")
    
    def detect_topic(self, message: str) -> Optional[str]:
        """Определение темы электротехнического вопроса"""
        message_lower = message.lower()
        
        # Проверяем каждую тему в порядке приоритета
        topic_priorities = [
            'short_circuit_protection',  # Высший приоритет - исправляем проблему
            'power_factor',
            'electronic_components', 
            'electrical_laws',
            'motors'
        ]
        
        best_topic = None
        max_matches = 0
        
        for topic in topic_priorities:
            keywords = self.knowledge_base[topic]['keywords']
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            
            # Специальная обработка для защиты от КЗ
            if topic == 'short_circuit_protection':
                if any(phrase in message_lower for phrase in [
                    'как защитить', 'защита цепей', 'короткое замыкание',
                    'how to protect', 'circuit protection', 'short circuit'
                ]):
                    return topic  # Немедленно возвращаем для исправления проблемы
            
            if matches > max_matches:
                max_matches = matches
                best_topic = topic
        
        return best_topic if max_matches >= 1 else None
    
    def detect_language(self, message: str) -> str:
        """Определение языка сообщения"""
        cyrillic_chars = len(re.findall(r'[а-яё]', message.lower()))
        latin_chars = len(re.findall(r'[a-z]', message.lower()))
        
        return 'ru' if cyrillic_chars > latin_chars else 'en'
    
    def handle_request(self, message: str) -> Dict:
        """Обработка электротехнического запроса"""
        try:
            # Определяем тему и язык
            topic = self.detect_topic(message)
            language = self.detect_language(message)
            
            if not topic:
                # Общий ответ по электротехнике
                return self._get_general_electrical_response(language)
            
            # Получаем специализированный ответ
            knowledge = self.knowledge_base[topic]
            response_text = knowledge['response_ru']  # Пока только русский
            
            self.logger.info(f"Обработан электротехнический запрос: тема='{topic}', язык='{language}'")
            
            return {
                'response': response_text,
                'provider': 'Electrical Knowledge Handler',
                'category': 'electrical',
                'topic': topic,
                'language': language,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки электротехнического запроса: {e}")
            return {
                'response': f'Произошла ошибка при обработке электротехнического вопроса: {str(e)}',
                'provider': 'Electrical Knowledge Handler',
                'category': 'electrical',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_general_electrical_response(self, language: str) -> Dict:
        """Общий ответ по электротехнике"""
        if language == 'ru':
            response = """⚡ **Электротехника - моя специализация!**

**Основные области:**
• **Защита электрических цепей** - предохранители, автоматы, УЗО
• **Электронные компоненты** - резисторы, конденсаторы, транзисторы
• **Электрические машины** - двигатели, генераторы, трансформаторы
• **Силовая электроника** - преобразователи, инверторы, выпрямители
• **Измерения и контроль** - датчики, измерительные приборы
• **Электробезопасность** - заземление, защитные меры

**Могу помочь с:**
• Расчетом электрических цепей
• Выбором защитных устройств
• Принципами работы компонентов
• Электробезопасностью
• Энергосбережением и эффективностью

**Задайте конкретный вопрос** - например:
• "Как защитить цепи от короткого замыкания?"
• "Что такое коэффициент мощности?"
• "Принцип работы трансформатора"

✅ **Готов помочь с любыми вопросами по электротехнике!**"""
        else:
            response = """⚡ **Electrical Engineering - My Specialization!**

**Main Areas:**
• **Electrical Circuit Protection** - fuses, breakers, RCDs
• **Electronic Components** - resistors, capacitors, transistors
• **Electrical Machines** - motors, generators, transformers
• **Power Electronics** - converters, inverters, rectifiers
• **Measurements and Control** - sensors, measuring instruments
• **Electrical Safety** - grounding, protective measures

**I can help with:**
• Electrical circuit calculations
• Protective device selection
• Component working principles
• Electrical safety
• Energy saving and efficiency

**Ask a specific question** - for example:
• "How to protect circuits from short circuit?"
• "What is power factor?"
• "Transformer working principle"

✅ **Ready to help with any electrical engineering questions!**"""
        
        return {
            'response': response,
            'provider': 'Electrical Knowledge Handler',
            'category': 'electrical',
            'topic': 'general',
            'language': language,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_supported_topics(self) -> List[str]:
        """Получение списка поддерживаемых тем"""
        return list(self.knowledge_base.keys())
    
    def get_statistics(self) -> Dict:
        """Получение статистики обработчика"""
        return {
            'supported_topics': len(self.knowledge_base),
            'topics': list(self.knowledge_base.keys()),
            'supported_languages': ['ru', 'en']
        }

# Глобальный экземпляр
_electrical_handler_instance = None

def get_electrical_handler() -> ElectricalKnowledgeHandler:
    """Получение глобального экземпляра обработчика"""
    global _electrical_handler_instance
    if _electrical_handler_instance is None:
        _electrical_handler_instance = ElectricalKnowledgeHandler()
    return _electrical_handler_instance

# Тестирование
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    handler = ElectricalKnowledgeHandler()
    
    test_questions = [
        "Как защитить электрические цепи от короткого замыкания?",
        "How to protect electrical circuits from short circuit?",
        "Что такое коэффициент мощности?",
        "Принцип работы транзистора",
        "Закон Ома",
        "Асинхронный двигатель",
        "Общий вопрос по электротехнике"
    ]
    
    print("=== ТЕСТИРОВАНИЕ ОБРАБОТЧИКА ЭЛЕКТРОТЕХНИКИ ===")
    for question in test_questions:
        print(f"\nВопрос: {question}")
        result = handler.handle_request(question)
        print(f"Тема: {result.get('topic', 'N/A')}")
        print(f"Язык: {result.get('language', 'N/A')}")
        print(f"Успех: {result.get('success', False)}")
        print("Ответ:", result['response'][:200] + "..." if len(result['response']) > 200 else result['response'])
    
    print(f"\nСтатистика: {handler.get_statistics()}")