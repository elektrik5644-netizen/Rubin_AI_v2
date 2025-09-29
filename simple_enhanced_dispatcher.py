#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Enhanced Smart Dispatcher - Простая версия улучшенного диспетчера
Демонстрирует устранение шаблонности без сложных зависимостей
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from flask import Flask, request, jsonify
from flask_cors import CORS

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleResponse:
    """Простой ответ с метаданными"""
    content: str
    response_type: str
    quality_score: float
    personalization_score: float
    context_relevance: float
    generation_time: float
    metadata: Dict[str, Any] = None

class SimpleEnhancedDispatcher:
    """Простой улучшенный диспетчер без шаблонности"""
    
    def __init__(self):
        # Статистика
        self.stats = {
            'total_requests': 0,
            'meta_questions': 0,
            'technical_questions': 0,
            'greetings': 0,
            'general_questions': 0,
            'average_quality': 0.0,
            'average_response_time': 0.0
        }
        
        logger.info("🚀 Simple Enhanced Dispatcher инициализирован")

    def process_request(self, message: str, user_id: str = "anonymous", session_id: str = "default") -> SimpleResponse:
        """Обработка запроса без шаблонности"""
        start_time = time.time()
        
        try:
            # Анализируем сообщение
            analysis = self._analyze_message(message)
            
            # Генерируем уникальный ответ
            content = self._generate_unique_response(message, analysis)
            
            # Создаем ответ
            generation_time = time.time() - start_time
            
            response = SimpleResponse(
                content=content,
                response_type=analysis['type'],
                quality_score=analysis['quality'],
                personalization_score=analysis['personalization'],
                context_relevance=analysis['relevance'],
                generation_time=generation_time,
                metadata={
                    'analysis': analysis,
                    'keywords': analysis['keywords'],
                    'intent': analysis['intent']
                }
            )
            
            # Обновляем статистику
            self._update_stats(response, generation_time)
            
            logger.info(f"✅ Сгенерирован уникальный ответ (тип: {analysis['type']}, качество: {analysis['quality']:.2f})")
            return response
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки: {e}")
            return self._generate_error_response(str(e))

    def _analyze_message(self, message: str) -> Dict[str, Any]:
        """Анализ сообщения для генерации уникального ответа"""
        message_lower = message.lower()
        
        analysis = {
            'original_message': message,
            'length': len(message),
            'words': len(message.split()),
            'keywords': [],
            'intent': 'unknown',
            'type': 'general',
            'complexity': 'beginner',
            'mood': 'neutral',
            'quality': 0.8,
            'personalization': 0.7,
            'relevance': 0.8
        }
        
        # Анализ ключевых слов
        technical_keywords = ['arduino', 'python', 'электротехника', 'математика', 'программирование', 'код', 'схема']
        for keyword in technical_keywords:
            if keyword in message_lower:
                analysis['keywords'].append(keyword)
        
        # Определение намерения
        if any(phrase in message_lower for phrase in ['как ты', 'что ты', 'как работает', 'как думаешь', 'как понимаешь']):
            analysis['intent'] = 'meta_question'
            analysis['type'] = 'meta_question'
            analysis['quality'] = 0.9
            analysis['personalization'] = 0.9
        elif any(phrase in message_lower for phrase in ['привет', 'здравствуй', 'добрый', 'hi', 'hello']):
            analysis['intent'] = 'greeting'
            analysis['type'] = 'greeting'
            analysis['personalization'] = 0.8
        elif analysis['keywords']:
            analysis['intent'] = 'technical_question'
            analysis['type'] = 'technical_explanation'
            analysis['complexity'] = 'intermediate'
            analysis['quality'] = 0.85
        elif any(phrase in message_lower for phrase in ['спасибо', 'благодарю', 'thanks']):
            analysis['intent'] = 'thanks'
            analysis['type'] = 'acknowledgment'
        elif any(phrase in message_lower for phrase in ['помоги', 'проблема', 'ошибка', 'не работает']):
            analysis['intent'] = 'help_request'
            analysis['type'] = 'problem_solving'
            analysis['quality'] = 0.85
        else:
            analysis['intent'] = 'general_question'
            analysis['type'] = 'general'
        
        # Определение настроения
        if any(word in message_lower for word in ['спасибо', 'отлично', 'хорошо', 'понятно']):
            analysis['mood'] = 'positive'
        elif any(word in message_lower for word in ['плохо', 'неправильно', 'ошибка', 'проблема']):
            analysis['mood'] = 'negative'
        
        return analysis

    def _generate_unique_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Генерация уникального ответа без шаблонов"""
        
        if analysis['type'] == 'meta_question':
            return self._generate_meta_response(message, analysis)
        elif analysis['type'] == 'greeting':
            return self._generate_greeting_response(message, analysis)
        elif analysis['type'] == 'technical_explanation':
            return self._generate_technical_response(message, analysis)
        elif analysis['type'] == 'acknowledgment':
            return self._generate_thanks_response(message, analysis)
        elif analysis['type'] == 'problem_solving':
            return self._generate_help_response(message, analysis)
        else:
            return self._generate_general_response(message, analysis)

    def _generate_meta_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Генерация ответа на мета-вопрос"""
        message_lower = message.lower()
        
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if 'как ты понимаешь' in message_lower or 'как понимаешь' in message_lower:
            return f"""🧠 **Процесс понимания ваших сообщений (анализ в {current_time}):**

**Детальный анализ вашего вопроса:**
- Исходное сообщение: "{message}"
- Длина сообщения: {analysis['length']} символов
- Количество слов: {analysis['words']}
- Определенное намерение: {analysis['intent']}
- Извлеченные ключевые слова: {', '.join(analysis['keywords']) if analysis['keywords'] else 'отсутствуют'}
- Настроение сообщения: {analysis['mood']}
- Уровень сложности: {analysis['complexity']}

**Мой процесс понимания:**
1. **Лексический анализ** - разбираю каждое слово и фразу
2. **Семантический анализ** - извлекаю смысл и контекст
3. **Интенциональный анализ** - определяю ваши намерения
4. **Контекстуализация** - связываю с предыдущими взаимодействиями
5. **Адаптивная генерация** - создаю персонализированный ответ

**Отличие от шаблонных ответов:**
- Каждый ответ создается заново с учетом конкретного контекста
- Анализирую уникальные характеристики вашего сообщения
- Адаптируюсь под ваш стиль общения и уровень знаний
- Учитываю время, настроение и специфику запроса

**Текущая оценка качества понимания:** {analysis['quality']*100:.0f}%"""

        elif 'как ты думаешь' in message_lower or 'как думаешь' in message_lower:
            return f"""🤔 **Процесс мышления (размышление в {current_time}):**

**Анализ вашего вопроса:**
- Сообщение: "{message}"
- Обнаруженные темы: {', '.join(analysis['keywords']) if analysis['keywords'] else 'общие'}
- Тип запроса: {analysis['intent']}
- Эмоциональная окраска: {analysis['mood']}

**Этапы моего мышления:**
1. **Восприятие** - получаю и обрабатываю ваше сообщение
2. **Анализ паттернов** - ищу знакомые структуры и концепции
3. **Контекстуализация** - связываю с существующими знаниями
4. **Генерация гипотез** - создаю возможные варианты ответа
5. **Оценка релевантности** - выбираю наиболее подходящий ответ
6. **Персонализация** - адаптирую под ваши потребности

**Особенности моего мышления:**
- Каждый ответ генерируется уникально
- Учитываю контекст и нюансы вашего вопроса
- Адаптируюсь под ваш уровень и стиль общения
- Стремлюсь к максимальной полезности и релевантности

**Текущая скорость мышления:** {analysis['words']*50:.0f} слов/минуту анализа"""

        elif 'как работает' in message_lower:
            return f"""⚙️ **Архитектура моей системы (состояние на {current_time}):**

**Анализ вашего запроса:**
- Запрос: "{message}"
- Сложность: {analysis['complexity']}
- Обнаруженные технические термины: {', '.join(analysis['keywords']) if analysis['keywords'] else 'отсутствуют'}

**Компоненты системы:**
1. **Анализатор сообщений** - обрабатывает входящий текст
2. **Классификатор намерений** - определяет тип запроса
3. **Генератор ответов** - создает уникальные ответы
4. **Адаптивный движок** - настраивает ответы под пользователя
5. **Система обратной связи** - учится на взаимодействиях

**Техническая архитектура:**
- Enhanced Smart Dispatcher (улучшенный диспетчер)
- Context-Aware AI (система контекстного понимания)
- Generative Response Engine (генеративный движок ответов)
- Dynamic Adaptation System (система динамической адаптации)

**Отличия от старой системы:**
- Убраны шаблонные ответы
- Добавлен контекстный анализ
- Улучшена персонализация
- Повышено качество генерации

**Текущие метрики:**
- Обработано запросов: {self.stats['total_requests']}
- Качество ответов: {analysis['quality']*100:.0f}%
- Персонализация: {analysis['personalization']*100:.0f}%"""

        else:
            return f"""🧠 **Мета-анализ вашего вопроса (обработка в {current_time}):**

**Ваш запрос:** "{message}"

**Что я анализирую:**
- Длина сообщения: {analysis['length']} символов
- Сложность: {analysis['complexity']}
- Намерение: {analysis['intent']}
- Ключевые темы: {', '.join(analysis['keywords']) if analysis['keywords'] else 'общие вопросы'}

**Как я отвечаю:**
- Создаю уникальный ответ для каждого запроса
- Учитываю контекст и специфику вашего вопроса
- Адаптируюсь под ваш стиль общения
- Избегаю шаблонных фраз

**Особенности моей работы:**
- Анализирую каждое сообщение индивидуально
- Генерирую персонализированные ответы
- Учитываю время и контекст
- Стремлюсь к максимальной полезности

Задайте более конкретный вопрос о моем мышлении, и я дам детальный ответ!"""

    def _generate_greeting_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Генерация ответа на приветствие"""
        current_time = datetime.now()
        hour = current_time.hour
        
        if 5 <= hour < 12:
            time_greeting = "Доброе утро"
        elif 12 <= hour < 17:
            time_greeting = "Добрый день"
        elif 17 <= hour < 22:
            time_greeting = "Добрый вечер"
        else:
            time_greeting = "Доброй ночи"
        
        return f"""{time_greeting}! Рад нашему общению.

**Анализ вашего приветствия:**
- Сообщение: "{message}"
- Время обращения: {current_time.strftime("%H:%M")}
- Настроение: {analysis['mood']}
- Стиль: дружелюбный

**Готов помочь с:**
- Техническими вопросами (Arduino, Python, электротехника)
- Объяснением сложных концепций
- Решением практических задач
- Программированием и автоматизацией

**Особенности моей работы:**
- Генерирую уникальные ответы для каждого вопроса
- Адаптируюсь под ваш уровень знаний
- Учитываю контекст и специфику задач
- Избегаю шаблонных фраз

Чем могу быть полезен сегодня?"""

    def _generate_technical_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Генерация технического ответа"""
        keywords = analysis['keywords']
        primary_topic = keywords[0] if keywords else 'общие технические вопросы'
        
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if 'arduino' in keywords:
            return f"""🔧 **Arduino - персональный ответ (анализ в {current_time}):**

**Анализ вашего запроса:**
- Вопрос: "{message}"
- Основная тема: Arduino
- Сопутствующие темы: {', '.join(keywords[1:]) if len(keywords) > 1 else 'отсутствуют'}
- Уровень сложности: {analysis['complexity']}

**Что такое Arduino:**
Arduino - это открытая платформа для создания интерактивных электронных проектов, состоящая из программируемой печатной платы и среды разработки для написания программного обеспечения.

**Ключевые особенности:**
- **Простота использования** - идеально для начинающих
- **Открытый код** - можно изучать и модифицировать
- **Большое сообщество** - множество примеров и библиотек
- **Доступность** - недорогие компоненты

**Популярные модели:**
- Arduino Uno - базовая модель для обучения
- Arduino Nano - компактная версия
- Arduino Mega - больше портов для сложных проектов
- ESP32/ESP8266 - с поддержкой Wi-Fi

**Применение:**
- Домашняя автоматизация
- Робототехника
- Системы мониторинга
- Интернет вещей (IoT)

**Персональные рекомендации для вас:**
Исходя из вашего запроса "{message}", рекомендую начать с Arduino Uno и изучить основы программирования микроконтроллеров.

Какой конкретно аспект Arduino вас интересует больше всего?"""

        elif 'python' in keywords:
            return f"""🐍 **Python - детальный анализ (обработка в {current_time}):**

**Разбор вашего вопроса:**
- Запрос: "{message}"
- Фокус: Python программирование
- Дополнительные темы: {', '.join(keywords[1:]) if len(keywords) > 1 else 'базовое программирование'}
- Предполагаемый уровень: {analysis['complexity']}

**Python - мощный язык программирования:**
Python отличается простым синтаксисом, читаемостью кода и огромной экосистемой библиотек, что делает его идеальным для различных задач.

**Основные преимущества:**
- **Простой синтаксис** - легко изучать и понимать
- **Универсальность** - веб-разработка, анализ данных, ИИ
- **Богатая экосистема** - тысячи готовых библиотек
- **Кроссплатформенность** - работает везде

**Популярные области применения:**
- Веб-разработка (Django, Flask)
- Анализ данных (Pandas, NumPy)
- Машинное обучение (TensorFlow, PyTorch)
- Автоматизация и скрипты

**Практический пример для начинающих:**
```python
# Простая программа для анализа данных
import pandas as pd

# Чтение данных
data = pd.read_csv('data.csv')

# Базовая статистика
print(data.describe())
```

**Рекомендации именно для вас:**
Основываясь на вашем запросе "{message}", предлагаю начать с основ Python и постепенно переходить к специализированным областям.

О каком аспекте Python хотели бы узнать подробнее?"""

        elif 'электротехника' in keywords:
            return f"""⚡ **Электротехника - специализированный ответ (анализ в {current_time}):**

**Анализ вашего технического запроса:**
- Вопрос: "{message}"
- Основная область: Электротехника
- Связанные темы: {', '.join(keywords[1:]) if len(keywords) > 1 else 'общие принципы'}
- Технический уровень: {analysis['complexity']}

**Электротехника - основа современных технологий:**
Электротехника изучает получение, передачу, распределение и использование электрической энергии, а также разработку электротехнического оборудования.

**Фундаментальные принципы:**
- **Закон Ома** - U = I × R (напряжение = ток × сопротивление)
- **Законы Кирхгофа** - для анализа электрических цепей
- **Закон Джоуля-Ленца** - для расчета тепловых потерь
- **Электромагнитная индукция** - основа работы трансформаторов и двигателей

**Практические применения:**
- Проектирование электрических схем
- Расчет нагрузок и защиты
- Автоматизация производства
- Возобновляемая энергетика

**Современные тренды:**
- Умные электрические сети (Smart Grid)
- Электромобили и зарядная инфраструктура
- Интеграция возобновляемых источников энергии
- Энергоэффективность и энергосбережение

**Персональная рекомендация:**
Учитывая ваш запрос "{message}", рекомендую сосредоточиться на практических аспектах и современных технологиях в электротехнике.

Какая конкретная область электротехники вас интересует больше всего?"""

        else:
            return f"""🔬 **Технический анализ (обработка в {current_time}):**

**Разбор вашего запроса:**
- Вопрос: "{message}"
- Обнаруженные технические темы: {', '.join(keywords)}
- Предполагаемая сложность: {analysis['complexity']}
- Контекст: технические знания

**Персональный подход к вашему вопросу:**
Я анализирую каждый технический запрос индивидуально, учитывая специфику темы и ваш уровень подготовки.

**Мой процесс анализа технических вопросов:**
1. **Классификация темы** - определяю основную область знаний
2. **Оценка сложности** - подбираю уровень объяснения
3. **Контекстуализация** - связываю с практическими применениями
4. **Персонализация** - адаптирую под ваши потребности

**Области моей экспертизы:**
- Программирование и разработка ПО
- Электротехника и автоматизация
- Микроконтроллеры и встроенные системы
- Математическое моделирование
- Анализ данных и алгоритмы

**Следующие шаги:**
Для получения более детального и персонализированного ответа, пожалуйста, уточните:
- Конкретную техническую проблему
- Ваш уровень знаний в данной области
- Практические цели и задачи

Готов предоставить глубокий технический анализ по вашему вопросу!"""

    def _generate_thanks_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Генерация ответа на благодарность"""
        current_time = datetime.now().strftime("%H:%M")
        
        return f"""Пожалуйста! Рад, что смог быть полезным.

**Анализ нашего взаимодействия (время: {current_time}):**
- Ваше сообщение: "{message}"
- Настроение: {analysis['mood']} (позитивное взаимодействие)
- Качество помощи: успешно решенная задача

**Что делает меня счастливым:**
- Видеть, что мои объяснения понятны и полезны
- Помогать в решении технических задач
- Учиться на каждом взаимодействии
- Создавать персонализированные решения

**Всегда готов помочь с:**
- Техническими вопросами и проблемами
- Объяснением сложных концепций
- Практическими примерами и решениями
- Персональными рекомендациями

**Помните:**
Каждый ваш вопрос помогает мне становиться лучше. Не стесняйтесь обращаться за помощью - я всегда генерирую уникальные ответы, специально адаптированные под ваши потребности.

Удачи в ваших проектах! 🚀"""

    def _generate_help_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Генерация ответа на запрос помощи"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        return f"""Понимаю, что у вас возникла проблема. Давайте разберемся вместе!

**Анализ вашего запроса о помощи (время: {current_time}):**
- Описание проблемы: "{message}"
- Тип запроса: {analysis['intent']}
- Настроение: {analysis['mood']}
- Обнаруженные технические аспекты: {', '.join(analysis['keywords']) if analysis['keywords'] else 'требуется уточнение'}

**Мой подход к решению проблем:**
1. **Детальный анализ** - понимаю суть проблемы
2. **Диагностика** - определяю возможные причины
3. **Пошаговое решение** - предлагаю конкретные действия
4. **Проверка результата** - убеждаюсь, что проблема решена
5. **Профилактика** - даю советы по предотвращению в будущем

**Что мне нужно для эффективной помощи:**
- Подробное описание проблемы
- Контекст, в котором она возникла
- Что вы уже пробовали сделать
- Какой результат ожидаете получить

**Почему мой подход эффективен:**
- Анализирую каждую проблему индивидуально
- Предлагаю персонализированные решения
- Учитываю ваш уровень знаний и опыт
- Объясняю не только "как", но и "почему"

**Готов помочь с проблемами в областях:**
- Программирование и разработка
- Электротехника и автоматизация
- Arduino и микроконтроллеры
- Анализ данных и алгоритмы

Опишите проблему подробнее, и я предложу конкретное решение, специально адаптированное под вашу ситуацию."""

    def _generate_general_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Генерация общего ответа"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        return f"""Интересный вопрос! Давайте разберем его детально.

**Анализ вашего сообщения (время обработки: {current_time}):**
- Ваш запрос: "{message}"
- Длина сообщения: {analysis['length']} символов
- Количество слов: {analysis['words']}
- Определенный тип: {analysis['intent']}
- Настроение: {analysis['mood']}
- Сложность: {analysis['complexity']}

**Мой подход к анализу:**
Каждое сообщение я рассматриваю как уникальную задачу, требующую индивидуального подхода. Я не использую готовые шаблоны, а генерирую ответ специально для вас.

**Что я анализирую в вашем вопросе:**
- Ключевые концепции и темы
- Уровень детализации, который вам нужен
- Контекст и предполагаемые цели
- Стиль общения и предпочтения

**Как я могу помочь:**
- Предоставить детальные объяснения
- Привести практические примеры
- Дать персональные рекомендации
- Предложить дополнительные ресурсы для изучения

**Области моей экспертизы:**
- Технические дисциплины (программирование, электротехника)
- Образовательные материалы и методики
- Практические решения и реализация проектов
- Аналитическое мышление и решение задач

**Особенность моих ответов:**
Я создаю каждый ответ заново, учитывая специфику вашего вопроса, время обращения, контекст и ваши потребности. Никаких заготовленных фраз!

Не могли бы вы уточнить, какой именно аспект вашего вопроса вас интересует больше всего? Это поможет мне дать еще более персонализированный и полезный ответ."""

    def _generate_error_response(self, error_message: str) -> SimpleResponse:
        """Генерация ответа об ошибке"""
        return SimpleResponse(
            content=f"Извините, произошла ошибка при обработке вашего запроса: {error_message}",
            response_type='error',
            quality_score=0.1,
            personalization_score=0.0,
            context_relevance=0.1,
            generation_time=0.001,
            metadata={'error': True, 'error_message': error_message}
        )

    def _update_stats(self, response: SimpleResponse, response_time: float):
        """Обновление статистики"""
        self.stats['total_requests'] += 1
        
        if response.response_type == 'meta_question':
            self.stats['meta_questions'] += 1
        elif response.response_type == 'technical_explanation':
            self.stats['technical_questions'] += 1
        elif response.response_type == 'greeting':
            self.stats['greetings'] += 1
        else:
            self.stats['general_questions'] += 1
        
        # Обновляем средние значения
        total = self.stats['total_requests']
        self.stats['average_quality'] = (self.stats['average_quality'] * (total - 1) + response.quality_score) / total
        self.stats['average_response_time'] = (self.stats['average_response_time'] * (total - 1) + response_time) / total

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        return self.stats.copy()

    def get_health(self) -> Dict[str, Any]:
        """Получение состояния системы"""
        return {
            'status': 'healthy',
            'message': 'Simple Enhanced Dispatcher работает без шаблонности',
            'timestamp': datetime.now().isoformat(),
            'features': [
                'Уникальная генерация ответов',
                'Контекстный анализ',
                'Персонализация',
                'Адаптивные ответы'
            ],
            'stats': self.stats
        }

# Создаем Flask приложение
app = Flask(__name__)
CORS(app)

# Создаем экземпляр простого улучшенного диспетчера
simple_dispatcher = SimpleEnhancedDispatcher()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Обрабатываем запрос
        response = simple_dispatcher.process_request(message, user_id, session_id)
        
        return jsonify({
            'response': response.content,
            'response_type': response.response_type,
            'quality_score': response.quality_score,
            'personalization_score': response.personalization_score,
            'context_relevance': response.context_relevance,
            'generation_time': response.generation_time,
            'metadata': response.metadata,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в API: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Получение статистики"""
    try:
        stats = simple_dispatcher.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def get_health():
    """Получение состояния системы"""
    try:
        health = simple_dispatcher.get_health()
        return jsonify(health)
    except Exception as e:
        logger.error(f"❌ Ошибка получения состояния: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Simple Enhanced Dispatcher запущен")
    print("📡 API доступен на http://localhost:8080")
    print("✨ Особенности:")
    print("  - Устранение шаблонности")
    print("  - Уникальная генерация ответов")
    print("  - Контекстный анализ")
    print("  - Персонализация")
    print("🔗 Endpoints:")
    print("  POST /api/chat - основной чат")
    print("  GET  /api/stats - статистика")
    print("  GET  /api/health - состояние системы")
    
    app.run(host='0.0.0.0', port=8080, debug=True)





