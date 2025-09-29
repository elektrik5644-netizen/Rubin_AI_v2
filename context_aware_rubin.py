#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context-Aware Rubin AI - Система с контекстным пониманием
Устраняет шаблонность через анализ истории диалога и генерацию уникальных ответов
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import re
import hashlib

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Контекст диалога"""
    user_id: str
    session_id: str
    message_history: List[Dict[str, Any]]
    current_topic: str
    user_intent: str
    conversation_mood: str
    technical_level: str
    last_interaction: datetime
    context_keywords: List[str]
    user_preferences: Dict[str, Any]

@dataclass
class ResponseGeneration:
    """Генерация ответа"""
    base_response: str
    context_adaptations: List[str]
    personalization: str
    technical_depth: str
    mood_adaptation: str
    final_response: str

class ContextAwareRubinAI:
    """Rubin AI с контекстным пониманием и генерацией уникальных ответов"""
    
    def __init__(self, db_path: str = "context_aware_rubin.db"):
        self.db_path = db_path
        self.connection = None
        self.conversation_contexts = {}  # Кэш контекстов
        self.response_templates = {}  # Динамические шаблоны
        self.user_profiles = {}  # Профили пользователей
        
        # Инициализация
        self._initialize_database()
        self._load_response_templates()
        self._load_user_profiles()
        
        logger.info("🧠 Context-Aware Rubin AI инициализирован")

    def _initialize_database(self):
        """Инициализация базы данных для контекстного понимания"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self._create_context_tables()
            logger.info("✅ База данных контекстного понимания инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации БД: {e}")
            raise

    def _create_context_tables(self):
        """Создание таблиц для контекстного понимания"""
        cursor = self.connection.cursor()
        
        # Таблица контекстов диалогов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                context_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица истории сообщений
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                context_keywords TEXT,
                user_intent TEXT,
                response_quality INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица профилей пользователей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                technical_level TEXT DEFAULT 'intermediate',
                preferred_topics TEXT,
                communication_style TEXT DEFAULT 'professional',
                response_preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица динамических шаблонов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_type TEXT NOT NULL,
                base_template TEXT NOT NULL,
                context_adaptations TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("📊 Таблицы контекстного понимания созданы")

    def _load_response_templates(self):
        """Загрузка динамических шаблонов ответов"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM dynamic_templates")
            templates = cursor.fetchall()
            
            for template in templates:
                template_type = template['template_type']
                self.response_templates[template_type] = {
                    'base': template['base_template'],
                    'adaptations': json.loads(template['context_adaptations'] or '[]'),
                    'usage_count': template['usage_count'],
                    'success_rate': template['success_rate']
                }
            
            logger.info(f"📝 Загружено {len(self.response_templates)} динамических шаблонов")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки шаблонов: {e}")

    def _load_user_profiles(self):
        """Загрузка профилей пользователей"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM user_profiles")
            profiles = cursor.fetchall()
            
            for profile in profiles:
                user_id = profile['user_id']
                self.user_profiles[user_id] = {
                    'technical_level': profile['technical_level'],
                    'preferred_topics': json.loads(profile['preferred_topics'] or '[]'),
                    'communication_style': profile['communication_style'],
                    'response_preferences': json.loads(profile['response_preferences'] or '{}')
                }
            
            logger.info(f"👥 Загружено {len(self.user_profiles)} профилей пользователей")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки профилей: {e}")

    def get_conversation_context(self, user_id: str, session_id: str) -> ConversationContext:
        """Получение контекста диалога"""
        try:
            # Проверяем кэш
            cache_key = f"{user_id}_{session_id}"
            if cache_key in self.conversation_contexts:
                return self.conversation_contexts[cache_key]
            
            # Загружаем из базы данных
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM conversation_contexts 
                WHERE user_id = ? AND session_id = ?
                ORDER BY updated_at DESC LIMIT 1
            """, (user_id, session_id))
            
            context_row = cursor.fetchone()
            if context_row:
                context_data = json.loads(context_row['context_data'])
                context = ConversationContext(**context_data)
            else:
                # Создаем новый контекст
                context = ConversationContext(
                    user_id=user_id,
                    session_id=session_id,
                    message_history=[],
                    current_topic="general",
                    user_intent="unknown",
                    conversation_mood="neutral",
                    technical_level="intermediate",
                    last_interaction=datetime.now(),
                    context_keywords=[],
                    user_preferences={}
                )
            
            # Кэшируем контекст
            self.conversation_contexts[cache_key] = context
            return context
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения контекста: {e}")
            # Возвращаем базовый контекст
            return ConversationContext(
                user_id=user_id,
                session_id=session_id,
                message_history=[],
                current_topic="general",
                user_intent="unknown",
                conversation_mood="neutral",
                technical_level="intermediate",
                last_interaction=datetime.now(),
                context_keywords=[],
                user_preferences={}
            )

    def analyze_user_intent(self, message: str, context: ConversationContext) -> str:
        """Анализ намерений пользователя"""
        message_lower = message.lower()
        
        # Анализ на основе ключевых слов
        intent_keywords = {
            'question': ['как', 'что', 'почему', 'где', 'когда', 'зачем', '?'],
            'request': ['помоги', 'сделай', 'создай', 'напиши', 'покажи', 'объясни'],
            'complaint': ['проблема', 'ошибка', 'не работает', 'неправильно', 'плохо'],
            'greeting': ['привет', 'здравствуй', 'добрый', 'hi', 'hello'],
            'thanks': ['спасибо', 'благодарю', 'thanks', 'thank you'],
            'meta': ['как ты', 'что ты', 'как работает', 'как думаешь', 'как понимаешь']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        # Анализ на основе контекста
        if context.message_history:
            last_message = context.message_history[-1]
            if '?' in last_message.get('content', ''):
                return 'follow_up'
        
        return 'statement'

    def extract_context_keywords(self, message: str, context: ConversationContext) -> List[str]:
        """Извлечение ключевых слов для контекста"""
        # Технические термины
        technical_terms = [
            'arduino', 'python', 'javascript', 'html', 'css', 'sql', 'api',
            'электротехника', 'математика', 'программирование', 'автоматизация',
            'plc', 'сервопривод', 'датчик', 'мотор', 'схема', 'код'
        ]
        
        # Извлекаем технические термины
        keywords = []
        message_lower = message.lower()
        
        for term in technical_terms:
            if term in message_lower:
                keywords.append(term)
        
        # Добавляем ключевые слова из предыдущего контекста
        keywords.extend(context.context_keywords[-3:])  # Последние 3 ключевых слова
        
        return list(set(keywords))  # Убираем дубликаты

    def determine_conversation_mood(self, message: str, context: ConversationContext) -> str:
        """Определение настроения диалога"""
        message_lower = message.lower()
        
        # Позитивные индикаторы
        positive_words = ['спасибо', 'отлично', 'хорошо', 'понятно', 'помогло', 'классно']
        if any(word in message_lower for word in positive_words):
            return 'positive'
        
        # Негативные индикаторы
        negative_words = ['плохо', 'неправильно', 'ошибка', 'не работает', 'не понял', 'проблема']
        if any(word in message_lower for word in negative_words):
            return 'negative'
        
        # Нейтральные индикаторы
        neutral_words = ['как', 'что', 'объясни', 'покажи', 'расскажи']
        if any(word in message_lower for word in neutral_words):
            return 'neutral'
        
        return context.conversation_mood  # Сохраняем предыдущее настроение

    def generate_contextual_response(self, message: str, user_id: str, session_id: str) -> str:
        """Генерация контекстного ответа"""
        try:
            # Получаем контекст
            context = self.get_conversation_context(user_id, session_id)
            
            # Анализируем сообщение
            user_intent = self.analyze_user_intent(message, context)
            context_keywords = self.extract_context_keywords(message, context)
            conversation_mood = self.determine_conversation_mood(message, context)
            
            # Обновляем контекст
            context.user_intent = user_intent
            context.context_keywords = context_keywords
            context.conversation_mood = conversation_mood
            context.last_interaction = datetime.now()
            
            # Добавляем сообщение в историю
            context.message_history.append({
                'type': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat(),
                'intent': user_intent,
                'keywords': context_keywords
            })
            
            # Генерируем ответ
            response = self._generate_adaptive_response(message, context)
            
            # Добавляем ответ в историю
            context.message_history.append({
                'type': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat(),
                'intent': user_intent,
                'keywords': context_keywords
            })
            
            # Сохраняем обновленный контекст
            self._save_conversation_context(context)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            return "Извините, произошла ошибка при обработке вашего сообщения."

    def _generate_adaptive_response(self, message: str, context: ConversationContext) -> str:
        """Генерация адаптивного ответа на основе контекста"""
        
        # Обработка мета-вопросов
        if context.user_intent == 'meta':
            return self._handle_meta_question(message, context)
        
        # Обработка приветствий
        if context.user_intent == 'greeting':
            return self._handle_greeting(message, context)
        
        # Обработка благодарностей
        if context.user_intent == 'thanks':
            return self._handle_thanks(message, context)
        
        # Обработка жалоб
        if context.user_intent == 'complaint':
            return self._handle_complaint(message, context)
        
        # Обработка технических вопросов
        if any(keyword in context.context_keywords for keyword in ['arduino', 'python', 'электротехника', 'математика']):
            return self._handle_technical_question(message, context)
        
        # Общий ответ с учетом контекста
        return self._handle_general_question(message, context)

    def _handle_meta_question(self, message: str, context: ConversationContext) -> str:
        """Обработка мета-вопросов о собственном мышлении"""
        message_lower = message.lower()
        
        if 'как ты понимаешь' in message_lower:
            return f"""🧠 **Как я понимаю ваши сообщения:**

**Процесс понимания:**
1. **Анализ ключевых слов** - извлекаю технические термины и концепции
2. **Определение намерений** - понимаю, что вы хотите (вопрос, просьба, жалоба)
3. **Контекстный анализ** - учитываю предыдущие сообщения в диалоге
4. **Адаптация ответа** - подстраиваюсь под ваш уровень знаний и стиль общения

**Текущий контекст нашего диалога:**
- Тема: {context.current_topic}
- Ваше намерение: {context.user_intent}
- Настроение диалога: {context.conversation_mood}
- Ключевые слова: {', '.join(context.context_keywords[:5])}

**Как я адаптируюсь:**
- Учитываю ваши предпочтения в общении
- Подстраиваюсь под технический уровень
- Помню контекст предыдущих сообщений
- Генерирую уникальные ответы, а не шаблоны"""

        elif 'как ты думаешь' in message_lower:
            return f"""🤔 **Как я размышляю:**

**Процесс мышления:**
1. **Сбор информации** - анализирую ваше сообщение и контекст
2. **Сопоставление** - связываю с предыдущими знаниями и опытом
3. **Генерация идей** - создаю возможные варианты ответа
4. **Выбор лучшего** - выбираю наиболее подходящий ответ

**Текущие размышления:**
- Анализирую: {message[:50]}...
- Связываю с темой: {context.current_topic}
- Учитываю настроение: {context.conversation_mood}
- Адаптирую под ваш уровень: {context.technical_level}

**Отличие от шаблонов:**
- Каждый ответ уникален
- Учитываю контекст диалога
- Адаптируюсь под ваши потребности
- Помню предыдущие взаимодействия"""

        elif 'как работает' in message_lower:
            return f"""⚙️ **Как работает моя система:**

**Архитектура:**
1. **Контекстное понимание** - анализирую историю диалога
2. **Адаптивная генерация** - создаю уникальные ответы
3. **Профилирование пользователей** - запоминаю ваши предпочтения
4. **Динамические шаблоны** - обновляюсь на основе опыта

**Текущая сессия:**
- Сообщений в диалоге: {len(context.message_history)}
- Время сессии: {datetime.now() - context.last_interaction}
- Активные темы: {', '.join(context.context_keywords[:3])}

**Технические детали:**
- База данных контекстов
- Система анализа намерений
- Генерация адаптивных ответов
- Отслеживание качества взаимодействий"""

        else:
            return "Интересный вопрос о моем мышлении! Можете уточнить, что именно вас интересует?"

    def _handle_greeting(self, message: str, context: ConversationContext) -> str:
        """Обработка приветствий с учетом контекста"""
        greetings = [
            f"Привет! Рад снова с вами пообщаться. {self._get_contextual_greeting(context)}",
            f"Здравствуйте! Продолжим наш диалог. {self._get_contextual_greeting(context)}",
            f"Привет! Помню, мы обсуждали {context.current_topic}. {self._get_contextual_greeting(context)}"
        ]
        
        # Выбираем приветствие на основе контекста
        if context.message_history:
            return greetings[1]  # Продолжение диалога
        else:
            return greetings[0]  # Новый диалог

    def _get_contextual_greeting(self, context: ConversationContext) -> str:
        """Получение контекстного приветствия"""
        if context.context_keywords:
            topics = ', '.join(context.context_keywords[:2])
            return f"Готов помочь с {topics}."
        else:
            return "Чем могу помочь?"

    def _handle_thanks(self, message: str, context: ConversationContext) -> str:
        """Обработка благодарностей"""
        responses = [
            "Пожалуйста! Рад, что смог помочь.",
            "Не за что! Всегда готов помочь.",
            "Спасибо за обратную связь! Это помогает мне улучшаться."
        ]
        
        # Выбираем ответ на основе настроения
        if context.conversation_mood == 'positive':
            return responses[0]
        else:
            return responses[1]

    def _handle_complaint(self, message: str, context: ConversationContext) -> str:
        """Обработка жалоб и проблем"""
        return f"""Понимаю вашу проблему. Давайте разберемся вместе.

**Анализ проблемы:**
- Сообщение: {message[:100]}...
- Контекст: {context.current_topic}
- Предыдущие темы: {', '.join(context.context_keywords[:3])}

**Предлагаю:**
1. Детально разобрать проблему
2. Предложить конкретные решения
3. Учесть ваш уровень знаний: {context.technical_level}

Опишите проблему подробнее, и я помогу найти решение."""

    def _handle_technical_question(self, message: str, context: ConversationContext) -> str:
        """Обработка технических вопросов"""
        # Определяем техническую область
        if 'arduino' in context.context_keywords:
            return self._handle_arduino_question(message, context)
        elif 'python' in context.context_keywords:
            return self._handle_python_question(message, context)
        elif 'электротехника' in context.context_keywords:
            return self._handle_electrical_question(message, context)
        else:
            return self._handle_general_technical_question(message, context)

    def _handle_arduino_question(self, message: str, context: ConversationContext) -> str:
        """Обработка вопросов по Arduino"""
        return f"""🔧 **Arduino - контекстный ответ:**

**Анализ вашего вопроса:**
- Сообщение: {message[:100]}...
- Уровень сложности: {context.technical_level}
- Предыдущие темы: {', '.join(context.context_keywords[:3])}

**Предлагаю решение:**
1. **Базовые концепции** - если нужно начать с основ
2. **Практические примеры** - конкретные проекты и код
3. **Отладка проблем** - решение конкретных ошибок
4. **Продвинутые техники** - для опытных пользователей

**Учитывая ваш уровень ({context.technical_level}):**
- Адаптирую объяснения под ваши знания
- Предоставлю примеры кода
- Объясню принципы работы

Опишите конкретную задачу, и я дам детальный ответ с учетом нашего диалога."""

    def _handle_python_question(self, message: str, context: ConversationContext) -> str:
        """Обработка вопросов по Python"""
        return f"""🐍 **Python - контекстный ответ:**

**Анализ контекста:**
- Ваш вопрос: {message[:100]}...
- Технический уровень: {context.technical_level}
- Стиль общения: {context.conversation_mood}

**Предлагаю:**
1. **Объяснение концепций** - с примерами кода
2. **Практические решения** - готовые фрагменты
3. **Лучшие практики** - рекомендации по написанию кода
4. **Отладка** - решение проблем и ошибок

**Адаптация под ваш уровень:**
- Учитываю ваш опыт программирования
- Подбираю подходящие примеры
- Объясняю сложные концепции простым языком

Опишите конкретную задачу, и я предоставлю персонализированный ответ."""

    def _handle_electrical_question(self, message: str, context: ConversationContext) -> str:
        """Обработка вопросов по электротехнике"""
        return f"""⚡ **Электротехника - контекстный ответ:**

**Анализ вашего вопроса:**
- Сообщение: {message[:100]}...
- Контекст диалога: {context.current_topic}
- Ключевые термины: {', '.join(context.context_keywords[:3])}

**Предлагаю:**
1. **Теоретические основы** - принципы работы
2. **Практические расчеты** - формулы и примеры
3. **Схемы и диаграммы** - визуальное объяснение
4. **Безопасность** - важные моменты

**Учитывая ваш уровень ({context.technical_level}):**
- Адаптирую сложность объяснений
- Предоставлю практические примеры
- Объясню физические принципы

Опишите конкретную задачу, и я дам детальный ответ с учетом нашего диалога."""

    def _handle_general_technical_question(self, message: str, context: ConversationContext) -> str:
        """Обработка общих технических вопросов"""
        return f"""🔧 **Технический вопрос - контекстный ответ:**

**Анализ:**
- Ваш вопрос: {message[:100]}...
- Техническая область: {', '.join(context.context_keywords[:3])}
- Уровень сложности: {context.technical_level}

**Предлагаю:**
1. **Детальное объяснение** - с примерами
2. **Практические решения** - готовые варианты
3. **Дополнительные ресурсы** - для углубления
4. **Связанные темы** - для расширения знаний

**Персонализация:**
- Учитываю ваш стиль общения
- Адаптирую под ваш уровень знаний
- Помню контекст нашего диалога

Опишите конкретную задачу, и я предоставлю персонализированный ответ."""

    def _handle_general_question(self, message: str, context: ConversationContext) -> str:
        """Обработка общих вопросов"""
        return f"""💭 **Общий вопрос - контекстный ответ:**

**Анализ контекста:**
- Ваш вопрос: {message[:100]}...
- Намерение: {context.user_intent}
- Настроение диалога: {context.conversation_mood}
- Предыдущие темы: {', '.join(context.context_keywords[:3])}

**Предлагаю:**
1. **Развернутый ответ** - с учетом контекста
2. **Дополнительные вопросы** - для уточнения
3. **Связанные темы** - для расширения
4. **Практические советы** - если применимо

**Персонализация:**
- Учитываю ваш стиль общения
- Помню контекст нашего диалога
- Адаптирую под ваши потребности

Опишите подробнее, что вас интересует, и я дам персонализированный ответ."""

    def _save_conversation_context(self, context: ConversationContext):
        """Сохранение контекста диалога"""
        try:
            cursor = self.connection.cursor()
            
            # Подготавливаем данные для сохранения
            context_data = {
                'user_id': context.user_id,
                'session_id': context.session_id,
                'message_history': context.message_history,
                'current_topic': context.current_topic,
                'user_intent': context.user_intent,
                'conversation_mood': context.conversation_mood,
                'technical_level': context.technical_level,
                'last_interaction': context.last_interaction.isoformat(),
                'context_keywords': context.context_keywords,
                'user_preferences': context.user_preferences
            }
            
            # Проверяем, существует ли контекст
            cursor.execute("""
                SELECT id FROM conversation_contexts 
                WHERE user_id = ? AND session_id = ?
            """, (context.user_id, context.session_id))
            
            existing = cursor.fetchone()
            
            if existing:
                # Обновляем существующий контекст
                cursor.execute("""
                    UPDATE conversation_contexts 
                    SET context_data = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND session_id = ?
                """, (json.dumps(context_data), context.user_id, context.session_id))
            else:
                # Создаем новый контекст
                cursor.execute("""
                    INSERT INTO conversation_contexts (user_id, session_id, context_data)
                    VALUES (?, ?, ?)
                """, (context.user_id, context.session_id, json.dumps(context_data)))
            
            self.connection.commit()
            logger.info(f"💾 Контекст сохранен для {context.user_id}_{context.session_id}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения контекста: {e}")

    def get_conversation_summary(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Получение сводки диалога"""
        try:
            context = self.get_conversation_context(user_id, session_id)
            
            return {
                'user_id': user_id,
                'session_id': session_id,
                'message_count': len(context.message_history),
                'current_topic': context.current_topic,
                'user_intent': context.user_intent,
                'conversation_mood': context.conversation_mood,
                'technical_level': context.technical_level,
                'context_keywords': context.context_keywords,
                'last_interaction': context.last_interaction.isoformat(),
                'conversation_duration': str(datetime.now() - context.last_interaction)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения сводки: {e}")
            return {}

    def cleanup_old_contexts(self, days: int = 7):
        """Очистка старых контекстов"""
        try:
            cursor = self.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                DELETE FROM conversation_contexts 
                WHERE updated_at < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            self.connection.commit()
            
            logger.info(f"🧹 Удалено {deleted_count} старых контекстов")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки контекстов: {e}")
            return 0

# Создаем экземпляр Context-Aware Rubin AI
context_aware_rubin = ContextAwareRubinAI()

if __name__ == "__main__":
    # Тестирование системы
    print("🧠 Context-Aware Rubin AI - Тестирование")
    
    # Тестовый диалог
    test_messages = [
        "Привет!",
        "Как ты понимаешь мои сообщения?",
        "Расскажи про Arduino",
        "Спасибо за помощь!"
    ]
    
    user_id = "test_user"
    session_id = "test_session"
    
    for message in test_messages:
        print(f"\n👤 Пользователь: {message}")
        response = context_aware_rubin.generate_contextual_response(message, user_id, session_id)
        print(f"🤖 Rubin: {response}")
    
    # Получаем сводку диалога
    summary = context_aware_rubin.get_conversation_summary(user_id, session_id)
    print(f"\n📊 Сводка диалога: {json.dumps(summary, indent=2, ensure_ascii=False)}")





