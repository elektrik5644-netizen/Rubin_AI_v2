#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Response System - Динамическая система ответов с адаптацией под контекст
Устраняет шаблонность через динамическую генерацию и адаптацию ответов
"""

import json
import logging
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Типы ответов"""
    GREETING = "greeting"
    TECHNICAL_EXPLANATION = "technical_explanation"
    PROBLEM_SOLVING = "problem_solving"
    META_QUESTION = "meta_question"
    THANKS = "thanks"
    COMPLAINT = "complaint"
    GENERAL_QUESTION = "general_question"
    ERROR_HANDLING = "error_handling"

class AdaptationLevel(Enum):
    """Уровни адаптации"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class DynamicContext:
    """Динамический контекст для генерации ответов"""
    user_id: str
    session_id: str
    message: str
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    technical_level: str = "intermediate"
    communication_style: str = "professional"
    conversation_mood: str = "neutral"
    current_topic: str = "general"
    keywords: List[str] = field(default_factory=list)
    user_intent: str = "unknown"
    response_preferences: Dict[str, Any] = field(default_factory=dict)
    context_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResponseTemplate:
    """Динамический шаблон ответа"""
    template_id: str
    base_structure: str
    variables: List[str]
    adaptation_rules: Dict[str, Any]
    quality_metrics: Dict[str, float]
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None

@dataclass
class GeneratedResponse:
    """Сгенерированный ответ"""
    content: str
    response_type: ResponseType
    adaptation_level: AdaptationLevel
    quality_score: float
    personalization_score: float
    context_relevance: float
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicResponseSystem:
    """Динамическая система ответов с адаптацией под контекст"""
    
    def __init__(self):
        self.response_templates = {}
        self.adaptation_rules = {}
        self.personalization_engine = {}
        self.context_analyzer = {}
        self.quality_assessor = {}
        self.learning_system = {}
        
        # Статистика и метрики
        self.usage_stats = defaultdict(int)
        self.quality_stats = defaultdict(list)
        self.adaptation_stats = defaultdict(list)
        
        # Инициализация
        self._initialize_templates()
        self._initialize_adaptation_rules()
        self._initialize_personalization()
        self._initialize_context_analyzer()
        self._initialize_quality_assessor()
        self._initialize_learning_system()
        
        logger.info("🔄 Dynamic Response System инициализирован")

    def _initialize_templates(self):
        """Инициализация динамических шаблонов"""
        self.response_templates = {
            ResponseType.GREETING: {
                'basic': ResponseTemplate(
                    template_id="greeting_basic",
                    base_structure="{greeting} {context_reference} {offer_help}",
                    variables=['greeting', 'context_reference', 'offer_help'],
                    adaptation_rules={
                        'time_based': True,
                        'mood_based': True,
                        'history_based': True
                    },
                    quality_metrics={'relevance': 0.8, 'personalization': 0.7, 'clarity': 0.9}
                ),
                'advanced': ResponseTemplate(
                    template_id="greeting_advanced",
                    base_structure="{greeting} {personal_reference} {context_analysis} {offer_help} {next_steps}",
                    variables=['greeting', 'personal_reference', 'context_analysis', 'offer_help', 'next_steps'],
                    adaptation_rules={
                        'time_based': True,
                        'mood_based': True,
                        'history_based': True,
                        'profile_based': True,
                        'topic_based': True
                    },
                    quality_metrics={'relevance': 0.9, 'personalization': 0.9, 'clarity': 0.8}
                )
            },
            
            ResponseType.TECHNICAL_EXPLANATION: {
                'basic': ResponseTemplate(
                    template_id="tech_explanation_basic",
                    base_structure="{introduction} {main_explanation} {example} {conclusion}",
                    variables=['introduction', 'main_explanation', 'example', 'conclusion'],
                    adaptation_rules={
                        'level_based': True,
                        'topic_based': True,
                        'complexity_based': True
                    },
                    quality_metrics={'accuracy': 0.9, 'clarity': 0.8, 'completeness': 0.7}
                ),
                'advanced': ResponseTemplate(
                    template_id="tech_explanation_advanced",
                    base_structure="{introduction} {concept_overview} {detailed_explanation} {practical_examples} {best_practices} {troubleshooting} {conclusion}",
                    variables=['introduction', 'concept_overview', 'detailed_explanation', 'practical_examples', 'best_practices', 'troubleshooting', 'conclusion'],
                    adaptation_rules={
                        'level_based': True,
                        'topic_based': True,
                        'complexity_based': True,
                        'experience_based': True,
                        'context_based': True
                    },
                    quality_metrics={'accuracy': 0.95, 'clarity': 0.9, 'completeness': 0.9}
                )
            },
            
            ResponseType.PROBLEM_SOLVING: {
                'basic': ResponseTemplate(
                    template_id="problem_solving_basic",
                    base_structure="{problem_acknowledgment} {analysis} {solution} {verification}",
                    variables=['problem_acknowledgment', 'analysis', 'solution', 'verification'],
                    adaptation_rules={
                        'urgency_based': True,
                        'complexity_based': True,
                        'domain_based': True
                    },
                    quality_metrics={'helpfulness': 0.8, 'accuracy': 0.8, 'completeness': 0.7}
                ),
                'advanced': ResponseTemplate(
                    template_id="problem_solving_advanced",
                    base_structure="{problem_acknowledgment} {root_cause_analysis} {solution_options} {recommended_solution} {implementation_guide} {prevention_tips} {verification}",
                    variables=['problem_acknowledgment', 'root_cause_analysis', 'solution_options', 'recommended_solution', 'implementation_guide', 'prevention_tips', 'verification'],
                    adaptation_rules={
                        'urgency_based': True,
                        'complexity_based': True,
                        'domain_based': True,
                        'experience_based': True,
                        'context_based': True
                    },
                    quality_metrics={'helpfulness': 0.9, 'accuracy': 0.9, 'completeness': 0.9}
                )
            },
            
            ResponseType.META_QUESTION: {
                'basic': ResponseTemplate(
                    template_id="meta_question_basic",
                    base_structure="{acknowledgment} {simple_explanation} {example} {conclusion}",
                    variables=['acknowledgment', 'simple_explanation', 'example', 'conclusion'],
                    adaptation_rules={
                        'question_type_based': True,
                        'level_based': True
                    },
                    quality_metrics={'transparency': 0.8, 'clarity': 0.8, 'helpfulness': 0.7}
                ),
                'advanced': ResponseTemplate(
                    template_id="meta_question_advanced",
                    base_structure="{acknowledgment} {detailed_explanation} {technical_details} {examples} {limitations} {conclusion}",
                    variables=['acknowledgment', 'detailed_explanation', 'technical_details', 'examples', 'limitations', 'conclusion'],
                    adaptation_rules={
                        'question_type_based': True,
                        'level_based': True,
                        'context_based': True,
                        'depth_based': True
                    },
                    quality_metrics={'transparency': 0.9, 'clarity': 0.9, 'helpfulness': 0.9}
                )
            }
        }
        
        logger.info(f"📝 Инициализировано {len(self.response_templates)} типов шаблонов")

    def _initialize_adaptation_rules(self):
        """Инициализация правил адаптации"""
        self.adaptation_rules = {
            'time_based': {
                'morning': ['Доброе утро!', 'Хорошего дня!', 'Удачного начала дня!'],
                'afternoon': ['Добрый день!', 'Хорошего дня!', 'Продуктивного дня!'],
                'evening': ['Добрый вечер!', 'Хорошего вечера!', 'Приятного вечера!'],
                'night': ['Доброй ночи!', 'Спокойной ночи!', 'Хорошего отдыха!']
            },
            
            'mood_based': {
                'positive': {
                    'tone': 'энтузиазм',
                    'encouragement': 'Отлично! Продолжаем!',
                    'acknowledgment': 'Понимаю ваш интерес'
                },
                'negative': {
                    'tone': 'поддержка',
                    'encouragement': 'Давайте разберемся вместе',
                    'acknowledgment': 'Понимаю вашу проблему'
                },
                'neutral': {
                    'tone': 'профессионализм',
                    'encouragement': 'Готов помочь',
                    'acknowledgment': 'Интересный вопрос'
                }
            },
            
            'level_based': {
                'beginner': {
                    'complexity': 'simple',
                    'terminology': 'basic',
                    'examples': 'basic',
                    'explanation_style': 'step_by_step'
                },
                'intermediate': {
                    'complexity': 'moderate',
                    'terminology': 'standard',
                    'examples': 'practical',
                    'explanation_style': 'balanced'
                },
                'advanced': {
                    'complexity': 'complex',
                    'terminology': 'technical',
                    'examples': 'advanced',
                    'explanation_style': 'comprehensive'
                },
                'expert': {
                    'complexity': 'expert',
                    'terminology': 'specialized',
                    'examples': 'cutting_edge',
                    'explanation_style': 'detailed'
                }
            },
            
            'topic_based': {
                'arduino': {
                    'domain_knowledge': 'микроконтроллеры и электроника',
                    'examples': 'практические проекты',
                    'terminology': 'техническая',
                    'focus': 'практическое применение'
                },
                'python': {
                    'domain_knowledge': 'программирование и разработка',
                    'examples': 'код и алгоритмы',
                    'terminology': 'программистская',
                    'focus': 'эффективность и лучшие практики'
                },
                'mathematics': {
                    'domain_knowledge': 'математические концепции',
                    'examples': 'формулы и расчеты',
                    'terminology': 'математическая',
                    'focus': 'точность и понимание принципов'
                },
                'electronics': {
                    'domain_knowledge': 'электротехника и схемотехника',
                    'examples': 'схемы и расчеты',
                    'terminology': 'техническая',
                    'focus': 'практическое применение и безопасность'
                }
            },
            
            'urgency_based': {
                'low': {
                    'response_style': 'detailed',
                    'time_allocation': 'comprehensive',
                    'priority': 'quality'
                },
                'medium': {
                    'response_style': 'balanced',
                    'time_allocation': 'moderate',
                    'priority': 'efficiency'
                },
                'high': {
                    'response_style': 'concise',
                    'time_allocation': 'quick',
                    'priority': 'speed'
                },
                'critical': {
                    'response_style': 'immediate',
                    'time_allocation': 'minimal',
                    'priority': 'urgency'
                }
            }
        }
        
        logger.info(f"🔄 Инициализировано {len(self.adaptation_rules)} правил адаптации")

    def _initialize_personalization(self):
        """Инициализация системы персонализации"""
        self.personalization_engine = {
            'communication_styles': {
                'formal': {
                    'greeting': 'Здравствуйте',
                    'transition': 'Перейдем к',
                    'conclusion': 'Надеюсь, это поможет',
                    'tone': 'профессиональный',
                    'formality': 'высокая'
                },
                'casual': {
                    'greeting': 'Привет',
                    'transition': 'Теперь давай',
                    'conclusion': 'Удачи!',
                    'tone': 'дружелюбный',
                    'formality': 'низкая'
                },
                'professional': {
                    'greeting': 'Добро пожаловать',
                    'transition': 'Рассмотрим',
                    'conclusion': 'Готов помочь дальше',
                    'tone': 'деловой',
                    'formality': 'средняя'
                }
            },
            
            'response_preferences': {
                'length': {
                    'concise': 'краткие ответы',
                    'detailed': 'подробные объяснения',
                    'comprehensive': 'исчерпывающие ответы'
                },
                'style': {
                    'direct': 'прямые ответы',
                    'explanatory': 'объяснительные ответы',
                    'interactive': 'интерактивные ответы'
                },
                'focus': {
                    'practical': 'практические советы',
                    'theoretical': 'теоретические объяснения',
                    'balanced': 'сбалансированный подход'
                }
            },
            
            'learning_patterns': {
                'visual': 'схемы и диаграммы',
                'auditory': 'пошаговые объяснения',
                'kinesthetic': 'практические примеры',
                'reading': 'подробные тексты'
            }
        }
        
        logger.info(f"👤 Инициализирована система персонализации")

    def _initialize_context_analyzer(self):
        """Инициализация анализатора контекста"""
        self.context_analyzer = {
            'intent_classifiers': {
                'question': ['как', 'что', 'почему', 'где', 'когда', 'зачем', '?'],
                'request': ['помоги', 'сделай', 'создай', 'напиши', 'покажи', 'объясни'],
                'complaint': ['проблема', 'ошибка', 'не работает', 'неправильно', 'плохо'],
                'greeting': ['привет', 'здравствуй', 'добрый', 'hi', 'hello'],
                'thanks': ['спасибо', 'благодарю', 'thanks', 'thank you'],
                'meta': ['как ты', 'что ты', 'как работает', 'как думаешь']
            },
            
            'topic_classifiers': {
                'arduino': ['arduino', 'микроконтроллер', 'пин', 'pin', 'digitalwrite', 'analogread'],
                'python': ['python', 'питон', 'код', 'функция', 'класс', 'модуль'],
                'electronics': ['электротехника', 'схема', 'резистор', 'конденсатор', 'транзистор'],
                'mathematics': ['математика', 'формула', 'расчет', 'уравнение', 'функция'],
                'programming': ['программирование', 'алгоритм', 'структура', 'данные', 'логика']
            },
            
            'complexity_indicators': {
                'beginner': ['простой', 'базовый', 'начальный', 'легкий', 'основы'],
                'intermediate': ['средний', 'умеренный', 'стандартный', 'обычный'],
                'advanced': ['сложный', 'продвинутый', 'экспертный', 'профессиональный'],
                'expert': ['экспертный', 'профессиональный', 'высокий', 'специализированный']
            },
            
            'urgency_indicators': {
                'low': ['не спеша', 'когда будет время', 'не срочно'],
                'medium': ['нужно', 'требуется', 'желательно'],
                'high': ['срочно', 'быстро', 'немедленно', 'критично'],
                'critical': ['авария', 'критично', 'сейчас', 'немедленно']
            }
        }
        
        logger.info(f"🔍 Инициализирован анализатор контекста")

    def _initialize_quality_assessor(self):
        """Инициализация оценщика качества"""
        self.quality_assessor = {
            'relevance_metrics': {
                'keyword_match': 0.3,
                'topic_alignment': 0.2,
                'intent_satisfaction': 0.2,
                'context_consistency': 0.2,
                'user_satisfaction': 0.1
            },
            
            'clarity_metrics': {
                'readability': 0.3,
                'structure': 0.2,
                'terminology': 0.2,
                'examples': 0.2,
                'conclusion': 0.1
            },
            
            'completeness_metrics': {
                'coverage': 0.4,
                'depth': 0.3,
                'examples': 0.2,
                'references': 0.1
            },
            
            'personalization_metrics': {
                'style_match': 0.3,
                'level_adaptation': 0.3,
                'preference_alignment': 0.2,
                'context_awareness': 0.2
            }
        }
        
        logger.info(f"📊 Инициализирован оценщик качества")

    def _initialize_learning_system(self):
        """Инициализация системы обучения"""
        self.learning_system = {
            'feedback_mechanisms': {
                'implicit': ['response_time', 'follow_up_questions', 'conversation_continuation'],
                'explicit': ['user_ratings', 'corrections', 'preferences']
            },
            
            'adaptation_strategies': {
                'template_optimization': 'улучшение шаблонов',
                'rule_refinement': 'уточнение правил',
                'personalization_enhancement': 'улучшение персонализации',
                'context_understanding': 'улучшение понимания контекста'
            },
            
            'learning_algorithms': {
                'reinforcement_learning': 'обучение с подкреплением',
                'supervised_learning': 'обучение с учителем',
                'unsupervised_learning': 'обучение без учителя',
                'transfer_learning': 'трансферное обучение'
            }
        }
        
        logger.info(f"🧠 Инициализирована система обучения")

    def generate_dynamic_response(self, context: DynamicContext) -> GeneratedResponse:
        """Генерация динамического ответа"""
        start_time = datetime.now()
        
        try:
            # Анализируем контекст
            analyzed_context = self._analyze_context(context)
            
            # Определяем тип ответа
            response_type = self._determine_response_type(analyzed_context)
            
            # Выбираем уровень адаптации
            adaptation_level = self._determine_adaptation_level(analyzed_context)
            
            # Получаем шаблон
            template = self._get_template(response_type, adaptation_level)
            
            # Генерируем переменные
            variables = self._generate_variables(template, analyzed_context)
            
            # Применяем шаблон
            base_response = self._apply_template(template, variables)
            
            # Адаптируем под контекст
            adapted_response = self._adapt_response(base_response, analyzed_context)
            
            # Персонализируем
            personalized_response = self._personalize_response(adapted_response, analyzed_context)
            
            # Оцениваем качество
            quality_score = self._assess_quality(personalized_response, analyzed_context)
            
            # Создаем финальный ответ
            generation_time = (datetime.now() - start_time).total_seconds()
            
            response = GeneratedResponse(
                content=personalized_response,
                response_type=response_type,
                adaptation_level=adaptation_level,
                quality_score=quality_score,
                personalization_score=self._calculate_personalization_score(personalized_response, analyzed_context),
                context_relevance=self._calculate_context_relevance(personalized_response, analyzed_context),
                generation_time=generation_time,
                metadata={
                    'template_id': template.template_id,
                    'variables_used': list(variables.keys()),
                    'adaptations_applied': self._get_applied_adaptations(analyzed_context),
                    'quality_breakdown': self._get_quality_breakdown(personalized_response, analyzed_context)
                }
            )
            
            # Обновляем статистику
            self._update_usage_stats(response)
            
            logger.info(f"✅ Сгенерирован динамический ответ (качество: {quality_score:.2f}, время: {generation_time:.3f}с)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации динамического ответа: {e}")
            return self._generate_fallback_response(context)

    def _analyze_context(self, context: DynamicContext) -> DynamicContext:
        """Анализ контекста"""
        # Анализируем намерения
        context.user_intent = self._classify_intent(context.message)
        
        # Анализируем темы
        context.keywords = self._extract_keywords(context.message)
        context.current_topic = self._classify_topic(context.keywords)
        
        # Анализируем сложность
        context.technical_level = self._assess_complexity(context.message, context.keywords)
        
        # Анализируем срочность
        context.context_metadata['urgency'] = self._assess_urgency(context.message)
        
        # Анализируем настроение
        context.conversation_mood = self._assess_mood(context.message, context.message_history)
        
        return context

    def _classify_intent(self, message: str) -> str:
        """Классификация намерений"""
        message_lower = message.lower()
        
        for intent, keywords in self.context_analyzer['intent_classifiers'].items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return 'unknown'

    def _extract_keywords(self, message: str) -> List[str]:
        """Извлечение ключевых слов"""
        keywords = []
        message_lower = message.lower()
        
        for topic, topic_keywords in self.context_analyzer['topic_classifiers'].items():
            for keyword in topic_keywords:
                if keyword in message_lower:
                    keywords.append(topic)
                    break
        
        return list(set(keywords))

    def _classify_topic(self, keywords: List[str]) -> str:
        """Классификация темы"""
        if not keywords:
            return 'general'
        
        # Возвращаем первую найденную тему
        return keywords[0]

    def _assess_complexity(self, message: str, keywords: List[str]) -> str:
        """Оценка сложности"""
        message_lower = message.lower()
        
        for level, indicators in self.context_analyzer['complexity_indicators'].items():
            if any(indicator in message_lower for indicator in indicators):
                return level
        
        # Определяем по ключевым словам
        if any(keyword in ['arduino', 'python', 'electronics'] for keyword in keywords):
            return 'intermediate'
        
        return 'beginner'

    def _assess_urgency(self, message: str) -> str:
        """Оценка срочности"""
        message_lower = message.lower()
        
        for urgency, indicators in self.context_analyzer['urgency_indicators'].items():
            if any(indicator in message_lower for indicator in indicators):
                return urgency
        
        return 'medium'

    def _assess_mood(self, message: str, history: List[Dict[str, Any]]) -> str:
        """Оценка настроения"""
        message_lower = message.lower()
        
        # Позитивные индикаторы
        positive_words = ['спасибо', 'отлично', 'хорошо', 'понятно', 'помогло', 'классно']
        if any(word in message_lower for word in positive_words):
            return 'positive'
        
        # Негативные индикаторы
        negative_words = ['плохо', 'неправильно', 'ошибка', 'не работает', 'не понял', 'проблема']
        if any(word in message_lower for word in negative_words):
            return 'negative'
        
        return 'neutral'

    def _determine_response_type(self, context: DynamicContext) -> ResponseType:
        """Определение типа ответа"""
        if context.user_intent == 'greeting':
            return ResponseType.GREETING
        elif context.user_intent == 'thanks':
            return ResponseType.THANKS
        elif context.user_intent == 'complaint':
            return ResponseType.COMPLAINT
        elif context.user_intent == 'meta':
            return ResponseType.META_QUESTION
        elif context.user_intent == 'question' and context.keywords:
            return ResponseType.TECHNICAL_EXPLANATION
        elif context.user_intent == 'request':
            return ResponseType.PROBLEM_SOLVING
        else:
            return ResponseType.GENERAL_QUESTION

    def _determine_adaptation_level(self, context: DynamicContext) -> AdaptationLevel:
        """Определение уровня адаптации"""
        if context.technical_level == 'expert':
            return AdaptationLevel.EXPERT
        elif context.technical_level == 'advanced':
            return AdaptationLevel.ADVANCED
        elif context.technical_level == 'intermediate':
            return AdaptationLevel.INTERMEDIATE
        else:
            return AdaptationLevel.BASIC

    def _get_template(self, response_type: ResponseType, adaptation_level: AdaptationLevel) -> ResponseTemplate:
        """Получение шаблона"""
        level_key = adaptation_level.value
        
        if response_type in self.response_templates and level_key in self.response_templates[response_type]:
            return self.response_templates[response_type][level_key]
        
        # Fallback к базовому уровню
        if 'basic' in self.response_templates[response_type]:
            return self.response_templates[response_type]['basic']
        
        # Fallback к общему шаблону
        return self.response_templates[ResponseType.GENERAL_QUESTION]['basic']

    def _generate_variables(self, template: ResponseTemplate, context: DynamicContext) -> Dict[str, str]:
        """Генерация переменных для шаблона"""
        variables = {}
        
        for variable in template.variables:
            if variable == 'greeting':
                variables[variable] = self._generate_greeting(context)
            elif variable == 'context_reference':
                variables[variable] = self._generate_context_reference(context)
            elif variable == 'offer_help':
                variables[variable] = self._generate_offer_help(context)
            elif variable == 'introduction':
                variables[variable] = self._generate_introduction(context)
            elif variable == 'main_explanation':
                variables[variable] = self._generate_main_explanation(context)
            elif variable == 'example':
                variables[variable] = self._generate_example(context)
            elif variable == 'conclusion':
                variables[variable] = self._generate_conclusion(context)
            else:
                variables[variable] = self._generate_generic_variable(variable, context)
        
        return variables

    def _generate_greeting(self, context: DynamicContext) -> str:
        """Генерация приветствия"""
        style = context.communication_style
        time_of_day = self._get_time_of_day()
        
        if style == 'formal':
            greetings = ['Здравствуйте', 'Добро пожаловать', 'Приветствую']
        elif style == 'casual':
            greetings = ['Привет', 'Хай', 'Доброго времени суток']
        else:
            greetings = ['Добро пожаловать', 'Рад помочь', 'Готов к работе']
        
        return random.choice(greetings)

    def _get_time_of_day(self) -> str:
        """Получение времени дня"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def _generate_context_reference(self, context: DynamicContext) -> str:
        """Генерация ссылки на контекст"""
        if len(context.message_history) == 0:
            return "Чем могу помочь?"
        elif context.keywords:
            topic = context.keywords[0]
            return f"Продолжим работу с {topic}."
        else:
            return "Продолжим наш диалог."

    def _generate_offer_help(self, context: DynamicContext) -> str:
        """Генерация предложения помощи"""
        if context.keywords:
            return "Какая конкретная задача стоит?"
        elif context.user_intent == 'question':
            return "Опишите подробнее, что вас интересует."
        else:
            return "Чем могу помочь?"

    def _generate_introduction(self, context: DynamicContext) -> str:
        """Генерация введения"""
        level = context.technical_level
        
        if level == 'beginner':
            return "Начнем с основ:"
        elif level == 'advanced':
            return "Перейдем к сложным аспектам:"
        else:
            return "Рассмотрим подробнее:"

    def _generate_main_explanation(self, context: DynamicContext) -> str:
        """Генерация основного объяснения"""
        if context.current_topic == 'arduino':
            return "Arduino - это платформа для создания интерактивных проектов с микроконтроллерами."
        elif context.current_topic == 'python':
            return "Python - мощный язык программирования для различных задач."
        elif context.current_topic == 'electronics':
            return "Электротехника изучает принципы работы электрических цепей и устройств."
        else:
            return "Рассмотрим основные принципы:"

    def _generate_example(self, context: DynamicContext) -> str:
        """Генерация примера"""
        if context.current_topic == 'arduino':
            return "Например, для управления светодиодом: digitalWrite(LED_PIN, HIGH);"
        elif context.current_topic == 'python':
            return "Например, создание списка: my_list = [1, 2, 3, 4, 5]"
        elif context.current_topic == 'electronics':
            return "Например, закон Ома: U = I × R"
        else:
            return "Рассмотрим практический пример:"

    def _generate_conclusion(self, context: DynamicContext) -> str:
        """Генерация заключения"""
        conclusions = [
            "Надеюсь, это поможет в решении вашей задачи.",
            "Если нужны дополнительные пояснения, спрашивайте.",
            "Готов помочь с дальнейшими вопросами.",
            "Удачи в реализации!"
        ]
        return random.choice(conclusions)

    def _generate_generic_variable(self, variable: str, context: DynamicContext) -> str:
        """Генерация общей переменной"""
        return f"[{variable}]"

    def _apply_template(self, template: ResponseTemplate, variables: Dict[str, str]) -> str:
        """Применение шаблона"""
        response = template.base_structure
        
        for variable, value in variables.items():
            placeholder = f"{{{variable}}}"
            response = response.replace(placeholder, value)
        
        return response

    def _adapt_response(self, response: str, context: DynamicContext) -> str:
        """Адаптация ответа под контекст"""
        # Адаптация под настроение
        if context.conversation_mood == 'positive':
            response = f"😊 {response}"
        elif context.conversation_mood == 'negative':
            response = f"🤝 {response}"
        
        # Адаптация под срочность
        urgency = context.context_metadata.get('urgency', 'medium')
        if urgency == 'high':
            response = f"⚡ {response}"
        elif urgency == 'critical':
            response = f"🚨 {response}"
        
        return response

    def _personalize_response(self, response: str, context: DynamicContext) -> str:
        """Персонализация ответа"""
        style = context.communication_style
        
        if style == 'formal':
            response = response.replace('Привет', 'Здравствуйте')
            response = response.replace('давай', 'давайте')
        elif style == 'casual':
            response = response.replace('Здравствуйте', 'Привет')
            response = response.replace('давайте', 'давай')
        
        return response

    def _assess_quality(self, response: str, context: DynamicContext) -> float:
        """Оценка качества ответа"""
        quality_score = 0.0
        
        # Проверка релевантности
        if any(keyword in response.lower() for keyword in context.keywords):
            quality_score += 0.3
        
        # Проверка длины
        if 50 <= len(response) <= 500:
            quality_score += 0.2
        
        # Проверка структуры
        if any(marker in response for marker in ['**', '1.', '2.', '3.']):
            quality_score += 0.2
        
        # Проверка персонализации
        if context.communication_style in response:
            quality_score += 0.2
        
        # Проверка контекстной адаптации
        if context.conversation_mood in response or context.technical_level in response:
            quality_score += 0.1
        
        return min(quality_score, 1.0)

    def _calculate_personalization_score(self, response: str, context: DynamicContext) -> float:
        """Расчет оценки персонализации"""
        score = 0.0
        
        # Проверка стиля общения
        if context.communication_style in response:
            score += 0.4
        
        # Проверка уровня сложности
        if context.technical_level in response:
            score += 0.3
        
        # Проверка контекстных ссылок
        if context.current_topic in response:
            score += 0.3
        
        return min(score, 1.0)

    def _calculate_context_relevance(self, response: str, context: DynamicContext) -> float:
        """Расчет релевантности контексту"""
        score = 0.0
        
        # Проверка ключевых слов
        if any(keyword in response.lower() for keyword in context.keywords):
            score += 0.4
        
        # Проверка намерения
        if context.user_intent in response.lower():
            score += 0.3
        
        # Проверка темы
        if context.current_topic in response.lower():
            score += 0.3
        
        return min(score, 1.0)

    def _get_applied_adaptations(self, context: DynamicContext) -> List[str]:
        """Получение примененных адаптаций"""
        adaptations = []
        
        if context.conversation_mood != 'neutral':
            adaptations.append('mood_adaptation')
        
        if context.technical_level != 'intermediate':
            adaptations.append('level_adaptation')
        
        if context.communication_style != 'professional':
            adaptations.append('style_adaptation')
        
        return adaptations

    def _get_quality_breakdown(self, response: str, context: DynamicContext) -> Dict[str, float]:
        """Получение разбивки качества"""
        return {
            'relevance': self._calculate_context_relevance(response, context),
            'personalization': self._calculate_personalization_score(response, context),
            'clarity': 0.8,  # Заглушка
            'completeness': 0.7,  # Заглушка
            'helpfulness': 0.8  # Заглушка
        }

    def _update_usage_stats(self, response: GeneratedResponse):
        """Обновление статистики использования"""
        self.usage_stats[response.response_type.value] += 1
        self.quality_stats[response.response_type.value].append(response.quality_score)
        self.adaptation_stats[response.adaptation_level.value].append(response.quality_score)

    def _generate_fallback_response(self, context: DynamicContext) -> GeneratedResponse:
        """Генерация резервного ответа"""
        fallback_content = f"""Извините, произошла ошибка при обработке вашего сообщения: "{context.message}".

Попробуйте переформулировать вопрос или обратитесь к администратору системы."""
        
        return GeneratedResponse(
            content=fallback_content,
            response_type=ResponseType.ERROR_HANDLING,
            adaptation_level=AdaptationLevel.BASIC,
            quality_score=0.5,
            personalization_score=0.0,
            context_relevance=0.3,
            generation_time=0.001,
            metadata={'error': True, 'fallback': True}
        )

    def get_system_stats(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        return {
            'templates_count': sum(len(templates) for templates in self.response_templates.values()),
            'adaptation_rules_count': len(self.adaptation_rules),
            'usage_stats': dict(self.usage_stats),
            'average_quality_by_type': {
                response_type: sum(scores) / len(scores) if scores else 0.0
                for response_type, scores in self.quality_stats.items()
            },
            'average_quality_by_level': {
                level: sum(scores) / len(scores) if scores else 0.0
                for level, scores in self.adaptation_stats.items()
            }
        }

# Создаем экземпляр Dynamic Response System
dynamic_response_system = DynamicResponseSystem()

if __name__ == "__main__":
    # Тестирование динамической системы ответов
    print("🔄 Dynamic Response System - Тестирование")
    
    # Тестовые контексты
    test_contexts = [
        DynamicContext(
            user_id="test_user",
            session_id="test_session",
            message="Привет! Как дела?",
            user_profile={'communication_style': 'casual'},
            technical_level='beginner'
        ),
        DynamicContext(
            user_id="test_user",
            session_id="test_session",
            message="Как работает Arduino?",
            user_profile={'communication_style': 'professional'},
            technical_level='intermediate',
            keywords=['arduino']
        ),
        DynamicContext(
            user_id="test_user",
            session_id="test_session",
            message="Как ты понимаешь мои сообщения?",
            user_profile={'communication_style': 'formal'},
            technical_level='advanced'
        )
    ]
    
    # Тестируем каждый контекст
    for i, context in enumerate(test_contexts, 1):
        print(f"\n--- Тест {i} ---")
        print(f"Сообщение: {context.message}")
        
        response = dynamic_response_system.generate_dynamic_response(context)
        
        print(f"Тип ответа: {response.response_type.value}")
        print(f"Уровень адаптации: {response.adaptation_level.value}")
        print(f"Качество: {response.quality_score:.2f}")
        print(f"Персонализация: {response.personalization_score:.2f}")
        print(f"Релевантность: {response.context_relevance:.2f}")
        print(f"Время генерации: {response.generation_time:.3f}с")
        print(f"Ответ: {response.content[:200]}...")
    
    # Получаем статистику
    stats = dynamic_response_system.get_system_stats()
    print(f"\n📊 Статистика системы: {json.dumps(stats, indent=2, ensure_ascii=False)}")





