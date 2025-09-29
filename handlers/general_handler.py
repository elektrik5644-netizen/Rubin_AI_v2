#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Handler - Обработчик общих запросов
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GeneralHandler:
    """Обработчик общих запросов"""
    
    def __init__(self):
        self.greetings = [
            "Привет! Я Rubin AI, ваш интеллектуальный помощник.",
            "Здравствуйте! Чем могу помочь?",
            "Добро пожаловать! Я готов ответить на ваши вопросы.",
            "Привет! Как дела? Чем займемся сегодня?"
        ]
        
        self.help_responses = [
            "Я могу помочь с математикой, программированием, электротехникой и многим другим.",
            "Задайте мне любой вопрос - я постараюсь найти лучший ответ.",
            "Используйте меня для решения технических задач и получения информации.",
            "Я специализируюсь на технических вопросах и могу помочь с различными задачами."
        ]
        
        self.general_responses = [
            "Интересный вопрос! Давайте разберем его подробнее.",
            "Хороший вопрос. Я постараюсь дать вам полезный ответ.",
            "Понимаю ваш интерес к этой теме. Вот что я могу сказать...",
            "Отличный вопрос! Позвольте мне помочь вам с этим."
        ]
    
    def handle_request(self, message: str) -> Dict[str, Any]:
        """Обработка запроса"""
        message_lower = message.lower().strip()
        
        # Определяем тип запроса
        if any(word in message_lower for word in ["привет", "здравствуй", "добро пожаловать", "hi", "hello"]):
            return self._handle_greeting(message)
        elif any(word in message_lower for word in ["помощь", "справка", "help", "что ты умеешь"]):
            return self._handle_help(message)
        elif any(word in message_lower for word in ["как дела", "что нового", "как поживаешь"]):
            return self._handle_status(message)
        else:
            return self._handle_general(message)
    
    def _handle_greeting(self, message: str) -> Dict[str, Any]:
        """Обработка приветствия"""
        import random
        greeting = random.choice(self.greetings)
        
        return {
            "success": True,
            "response": {
                "content": greeting,
                "title": "Приветствие",
                "source": "General Handler"
            },
            "category": "greeting",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
            "provider": "General Handler"
        }
    
    def _handle_help(self, message: str) -> Dict[str, Any]:
        """Обработка запроса помощи"""
        import random
        help_text = random.choice(self.help_responses)
        
        help_content = f"""{help_text}

**Что я умею:**
• Решать математические задачи
• Помогать с программированием
• Объяснять электротехнику
• Работать с контроллерами и сервоприводами
• Анализировать данные
• И многое другое!

Просто задайте мне вопрос, и я найду лучший способ помочь вам."""
        
        return {
            "success": True,
            "response": {
                "content": help_content,
                "title": "Справка",
                "source": "General Handler"
            },
            "category": "help",
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat(),
            "provider": "General Handler"
        }
    
    def _handle_status(self, message: str) -> Dict[str, Any]:
        """Обработка запроса статуса"""
        status_content = """У меня все отлично! 

**Мой текущий статус:**
• Нейронная сеть активна и готова к работе
• База знаний подключена
• Все модули функционируют нормально
• Готов помочь с любыми техническими вопросами

Что бы вы хотели обсудить?"""
        
        return {
            "success": True,
            "response": {
                "content": status_content,
                "title": "Статус системы",
                "source": "General Handler"
            },
            "category": "status",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat(),
            "provider": "General Handler"
        }
    
    def _handle_general(self, message: str) -> Dict[str, Any]:
        """Обработка общих запросов"""
        import random
        general_text = random.choice(self.general_responses)
        
        response_content = f"""{general_text}

К сожалению, я не могу дать точный ответ на ваш вопрос: "{message}"

**Рекомендации:**
• Попробуйте переформулировать вопрос
• Уточните, в какой области вам нужна помощь
• Используйте более конкретные термины

**Я могу помочь с:**
• Математическими расчетами
• Программированием
• Электротехникой
• Работой с контроллерами
• Анализом данных

Задайте более конкретный вопрос, и я обязательно помогу!"""
        
        return {
            "success": True,
            "response": {
                "content": response_content,
                "title": "Общий ответ",
                "source": "General Handler"
            },
            "category": "general",
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat(),
            "provider": "General Handler"
        }

# Глобальный экземпляр
_general_handler = None

def get_general_handler():
    """Получает глобальный экземпляр обработчика"""
    global _general_handler
    if _general_handler is None:
        _general_handler = GeneralHandler()
    return _general_handler


