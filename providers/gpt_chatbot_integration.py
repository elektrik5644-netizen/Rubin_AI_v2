#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 GPT Chatbot Integration для Rubin AI
Интеграция с OpenAI GPT API для улучшения чат-бота
"""

import os
import json
import requests
import openai
from typing import Dict, List, Optional, Tuple
import time

class GPTChatbotProvider:
    """Провайдер для работы с GPT API"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Инициализация GPT провайдера
        
        Args:
            api_key: API ключ OpenAI
            model: Модель GPT для использования
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or "localai"
        self.model = model
        self.base_url = "http://localhost:11434/v1"
        
        # Настройка OpenAI клиента
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # История разговора
        self.conversation_history = []
        
        # Системный промпт для Rubin AI
        self.system_prompt = """Ты - Rubin AI, интеллектуальный помощник для программирования и анализа кода.

Твои возможности:
- Анализ и объяснение кода на различных языках программирования
- Помощь с PLC программированием
- Работа с PMAC системами
- Анализ технической документации
- Поиск решений в базе знаний

Стиль общения:
- Отвечай на русском языке
- Будь точным и профессиональным
- Приводи примеры кода когда это уместно
- Объясняй сложные концепции простым языком

Если у тебя есть доступ к базе знаний, используй эту информацию для более точных ответов."""

    def add_to_history(self, role: str, content: str):
        """Добавляет сообщение в историю разговора"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Ограничиваем историю последними 20 сообщениями
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def get_response(self, message: str, context: str = None) -> Tuple[str, str]:
        """
        Получает ответ от GPT
        
        Args:
            message: Сообщение пользователя
            context: Контекст из базы знаний
            
        Returns:
            Tuple[ответ, источник]
        """
        try:
            # Добавляем сообщение пользователя в историю
            self.add_to_history("user", message)
            
            # Формируем промпт с контекстом
            full_message = message
            if context:
                full_message = f"Контекст из базы знаний:\n{context}\n\nВопрос пользователя: {message}"
            
            # Подготавливаем сообщения для API
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Добавляем историю разговора
            messages.extend(self.conversation_history)
            
            # Заменяем последнее сообщение пользователя на полное с контекстом
            if messages:
                messages[-1]["content"] = full_message
            
            # Вызываем GPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                timeout=30
            )
            
            # Извлекаем ответ
            gpt_response = response.choices[0].message.content.strip()
            
            # Добавляем ответ в историю
            self.add_to_history("assistant", gpt_response)
            
            return gpt_response, "gpt"
            
        except Exception as e:
            print(f"❌ Ошибка GPT API: {e}")
            return self.get_fallback_response(message), "fallback"

    def get_fallback_response(self, message: str) -> str:
        """Fallback ответ когда GPT недоступен"""
        fallback_responses = {
            "привет": "Привет! Я Rubin AI. Готов помочь с программированием!",
            "помощь": "Я могу помочь с анализом кода, PLC программированием и техническими вопросами.",
            "статус": "GPT интеграция временно недоступна. Используется локальная система ответов.",
            "что умеешь": "Я умею анализировать код, работать с PLC, помогать с программированием."
        }
        
        message_lower = message.lower()
        for key, response in fallback_responses.items():
            if key in message_lower:
                return response
        
        return "Извините, GPT сервис временно недоступен. Попробуйте позже."

class GPTKnowledgeIntegrator:
    """Интегратор для работы с базой знаний и GPT"""
    
    def __init__(self, gpt_provider: GPTChatbotProvider):
        self.gpt_provider = gpt_provider
    
    def search_knowledge_base(self, query: str, documents: List[Dict]) -> str:
        """
        Поиск в базе знаний и формирование контекста
        
        Args:
            query: Поисковый запрос
            documents: Список документов
            
        Returns:
            Контекст для GPT
        """
        if not documents:
            return ""
        
        query_lower = query.lower()
        relevant_docs = []
        
        # Поиск релевантных документов
        for doc in documents:
            content_lower = doc.get('content', '').lower()
            title_lower = doc.get('title', '').lower()
            name_lower = doc.get('name', '').lower()
            
            # Проверяем совпадения в названии, заголовке и содержимом
            if (any(word in title_lower for word in query_lower.split() if len(word) > 3) or
                any(word in name_lower for word in query_lower.split() if len(word) > 3) or
                any(word in content_lower for word in query_lower.split() if len(word) > 3)):
                
                relevant_docs.append(doc)
        
        # Ограничиваем количество документов
        relevant_docs = relevant_docs[:3]
        
        if not relevant_docs:
            return ""
        
        # Формируем контекст
        context_parts = []
        for doc in relevant_docs:
            title = doc.get('title', doc.get('name', 'Документ'))
            content = doc.get('content', '')
            
            # Извлекаем релевантные части
            relevant_content = self.extract_relevant_content(content, query)
            if relevant_content:
                context_parts.append(f"📄 {title}:\n{relevant_content}")
        
        return "\n\n".join(context_parts)
    
    def extract_relevant_content(self, content: str, query: str) -> str:
        """Извлекает релевантное содержимое из документа"""
        query_words = [word for word in query.lower().split() if len(word) > 3]
        
        # Ищем предложения с ключевыми словами
        sentences = content.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            result = '. '.join(relevant_sentences[:3])
            if len(result) > 500:
                result = result[:500] + "..."
            return result
        
        # Если не найдено, возвращаем начало документа
        return content[:300] + "..." if len(content) > 300 else content

class GPTChatbotManager:
    """Менеджер для управления GPT чат-ботом"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.gpt_provider = GPTChatbotProvider(api_key, model)
        self.knowledge_integrator = GPTKnowledgeIntegrator(self.gpt_provider)
    
    def process_message(self, message: str, documents: List[Dict] = None) -> Dict:
        """
        Обрабатывает сообщение пользователя
        
        Args:
            message: Сообщение пользователя
            documents: База знаний
            
        Returns:
            Словарь с ответом и метаданными
        """
        try:
            # Поиск в базе знаний
            context = ""
            if documents:
                context = self.knowledge_integrator.search_knowledge_base(message, documents)
            
            # Получение ответа от GPT
            response, source = self.gpt_provider.get_response(message, context)
            
            return {
                "response": response,
                "ai_source": source,
                "context_used": bool(context),
                "model": self.gpt_provider.model,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"❌ Ошибка обработки сообщения: {e}")
            return {
                "response": "Извините, произошла ошибка при обработке вашего запроса.",
                "ai_source": "error",
                "context_used": False,
                "model": "error",
                "timestamp": time.time()
            }

# Функция для интеграции с существующим API
def get_gpt_response(message: str, documents: List[Dict] = None, api_key: str = None) -> Dict:
    """
    Получает ответ от GPT для интеграции с существующим API
    
    Args:
        message: Сообщение пользователя
        documents: База знаний
        api_key: API ключ OpenAI
        
    Returns:
        Словарь с ответом
    """
    manager = GPTChatbotManager(api_key)
    return manager.process_message(message, documents)

# Тестирование
if __name__ == "__main__":
    # Тест GPT интеграции
    print("🤖 Тестирование GPT интеграции...")
    
    # Создаем тестовые документы
    test_documents = [
        {
            "title": "Python основы",
            "content": "Python - это высокоуровневый язык программирования. Он прост в изучении и имеет чистый синтаксис.",
            "name": "python_basics.txt"
        },
        {
            "title": "PLC программирование",
            "content": "PLC (Programmable Logic Controller) - это промышленный контроллер для автоматизации процессов.",
            "name": "plc_guide.txt"
        }
    ]
    
    # Тестируем без API ключа (fallback режим)
    manager = GPTChatbotManager()
    
    test_messages = [
        "Привет! Что ты умеешь?",
        "Расскажи про Python",
        "Что такое PLC?",
        "Помоги с программированием"
    ]
    
    for message in test_messages:
        print(f"\n👤 Пользователь: {message}")
        result = manager.process_message(message, test_documents)
        print(f"🤖 Rubin AI ({result['ai_source']}): {result['response']}")
        print(f"📊 Контекст использован: {result['context_used']}")
