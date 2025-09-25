"""
LocalAI Provider для Rubin AI v2.0
Интеграция с локальным LocalAI сервером
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider, TaskType

logger = logging.getLogger(__name__)

class LocalAIProvider(BaseProvider):
    """Провайдер для работы с LocalAI"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        super().__init__("LocalAI", priority=1)
        self.base_url = base_url.rstrip('/')
        self.supported_tasks = [
            TaskType.GENERAL_CHAT,
            TaskType.CODE_GENERATION,
            TaskType.DOCUMENTATION,
            TaskType.TECHNICAL_DOCUMENTATION
        ]
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Инициализация LocalAI провайдера"""
        try:
            # Проверяем доступность сервера
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.is_initialized = True
                logger.info(f"✅ LocalAI провайдер подключен к {self.base_url}")
                
                # Получаем список доступных моделей
                models = self.get_available_models()
                logger.info(f"Доступные модели LocalAI: {models}")
                return True
            else:
                logger.warning(f"LocalAI сервер недоступен: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка подключения к LocalAI: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Получить список доступных моделей"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            return []
        except Exception as e:
            logger.error(f"Ошибка получения моделей: {e}")
            return []
    
    def generate_text(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Генерация текста через LocalAI"""
        try:
            payload = {
                "model": kwargs.get('model', 'gpt-3.5-turbo'),
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 1.0)
            }
            
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['text'].strip()
            else:
                logger.error(f"Ошибка генерации: {response.status_code}")
                return "Ошибка генерации текста"
                
        except Exception as e:
            logger.error(f"Ошибка LocalAI генерации: {e}")
            return "Ошибка подключения к LocalAI"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Чат через LocalAI"""
        try:
            payload = {
                "model": kwargs.get('model', 'gpt-3.5-turbo'),
                "messages": messages,
                "max_tokens": kwargs.get('max_tokens', 500),
                "temperature": kwargs.get('temperature', 0.7)
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"Ошибка чата: {response.status_code}")
                return "Ошибка обработки чата"
                
        except Exception as e:
            logger.error(f"Ошибка LocalAI чата: {e}")
            return "Ошибка подключения к LocalAI"
    
    def answer_question(self, question: str, context: str = "", **kwargs) -> str:
        """Ответ на вопрос с контекстом"""
        if context:
            prompt = f"Контекст: {context}\n\nВопрос: {question}\n\nОтвет:"
        else:
            prompt = f"Вопрос: {question}\n\nОтвет:"
        
        return self.generate_text(prompt, **kwargs)
    
    def summarize_text(self, text: str, max_length: int = 5000, **kwargs) -> str:
        """Суммаризация текста"""
        prompt = f"Суммаризируй следующий текст в {max_length} символов:\n\n{text}\n\nКраткое изложение:"
        return self.generate_text(prompt, **kwargs)
    
    def translate_text(self, text: str, target_language: str = "русский", **kwargs) -> str:
        """Перевод текста"""
        prompt = f"Переведи следующий текст на {target_language}:\n\n{text}\n\nПеревод:"
        return self.generate_text(prompt, **kwargs)
    
    def get_response(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Получить ответ от LocalAI провайдера"""
        try:
            # Используем чат для получения ответа
            messages = [{"role": "user", "content": message}]
            response_text = self.chat(messages)
            
            return {
                'content': response_text,
                'provider': self.name,
                'task_type': 'general_chat',
                'metadata': {
                    'model': 'gpt-3.5-turbo',
                    'base_url': self.base_url
                },
                'thinking_process': [],
                'timestamp': None,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения ответа: {e}")
            return {
                'content': f"Ошибка LocalAI: {str(e)}",
                'provider': self.name,
                'task_type': 'general_chat',
                'metadata': {},
                'thinking_process': [],
                'timestamp': None,
                'success': False,
                'error': str(e)
            }
    
    def get_capabilities(self) -> List[str]:
        """Получить список возможностей провайдера"""
        return [task.value for task in self.supported_tasks]
    
    def health_check(self) -> bool:
        """Проверка здоровья провайдера"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False





