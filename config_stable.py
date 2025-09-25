"""
Rubin AI v2.0 - Стабильная конфигурация системы
"""

import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class StableConfig:
    """Стабильная конфигурация Rubin AI v2.0"""
    
    # Основные настройки
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 8084
    
    # База данных
    DATABASE_PATH = 'rubin_ai_v2.db'
    DOCUMENTS_STORAGE = 'documents_storage_v2.pkl'
    
    # API ключи (опционально)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # Модели AI (стабильные настройки)
    GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-3.5-turbo')
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
    
    # Hugging Face модели (стабильные)
    HUGGINGFACE_MODELS = {
        'code_analyzer': 'microsoft/codebert-base',
        'safety_checker': 'distilbert-base-uncased',
        'text_generator': 'gpt2'
    }
    
    # Лимиты и квоты (консервативные)
    MAX_TOKENS_PER_REQUEST = 1000
    MAX_REQUESTS_PER_MINUTE = 30
    MAX_CONVERSATION_HISTORY = 5
    
    # Поддерживаемые форматы файлов
    SUPPORTED_FILE_EXTENSIONS = {
        'code': ['.py', '.js', '.ts', '.cpp', '.c', '.h'],
        'plc': ['.plc', '.st', '.iec'],
        'documents': ['.txt', '.md', '.pdf', '.doc', '.docx'],
        'images': ['.png', '.jpg', '.jpeg', '.gif']
    }
    
    # Настройки безопасности
    ALLOWED_ORIGINS = [
        'http://localhost:8084', 
        'http://127.0.0.1:8084',
        'http://localhost:3000',
        'http://127.0.0.1:3000',
        'file://'
    ]
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
    # Логирование (без эмодзи для стабильности)
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'rubin_ai_v2.log'
    
    # Мониторинг
    ENABLE_METRICS = True
    METRICS_PORT = 9090
    
    # Настройки стабильности
    REQUEST_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    CACHE_TTL = 300  # 5 минут
    
    # Провайдеры (приоритеты)
    PROVIDER_PRIORITIES = {
        'specialized': 1,      # Встроенные ответы - высший приоритет
        'huggingface': 2,      # HuggingFace - средний приоритет
        'openai': 3,           # OpenAI - низкий приоритет (fallback)
        'anthropic': 4         # Anthropic - самый низкий приоритет
    }
    
    # Категории и их настройки
    CATEGORIES = {
        'programming': {
            'backend': 'specialized',
            'fallback': 'huggingface',
            'timeout': 10
        },
        'controllers': {
            'backend': 'specialized',
            'fallback': 'specialized',
            'timeout': 5
        },
        'electrical': {
            'backend': 'specialized',
            'fallback': 'huggingface',
            'timeout': 10
        },
        'radiomechanics': {
            'backend': 'specialized',
            'fallback': 'huggingface',
            'timeout': 10
        },
        'general': {
            'backend': 'huggingface',
            'fallback': 'openai',
            'timeout': 15
        }
    }
    
    @classmethod
    def get_provider_status(cls):
        """Получить статус всех провайдеров"""
        return {
            'specialized': True,  # Всегда доступен
            'huggingface': True,  # Всегда доступен
            'openai': bool(cls.OPENAI_API_KEY),
            'anthropic': bool(cls.ANTHROPIC_API_KEY)
        }
    
    @classmethod
    def get_available_providers(cls):
        """Получить список доступных провайдеров"""
        status = cls.get_provider_status()
        return [provider for provider, available in status.items() if available]
    
    @classmethod
    def get_category_config(cls, category):
        """Получить конфигурацию для категории"""
        return cls.CATEGORIES.get(category, cls.CATEGORIES['general'])
    
    @classmethod
    def get_provider_priority(cls, provider):
        """Получить приоритет провайдера"""
        return cls.PROVIDER_PRIORITIES.get(provider, 999)

















