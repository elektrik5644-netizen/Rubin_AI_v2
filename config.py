"""
Rubin AI v2.0 - Конфигурация системы
Расширенная многоуровневая система AI провайдеров
"""

import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class Config:
    """Основная конфигурация Rubin AI v2.0"""
    
    # Основные настройки
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 8084  # Новый порт для v2.0
    
    # База данных
    DATABASE_PATH = 'rubin_ai_v2.db'
    DOCUMENTS_STORAGE = 'documents_storage_v2.pkl'
    
    # API ключи
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    ZAI_API_KEY = os.getenv('ZAI_API_KEY')
    GOOGLE_CLOUD_CREDENTIALS = os.getenv('GOOGLE_CLOUD_CREDENTIALS')
    IBM_WATSON_API_KEY = os.getenv('IBM_WATSON_API_KEY')
    IBM_WATSON_URL = os.getenv('IBM_WATSON_URL')
    
    # Модели AI
    GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-3.5-turbo')
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
    ZAI_MODEL = os.getenv('ZAI_MODEL', 'GLM-4.5-Air')
    
    # Hugging Face модели
    HUGGINGFACE_MODELS = {
        'code_analyzer': 'microsoft/codebert-base',
        'safety_checker': 'distilbert-base-uncased',
        'plc_analyzer': 'Salesforce/codegen-350M-mono',
        'text_generator': 'gpt2'
    }
    
    # Google Cloud настройки
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    GOOGLE_CLOUD_REGION = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
    
    # Лимиты и квоты
    MAX_TOKENS_PER_REQUEST = 2000
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_CONVERSATION_HISTORY = 10
    
    # Поддерживаемые форматы файлов
    SUPPORTED_FILE_EXTENSIONS = {
        'code': ['.py', '.js', '.ts', '.cpp', '.c', '.h', '.hpp', '.java', '.cs'],
        'plc': ['.plc', '.st', '.iec', '.cfg', '.ini', '.conf', '.config'],
        'cnc': ['.gcode', '.nc', '.cnc', '.tap'],
        'pmac': ['.pmac', '.pma'],
        'documents': ['.txt', '.md', '.pdf', '.doc', '.docx'],
        'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'],
        'schematics': ['.sch', '.brd', '.kicad_pcb', '.kicad_sch']
    }
    
    # Настройки безопасности
    ALLOWED_ORIGINS = [
        'http://localhost:8084', 
        'http://127.0.0.1:8084',
        'http://localhost:3000',
        'http://127.0.0.1:3000',
        'http://localhost:8080',
        'http://127.0.0.1:8080',
        'file://',
        '*'
    ]
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Логирование
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'rubin_ai_v2.log'
    
    # Мониторинг
    ENABLE_METRICS = True
    METRICS_PORT = 9090
    
    @classmethod
    def get_provider_status(cls):
        """Получить статус всех провайдеров"""
        return {
            'openai': bool(cls.OPENAI_API_KEY),
            'anthropic': bool(cls.ANTHROPIC_API_KEY),
            'zai': bool(cls.ZAI_API_KEY),
            'google_cloud': bool(cls.GOOGLE_CLOUD_CREDENTIALS),
            'ibm_watson': bool(cls.IBM_WATSON_API_KEY),
            'huggingface': True,  # Всегда доступен
            'local': True  # Всегда доступен как fallback
        }
    
    @classmethod
    def get_available_providers(cls):
        """Получить список доступных провайдеров"""
        status = cls.get_provider_status()
        return [provider for provider, available in status.items() if available]
