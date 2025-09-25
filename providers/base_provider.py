"""
Rubin AI v2.0 - Базовый класс для AI провайдеров
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging

class BaseProvider(ABC):
    """Базовый класс для всех AI провайдеров"""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # Чем меньше число, тем выше приоритет
        self.logger = logging.getLogger(f"rubin_ai.{name}")
        self.is_available = False
        self.last_error = None
        
    @abstractmethod
    def initialize(self) -> bool:
        """Инициализация провайдера"""
        pass
    
    @abstractmethod
    def get_response(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Получить ответ от провайдера"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Получить список возможностей провайдера"""
        pass
    
    def is_suitable_for_task(self, task_type: str) -> bool:
        """Проверить, подходит ли провайдер для задачи"""
        capabilities = self.get_capabilities()
        return task_type in capabilities
    
    def get_priority_for_task(self, task_type: str) -> int:
        """Получить приоритет провайдера для конкретной задачи"""
        if self.is_suitable_for_task(task_type):
            return self.priority
        return 999  # Низкий приоритет для неподходящих задач
    
    def log_error(self, error: Exception):
        """Логировать ошибку"""
        self.last_error = str(error)
        self.logger.error(f"Ошибка в {self.name}: {error}")
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус провайдера"""
        return {
            'name': self.name,
            'available': self.is_available,
            'priority': self.priority,
            'capabilities': self.get_capabilities(),
            'last_error': self.last_error
        }

class TaskType:
    """Типы задач для AI провайдеров"""
    
    # Основные типы
    GENERAL_CHAT = "general_chat"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    DOCUMENTATION = "documentation"
    SECURITY_CHECK = "security_check"
    
    # Специализированные типы
    PLC_ANALYSIS = "plc_analysis"
    PMAC_ANALYSIS = "pmac_analysis"
    CNC_ANALYSIS = "cnc_analysis"
    SCHEMATIC_ANALYSIS = "schematic_analysis"
    
    # Мультимедиа
    IMAGE_ANALYSIS = "image_analysis"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    
    # Корпоративные
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    DATA_ANALYSIS = "data_analysis"
    PREDICTION = "prediction"

class ResponseFormat:
    """Формат ответа от AI провайдера"""
    
    @staticmethod
    def create_response(
        content: str,
        provider: str,
        task_type: str,
        metadata: Optional[Dict] = None,
        thinking_process: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Создать стандартизированный ответ"""
        return {
            'content': content,
            'provider': provider,
            'task_type': task_type,
            'metadata': metadata or {},
            'thinking_process': thinking_process or [],
            'timestamp': None,  # Будет установлен в API
            'success': True,
            'error': None
        }
    
    @staticmethod
    def create_error_response(
        error: str,
        provider: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Создать ответ об ошибке"""
        return {
            'content': f"Ошибка в {provider}: {error}",
            'provider': provider,
            'task_type': task_type,
            'metadata': {},
            'thinking_process': [],
            'timestamp': None,
            'success': False,
            'error': error
        }
