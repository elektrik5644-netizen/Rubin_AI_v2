"""
Конфигурация для Rubin AI Matrix Gateway
"""

import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Основные настройки
    HOST: str = "0.0.0.0"
    PORT: int = 80
    DEBUG: bool = False
    
    # База данных PostgreSQL
    POSTGRES_HOST: str = "postgres_db"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "rubin"
    POSTGRES_PASSWORD: str = "rubin_matrix_2025"
    POSTGRES_DB: str = "rubin_ai"
    
    # Qdrant векторная база
    QDRANT_HOST: str = "qdrant_db"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    
    # Ollama AI сервис
    OLLAMA_HOST: str = "ollama_service"
    OLLAMA_PORT: int = 11434
    
    # Compute Core
    COMPUTE_CORE_HOST: str = "compute_core"
    COMPUTE_CORE_PORT: int = 5000
    
    # Безопасность
    JWT_SECRET: str = "rubin_matrix_jwt_secret_key_2025"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30
    
    # API ключи
    API_KEY: str = "rubin_matrix_api_key_2025"
    
    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/gateway.log"
    
    # Мониторинг
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Кэширование
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
    
    @property
    def ollama_url(self) -> str:
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"
    
    @property
    def compute_core_url(self) -> str:
        return f"http://{self.COMPUTE_CORE_HOST}:{self.COMPUTE_CORE_PORT}"
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Создание экземпляра настроек
settings = Settings()
