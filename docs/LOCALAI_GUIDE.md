# LocalAI Integration Guide

Руководство по настройке и использованию LocalAI интеграции в Rubin AI v2.

## 🤖 Обзор

LocalAI интеграция обеспечивает работу с локальными AI моделями, позволяя системе Rubin AI функционировать полностью офлайн. Поддерживает различные провайдеры и умный выбор модели для конкретных задач.

## 🚀 Возможности

- **Локальная обработка** - работа без интернета
- **Множественные провайдеры** - поддержка различных AI сервисов
- **Умный выбор модели** - автоматический выбор лучшей модели для задачи
- **Кэширование ответов** - оптимизация производительности
- **Fallback механизм** - резервные варианты при недоступности

## ⚙️ Настройка

### 1. Установка LocalAI

```bash
# Установка LocalAI сервера
docker run -d --name localai -p 11434:11434 localai/localai:latest

# Или через Docker Compose
docker-compose up -d localai
```

### 2. Настройка переменных окружения

```bash
# Добавьте в .env файл
LOCALAI_URL=http://127.0.0.1:11434
LOCALAI_MODEL=llama2
LOCALAI_TIMEOUT=30
```

### 3. Запуск провайдера

```bash
python providers/localai_provider.py
```

## 🏗️ Архитектура

### Структура провайдеров

```
providers/
├── base_provider.py           # Базовый класс провайдера
├── localai_provider.py       # LocalAI провайдер
├── google_cloud_provider.py  # Google Cloud провайдер
├── huggingface_provider.py   # HuggingFace провайдер
├── gpt_chatbot_integration.py # GPT интеграция
└── smart_provider_selector.py # Умный выбор провайдера
```

### Типы задач

```python
class TaskType(Enum):
    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    DOCUMENTATION = "documentation"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    MATHEMATICAL_SOLVING = "mathematical_solving"
    ELECTRICAL_ANALYSIS = "electrical_analysis"
```

## 📡 API Endpoints

### Основные эндпоинты

- `POST /api/localai/chat` - Чат с локальной моделью
- `POST /api/localai/generate` - Генерация текста
- `GET /api/localai/models` - Доступные модели
- `GET /api/localai/health` - Проверка здоровья
- `POST /api/localai/select` - Выбор провайдера

### Примеры запросов

#### Чат с моделью
```json
POST /api/localai/chat
{
    "message": "Объясни принцип работы нейросети",
    "model": "llama2",
    "max_tokens": 500,
    "temperature": 0.7
}
```

#### Генерация кода
```json
POST /api/localai/generate
{
    "prompt": "Напиши функцию для сортировки массива",
    "task_type": "code_generation",
    "language": "python"
}
```

## 🔧 Конфигурация провайдеров

### LocalAI Provider

```python
class LocalAIProvider(BaseProvider):
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        super().__init__("LocalAI", priority=1)
        self.base_url = base_url
        self.supported_tasks = [
            TaskType.GENERAL_CHAT,
            TaskType.CODE_GENERATION,
            TaskType.DOCUMENTATION
        ]
```

### Google Cloud Provider

```python
class GoogleCloudProvider(BaseProvider):
    def __init__(self, project_id: str, credentials_path: str):
        super().__init__("GoogleCloud", priority=2)
        self.project_id = project_id
        self.credentials_path = credentials_path
```

### HuggingFace Provider

```python
class HuggingFaceProvider(BaseProvider):
    def __init__(self, api_token: str):
        super().__init__("HuggingFace", priority=3)
        self.api_token = api_token
```

## 🧠 Умный выбор провайдера

### Алгоритм выбора

```python
def select_best_provider(task_type: TaskType, context: dict) -> BaseProvider:
    """Выбор лучшего провайдера для задачи"""
    
    # Фильтрация по поддерживаемым задачам
    available_providers = [
        p for p in ALL_PROVIDERS 
        if task_type in p.supported_tasks and p.is_available()
    ]
    
    if not available_providers:
        return get_fallback_provider()
    
    # Сортировка по приоритету и производительности
    available_providers.sort(
        key=lambda p: (p.priority, p.get_performance_score(task_type))
    )
    
    return available_providers[0]
```

### Критерии выбора

1. **Приоритет провайдера** - настройка важности
2. **Поддержка типа задачи** - совместимость с задачей
3. **Производительность** - скорость и качество ответов
4. **Доступность** - статус провайдера
5. **Стоимость** - экономические факторы

## 🔄 Fallback механизм

### Цепочка резервных вариантов

```python
def get_response_with_fallback(task_type: TaskType, request_data: dict):
    """Получение ответа с резервными вариантами"""
    
    providers_chain = [
        LocalAIProvider(),
        GoogleCloudProvider(),
        HuggingFaceProvider(),
        GPTProvider()
    ]
    
    for provider in providers_chain:
        try:
            if provider.is_available() and task_type in provider.supported_tasks:
                response = provider.process_request(task_type, request_data)
                if response.success:
                    return response
        except Exception as e:
            logger.warning(f"Provider {provider.name} failed: {e}")
            continue
    
    return ErrorResponse("All providers unavailable")
```

## 📊 Мониторинг и метрики

### Метрики производительности

```python
class ProviderMetrics:
    def __init__(self):
        self.response_times = {}
        self.success_rates = {}
        self.error_counts = {}
        self.token_usage = {}
    
    def record_response(self, provider_name: str, response_time: float, 
                       success: bool, tokens_used: int):
        """Запись метрик ответа"""
        if provider_name not in self.response_times:
            self.response_times[provider_name] = []
        
        self.response_times[provider_name].append(response_time)
        
        if success:
            self.success_rates[provider_name] = self.success_rates.get(provider_name, 0) + 1
        else:
            self.error_counts[provider_name] = self.error_counts.get(provider_name, 0) + 1
        
        self.token_usage[provider_name] = self.token_usage.get(provider_name, 0) + tokens_used
```

### Health Check

```python
def check_provider_health(provider: BaseProvider) -> dict:
    """Проверка здоровья провайдера"""
    health_status = {
        "name": provider.name,
        "available": provider.is_available(),
        "response_time": provider.get_average_response_time(),
        "success_rate": provider.get_success_rate(),
        "last_error": provider.get_last_error()
    }
    return health_status
```

## 🛠️ Разработка

### Создание нового провайдера

```python
class CustomProvider(BaseProvider):
    def __init__(self, config: dict):
        super().__init__("CustomProvider", priority=5)
        self.config = config
        self.supported_tasks = [TaskType.GENERAL_CHAT]
    
    def initialize(self) -> bool:
        """Инициализация провайдера"""
        try:
            # Логика инициализации
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def process_request(self, task_type: TaskType, request_data: dict) -> Response:
        """Обработка запроса"""
        # Логика обработки
        return Response(success=True, data=result)
```

### Добавление нового типа задачи

```python
# В base_provider.py
class TaskType(Enum):
    # Существующие типы...
    CUSTOM_TASK = "custom_task"

# В провайдере
def __init__(self):
    self.supported_tasks = [
        TaskType.GENERAL_CHAT,
        TaskType.CUSTOM_TASK  # Добавляем новый тип
    ]
```

## 🔒 Безопасность

### Аутентификация

```python
def authenticate_provider(provider: BaseProvider, credentials: dict) -> bool:
    """Аутентификация провайдера"""
    if provider.name == "GoogleCloud":
        return validate_google_credentials(credentials)
    elif provider.name == "HuggingFace":
        return validate_huggingface_token(credentials["token"])
    
    return True  # LocalAI не требует аутентификации
```

### Шифрование

```python
def encrypt_sensitive_data(data: str, key: str) -> str:
    """Шифрование чувствительных данных"""
    from cryptography.fernet import Fernet
    f = Fernet(key.encode())
    return f.encrypt(data.encode()).decode()
```

## 🐛 Отладка

### Общие проблемы

1. **LocalAI недоступен**
   - Проверьте статус Docker контейнера
   - Убедитесь в правильности URL
   - Проверьте логи LocalAI

2. **Медленные ответы**
   - Оптимизируйте размер модели
   - Используйте GPU ускорение
   - Настройте кэширование

3. **Ошибки аутентификации**
   - Проверьте токены API
   - Убедитесь в правильности учетных данных
   - Проверьте лимиты API

### Логирование

```python
import logging
logger = logging.getLogger("localai_provider")

def log_provider_activity(provider_name: str, activity: str, details: dict):
    logger.info(f"{provider_name}: {activity} - {details}")
```

## 🚀 Развертывание

### Локальное развертывание

```bash
# Запуск LocalAI
docker run -d -p 11434:11434 localai/localai:latest

# Запуск провайдеров
python providers/localai_provider.py
```

### Docker Compose

```yaml
version: '3.8'
services:
  localai:
    image: localai/localai:latest
    ports:
      - "11434:11434"
    volumes:
      - ./models:/models
  
  rubin-providers:
    build: .
    depends_on:
      - localai
    environment:
      - LOCALAI_URL=http://localai:11434
```

### Production настройки

- Используйте кластер LocalAI для высокой доступности
- Настройте мониторинг всех провайдеров
- Реализуйте автоматическое переключение при сбоях
- Добавьте метрики и алерты

## 📚 Дополнительные ресурсы

- [LocalAI Documentation](https://localai.io/)
- [Ollama Documentation](https://ollama.ai/)
- [HuggingFace API](https://huggingface.co/docs/api-inference)
- [Google Cloud AI](https://cloud.google.com/ai)
