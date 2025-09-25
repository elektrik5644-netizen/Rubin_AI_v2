# LocalAI Architecture в Rubin AI v2

## Обзор архитектуры

LocalAI в системе Rubin AI v2 работает как интеллектуальный провайдер для генерации ответов на основе локальных языковых моделей.

## Компоненты системы

### 1. LocalAI Provider (`providers/localai_provider.py`)
```
┌─────────────────────────────────────┐
│           LocalAIProvider           │
├─────────────────────────────────────┤
│ • base_url: http://127.0.0.1:8080  │
│ • Поддержка OpenAI API совместимости│
│ • Модели: gpt-3.5-turbo, gpt-4     │
│ • Задачи: TEXT_GENERATION, CHAT     │
└─────────────────────────────────────┘
```

**Основные методы:**
- `initialize()` - проверка доступности сервера
- `get_available_models()` - получение списка моделей
- `generate_text()` - генерация текста
- `chat()` - чат через OpenAI API
- `answer_question()` - ответы на вопросы
- `summarize_text()` - суммаризация
- `translate_text()` - перевод

### 2. Local LLM Provider (`providers/local_llm_provider.py`)
```
┌─────────────────────────────────────┐
│         LocalLLMProvider           │
├─────────────────────────────────────┤
│ • Локальная генерация ответов      │
│ • Анализ документов                 │
│ • Специализированные ответы        │
│ • Fallback для недоступного LocalAI │
└─────────────────────────────────────┘
```

**Возможности:**
- Генерация ответов на основе документов
- Специализированные ответы по техническим темам
- Fallback ответы при недоступности внешних сервисов
- Поддержка мета-вопросов о процессе мышления

### 3. GPT Chatbot Integration (`providers/gpt_chatbot_integration.py`)
```
┌─────────────────────────────────────┐
│        GPTChatbotProvider           │
├─────────────────────────────────────┤
│ • OpenAI API клиент                 │
│ • История разговора                 │
│ • Системные промпты                 │
│ • Интеграция с базой знаний         │
└─────────────────────────────────────┘
```

**Функции:**
- Интеграция с OpenAI API через LocalAI
- Управление историей разговора
- Поиск в базе знаний
- Формирование контекста для GPT

## Архитектура интеграции

### Схема подключения
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │ Smart Dispatcher│    │  LocalAI Server │
│                 │    │   (Port 8080)   │    │   (Port 11434)  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │ HTTP Request         │                      │
          ├─────────────────────►│                      │
          │                      │                      │
          │                      │ OpenAI API Request   │
          │                      ├─────────────────────►│
          │                      │                      │
          │                      │ Response             │
          │                      │◄─────────────────────┤
          │                      │                      │
          │ HTTP Response        │                      │
          │◄─────────────────────┤                      │
          │                      │                      │
```

### Провайдеры в системе
```
┌─────────────────────────────────────────────────────────────┐
│                    Provider Selector                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ LocalAIProvider │  │LocalLLMProvider │  │GoogleCloud   │ │
│  │   (Priority 1)  │  │   (Priority 2)  │  │  Provider    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Процесс обработки запросов

### 1. Инициализация системы
```python
# В api/rubin_ai_v2_server.py
localai_provider = LocalAIProvider()
if localai_provider.initialize():
    provider_selector.register_provider(localai_provider)
```

### 2. Обработка запроса
```python
# В intelligent_dispatcher.py
def process_request(self, message: str):
    # 1. Категоризация запроса
    category = self.categorize_request(message)
    
    # 2. Выбор провайдера
    provider = self.select_provider(category)
    
    # 3. Генерация ответа
    if provider == "localai":
        response = localai_provider.chat(messages)
    elif provider == "local_llm":
        response = local_llm_provider.get_response(message, context)
    
    return response
```

### 3. Fallback механизм
```
┌─────────────────┐
│   LocalAI       │
│   (Primary)     │
└─────────┬───────┘
          │
          │ Недоступен
          ▼
┌─────────────────┐
│   Local LLM     │
│   (Fallback)    │
└─────────┬───────┘
          │
          │ Недоступен
          ▼
┌─────────────────┐
│   Static        │
│   Responses     │
└─────────────────┘
```

## Конфигурация LocalAI

### Настройки подключения
```python
# providers/localai_provider.py
def __init__(self, base_url: str = "http://127.0.0.1:8080"):
    self.base_url = base_url.rstrip('/')
    self.supported_tasks = [
        TaskType.TEXT_GENERATION,
        TaskType.CHAT,
        TaskType.QUESTION_ANSWERING,
        TaskType.SUMMARIZATION,
        TaskType.TRANSLATION
    ]
```

### Проверка здоровья
```python
def health_check(self) -> bool:
    try:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
```

## Интеграция с базой знаний

### GPT Knowledge Integrator
```python
class GPTKnowledgeIntegrator:
    def search_knowledge_base(self, query: str, documents: List[Dict]) -> str:
        # Поиск релевантных документов
        relevant_docs = self.find_relevant_documents(query, documents)
        
        # Формирование контекста
        context = self.build_context(relevant_docs)
        
        return context
```

### Процесс интеграции
1. **Поиск документов** - поиск релевантных документов по ключевым словам
2. **Извлечение контента** - извлечение релевантных частей из документов
3. **Формирование контекста** - создание контекста для GPT
4. **Генерация ответа** - отправка контекста и вопроса в LocalAI

## Мониторинг и логирование

### Метрики производительности
```python
@dataclass
class ModuleMetrics:
    module_name: str
    port: int
    request_count: int = 0
    avg_response_time: float = 0.0
    error_count: int = 0
    last_health_check: Optional[datetime] = None
    status: str = "unknown"  # online, offline, degraded
```

### Логирование
```python
# providers/localai_provider.py
logger = logging.getLogger(__name__)

def initialize(self) -> bool:
    try:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ LocalAI провайдер подключен к {self.base_url}")
            return True
        else:
            logger.warning(f"LocalAI сервер недоступен: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Ошибка подключения к LocalAI: {e}")
        return False
```

## Текущий статус

### Доступность LocalAI
- **Статус**: Недоступен (сервер не запущен)
- **Порт**: 8080 занят smart_dispatcher.py
- **Fallback**: Используется LocalLLMProvider

### Рекомендации по настройке
1. **Запустить LocalAI сервер** на отдельном порту (например, 11434)
2. **Обновить конфигурацию** в LocalAIProvider
3. **Настроить модели** в LocalAI сервере
4. **Протестировать интеграцию** с базой знаний

## Примеры использования

### Простой запрос
```python
# Через LocalAI
response = localai_provider.chat([
    {"role": "user", "content": "Объясни принцип работы ПИД-регулятора"}
])
```

### Запрос с контекстом
```python
# Через GPT Knowledge Integrator
context = knowledge_integrator.search_knowledge_base(query, documents)
response = gpt_provider.get_response(message, context)
```

### Fallback ответ
```python
# Через Local LLM Provider
response = local_llm_provider.get_response(message, context)
```

## Заключение

LocalAI в Rubin AI v2 обеспечивает:
- **Локальную генерацию** ответов без зависимости от внешних API
- **Интеграцию с базой знаний** для контекстных ответов
- **Fallback механизм** для обеспечения надежности
- **Мониторинг и метрики** для отслеживания производительности

Система готова к работе с LocalAI, но требует настройки и запуска LocalAI сервера.






