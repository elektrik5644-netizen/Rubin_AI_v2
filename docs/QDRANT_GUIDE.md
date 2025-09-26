# Qdrant Vector Database Guide

Руководство по настройке и использованию Qdrant векторной базы данных в Rubin AI v2.

## 🗄️ Обзор

Qdrant - это высокопроизводительная векторная база данных для семантического поиска и машинного обучения. В системе Rubin AI v2 используется для индексации документов, знаний и обеспечения семантического поиска.

## 🚀 Возможности

- **Векторный поиск** - семантический поиск по документам
- **Индексация знаний** - автоматическая индексация базы знаний
- **Масштабируемость** - поддержка больших объемов данных
- **Cloudflare интеграция** - CDN и защита
- **REST API** - простой интерфейс для интеграции

## ⚙️ Настройка

### 1. Установка Qdrant

#### Docker (рекомендуется)
```bash
# Запуск Qdrant через Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# С персистентным хранилищем
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

#### Локальная установка
```bash
# Скачивание бинарного файла
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
./qdrant
```

### 2. Настройка переменных окружения

```bash
# Добавьте в .env файл
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=your_api_key_here
QDRANT_COLLECTION_NAME=rubin_knowledge
```

### 3. Инициализация коллекции

```bash
python setup_qdrant.py
```

## 🏗️ Архитектура

### Структура данных

```
Qdrant Collections:
├── rubin_knowledge          # Основная база знаний
│   ├── documents           # Документы и тексты
│   ├── code_snippets       # Фрагменты кода
│   └── technical_docs      # Техническая документация
├── rubin_context           # Контекстные данные
│   ├── conversations       # История диалогов
│   └── user_preferences    # Пользовательские настройки
└── rubin_vectors           # Векторные представления
    ├── embeddings          # Эмбеддинги текстов
    └── semantic_vectors    # Семантические векторы
```

### Конфигурация коллекций

```python
COLLECTION_CONFIG = {
    "rubin_knowledge": {
        "vector_size": 384,  # Размер вектора (зависит от модели)
        "distance": "Cosine",  # Метрика расстояния
        "payload_schema": {
            "title": "text",
            "content": "text", 
            "category": "keyword",
            "timestamp": "datetime"
        }
    },
    "rubin_context": {
        "vector_size": 512,
        "distance": "Euclidean",
        "payload_schema": {
            "user_id": "keyword",
            "session_id": "keyword",
            "context_type": "keyword"
        }
    }
}
```

## 📡 API Endpoints

### Основные эндпоинты

- `POST /api/vector/search` - Семантический поиск
- `POST /api/vector/index` - Индексация документа
- `POST /api/vector/batch_index` - Массовая индексация
- `GET /api/vector/collections` - Список коллекций
- `DELETE /api/vector/delete` - Удаление векторов

### Примеры запросов

#### Семантический поиск
```json
POST /api/vector/search
{
    "query": "как работает нейросеть",
    "collection": "rubin_knowledge",
    "limit": 10,
    "score_threshold": 0.7,
    "filter": {
        "category": "technical"
    }
}
```

#### Индексация документа
```json
POST /api/vector/index
{
    "collection": "rubin_knowledge",
    "document": {
        "id": "doc_001",
        "title": "Основы нейросетей",
        "content": "Нейросеть - это вычислительная модель...",
        "category": "technical",
        "timestamp": "2024-01-15T10:00:00Z"
    }
}
```

## 🔍 Семантический поиск

### Алгоритм поиска

```python
def semantic_search(query: str, collection: str, limit: int = 10) -> List[dict]:
    """Семантический поиск по коллекции"""
    
    # Генерация эмбеддинга запроса
    query_vector = generate_embedding(query)
    
    # Поиск в Qdrant
    search_results = qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        score_threshold=0.7
    )
    
    # Обработка результатов
    results = []
    for result in search_results:
        results.append({
            "id": result.id,
            "score": result.score,
            "payload": result.payload,
            "content": result.payload.get("content", "")
        })
    
    return results
```

### Фильтрация результатов

```python
def filtered_search(query: str, filters: dict) -> List[dict]:
    """Поиск с фильтрацией"""
    
    query_vector = generate_embedding(query)
    
    # Построение фильтра
    qdrant_filter = None
    if filters:
        qdrant_filter = {
            "must": [
                {"key": key, "match": {"value": value}}
                for key, value in filters.items()
            ]
        }
    
    search_results = qdrant_client.search(
        collection_name="rubin_knowledge",
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=10
    )
    
    return process_search_results(search_results)
```

## 📊 Индексация документов

### Автоматическая индексация

```python
def index_document(document: dict) -> bool:
    """Индексация документа в Qdrant"""
    
    try:
        # Генерация эмбеддинга
        content = document.get("content", "")
        vector = generate_embedding(content)
        
        # Подготовка точки
        point = {
            "id": document["id"],
            "vector": vector,
            "payload": {
                "title": document.get("title", ""),
                "content": content,
                "category": document.get("category", "general"),
                "timestamp": document.get("timestamp", datetime.now().isoformat())
            }
        }
        
        # Добавление в коллекцию
        qdrant_client.upsert(
            collection_name="rubin_knowledge",
            points=[point]
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to index document {document['id']}: {e}")
        return False
```

### Массовая индексация

```python
def batch_index_documents(documents: List[dict], batch_size: int = 100) -> int:
    """Массовая индексация документов"""
    
    indexed_count = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Подготовка точек
        points = []
        for doc in batch:
            vector = generate_embedding(doc["content"])
            point = {
                "id": doc["id"],
                "vector": vector,
                "payload": doc
            }
            points.append(point)
        
        # Индексация батча
        try:
            qdrant_client.upsert(
                collection_name="rubin_knowledge",
                points=points
            )
            indexed_count += len(points)
            
        except Exception as e:
            logger.error(f"Failed to index batch {i//batch_size}: {e}")
    
    return indexed_count
```

## 🔄 Интеграция с Cloudflare

### Настройка CDN

```python
def setup_cloudflare_integration():
    """Настройка интеграции с Cloudflare"""
    
    cloudflare_config = {
        "api_token": os.getenv("CLOUDFLARE_API_TOKEN"),
        "zone_id": os.getenv("CLOUDFLARE_ZONE_ID"),
        "domain": os.getenv("CLOUDFLARE_DOMAIN")
    }
    
    # Создание DNS записи для Qdrant
    create_dns_record(
        zone_id=cloudflare_config["zone_id"],
        name="qdrant",
        content=os.getenv("QDRANT_SERVER_IP"),
        type="A"
    )
    
    # Настройка Page Rules для кэширования
    create_page_rule(
        zone_id=cloudflare_config["zone_id"],
        url_pattern="qdrant.yourdomain.com/api/*",
        settings={
            "cache_level": "cache_everything",
            "edge_cache_ttl": 3600
        }
    )
```

### Мониторинг через Dash

```python
def setup_dash_monitoring():
    """Настройка мониторинга через Cloudflare Dash"""
    
    dash_config = {
        "api_key": os.getenv("DASH_API_KEY"),
        "dashboard_id": os.getenv("DASH_DASHBOARD_ID")
    }
    
    # Добавление метрик Qdrant
    metrics = [
        {
            "name": "qdrant_search_requests",
            "type": "counter",
            "description": "Количество поисковых запросов"
        },
        {
            "name": "qdrant_index_operations", 
            "type": "counter",
            "description": "Количество операций индексации"
        },
        {
            "name": "qdrant_response_time",
            "type": "histogram",
            "description": "Время ответа Qdrant"
        }
    ]
    
    for metric in metrics:
        create_dash_metric(dash_config, metric)
```

## 🛠️ Разработка

### Создание кастомных коллекций

```python
def create_custom_collection(name: str, config: dict) -> bool:
    """Создание пользовательской коллекции"""
    
    try:
        qdrant_client.create_collection(
            collection_name=name,
            vectors_config={
                "size": config["vector_size"],
                "distance": config["distance"]
            }
        )
        
        # Настройка индексов
        if "payload_schema" in config:
            for field, field_type in config["payload_schema"].items():
                qdrant_client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=field_type
                )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create collection {name}: {e}")
        return False
```

### Расширенные запросы

```python
def advanced_search(query: str, options: dict) -> List[dict]:
    """Расширенный поиск с дополнительными опциями"""
    
    query_vector = generate_embedding(query)
    
    # Построение сложного фильтра
    filters = build_advanced_filter(options.get("filters", {}))
    
    # Настройка поиска
    search_params = {
        "collection_name": options.get("collection", "rubin_knowledge"),
        "query_vector": query_vector,
        "limit": options.get("limit", 10),
        "score_threshold": options.get("score_threshold", 0.5),
        "query_filter": filters,
        "with_payload": True,
        "with_vectors": options.get("include_vectors", False)
    }
    
    # Выполнение поиска
    results = qdrant_client.search(**search_params)
    
    return format_search_results(results, options)
```

## 🐛 Отладка

### Общие проблемы

1. **Qdrant недоступен**
   - Проверьте статус Docker контейнера
   - Убедитесь в правильности URL и портов
   - Проверьте логи Qdrant

2. **Медленный поиск**
   - Оптимизируйте размер коллекции
   - Настройте индексы для часто используемых полей
   - Используйте фильтрацию для уменьшения области поиска

3. **Ошибки индексации**
   - Проверьте формат векторов
   - Убедитесь в совместимости размеров векторов
   - Проверьте лимиты памяти

### Логирование

```python
import logging
logger = logging.getLogger("qdrant_client")

def log_search_activity(query: str, results_count: int, response_time: float):
    logger.info(f"Search query: '{query}' returned {results_count} results in {response_time:.2f}s")
```

## 📊 Мониторинг

### Метрики производительности

```python
class QdrantMetrics:
    def __init__(self):
        self.search_requests = 0
        self.index_operations = 0
        self.average_response_time = 0
        self.error_count = 0
    
    def record_search(self, response_time: float, success: bool):
        """Запись метрик поиска"""
        self.search_requests += 1
        if success:
            self.average_response_time = (
                (self.average_response_time * (self.search_requests - 1) + response_time) 
                / self.search_requests
            )
        else:
            self.error_count += 1
```

### Health Check

```python
def check_qdrant_health() -> dict:
    """Проверка здоровья Qdrant"""
    
    try:
        # Проверка доступности
        collections = qdrant_client.get_collections()
        
        # Проверка производительности
        start_time = time.time()
        qdrant_client.search(
            collection_name="rubin_knowledge",
            query_vector=[0.0] * 384,  # Тестовый вектор
            limit=1
        )
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "collections_count": len(collections.collections),
            "response_time": response_time,
            "memory_usage": get_qdrant_memory_usage()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

## 🚀 Развертывание

### Локальное развертывание

```bash
# Запуск Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Инициализация коллекций
python setup_qdrant.py
```

### Docker Compose

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334

volumes:
  qdrant_storage:
```

### Production настройки

- Используйте кластер Qdrant для высокой доступности
- Настройте репликацию данных
- Реализуйте автоматическое масштабирование
- Добавьте мониторинг и алерты
- Используйте Cloudflare для CDN и защиты

## 📚 Дополнительные ресурсы

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [Vector Search Best Practices](https://qdrant.tech/articles/vector-search-best-practices/)
- [Cloudflare API](https://developers.cloudflare.com/api/)
