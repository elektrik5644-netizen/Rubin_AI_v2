# 🔄 Поток данных в Rubin AI v2.0

## 📊 Общая схема работы системы

```
ПОЛЬЗОВАТЕЛЬ
    │
    ▼
🌐 Веб-интерфейс (RubinIDE.html)
    │
    ▼
📡 HTTP POST → /api/chat
    │
    ▼
🧠 AI Чат сервер (8084)
    │
    ▼
🔍 Анализ запроса → Категоризация
    │
    ▼
📋 Маршрутизация к специализированным модулям
    │
    ▼
🗄️ Поиск в базе данных
    │
    ▼
🤖 Генерация ответа
    │
    ▼
📤 HTTP Response ← JSON
    │
    ▼
🌐 Веб-интерфейс
    │
    ▼
👤 ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА
```

## 🔧 Детальный поток данных

### 1. **Инициация запроса (Frontend)**

#### **Пользовательский ввод:**
```javascript
// В RubinIDE.html
function sendMessage() {
    const message = document.getElementById('messageInput').value;
    
    // Формирование JSON пакета
    const requestData = {
        message: message,
        timestamp: new Date().toISOString(),
        session_id: getSessionId()
    };
    
    // Отправка HTTP POST запроса
    fetch('http://localhost:8084/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => displayResponse(data));
}
```

### 2. **Обработка на сервере (Backend)**

#### **Получение HTTP запроса:**
```python
# В api/rubin_ai_v2_simple.py
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Получение JSON данных
        data = request.get_json()
        message = data.get('message', '')
        
        logger.info(f"Получен запрос: {message[:100]}...")
        
        # Анализ и категоризация
        category = analyze_message(message)
        logger.info(f"Категория: {category}")
        
        # Генерация ответа
        response = generate_response(message, category)
        
        # Формирование JSON ответа
        return jsonify({
            'response': response['response'],
            'category': category,
            'provider': response['provider'],
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка в AI чате: {e}")
        return jsonify({'error': str(e)}), 500
```

### 3. **Анализ и категоризация**

#### **Определение типа запроса:**
```python
def analyze_message(message):
    """Анализ сообщения и определение категории"""
    message_lower = message.lower()
    
    # Ключевые слова для категоризации
    categories = {
        'controllers': ['пид', 'регулятор', 'plc', 'контроллер', 'автоматизация'],
        'electrical': ['электричество', 'схема', 'ток', 'напряжение', 'резистор'],
        'radiomechanics': ['антенна', 'сигнал', 'радио', 'модуляция', 'частота'],
        'programming': ['программа', 'код', 'python', 'алгоритм', 'функция']
    }
    
    # Подсчет совпадений
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in message_lower)
        scores[category] = score
    
    # Возврат категории с наибольшим score
    if scores:
        return max(scores, key=scores.get)
    return 'general'
```

### 4. **Маршрутизация к специализированным модулям**

#### **Вызов специализированных API:**
```python
def route_to_specialist(message, category):
    """Маршрутизация к специализированным модулям"""
    
    specialist_apis = {
        'controllers': 'http://localhost:8090/api/controllers/explain',
        'electrical': 'http://localhost:8087/api/electrical/explain',
        'radiomechanics': 'http://localhost:8089/api/radiomechanics/explain'
    }
    
    if category in specialist_apis:
        try:
            # HTTP запрос к специализированному API
            response = requests.post(
                specialist_apis[category],
                json={'query': message},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Ошибка обращения к {category}: {e}")
    
    return None
```

### 5. **Поиск в базе данных**

#### **Текстовый поиск:**
```python
def search_documents(query, limit=10):
    """Поиск документов в базе данных"""
    try:
        conn = sqlite3.connect('rubin_ai_v2.db')
        cursor = conn.cursor()
        
        # Поиск по ключевым словам
        query_words = query.lower().split()
        placeholders = ','.join(['?' for _ in query_words])
        
        cursor.execute(f'''
            SELECT DISTINCT d.id, d.file_name, d.category, d.metadata, 
                   GROUP_CONCAT(DISTINCT di.keyword) as matched_keywords
            FROM documents d
            JOIN document_index di ON d.id = di.document_id
            WHERE di.keyword IN ({placeholders})
            GROUP BY d.id
            ORDER BY COUNT(DISTINCT di.keyword) DESC
            LIMIT ?
        ''', query_words + [limit])
        
        results = cursor.fetchall()
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return []
```

#### **Векторный поиск (новый):**
```python
def vector_search(query, top_k=5):
    """Семантический поиск в векторном пространстве"""
    try:
        # Генерация embedding для запроса
        query_embedding = model.encode(query)
        
        # Поиск похожих векторов
        similarities, indices = faiss_index.search(query_embedding, top_k)
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= threshold:
                results.append({
                    'document_id': document_ids[idx],
                    'similarity': similarity,
                    'content': get_document_content(document_ids[idx])
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка векторного поиска: {e}")
        return []
```

### 6. **Генерация ответа**

#### **Специализированные ответы:**
```python
def get_specialist_response(category, message):
    """Получение ответа от специализированного модуля"""
    
    responses = {
        'controllers': {
            'response': '''ПИД-регулятор (Пропорционально-Интегрально-Дифференциальный) - это устройство автоматического управления.

**Принцип работы:**
- **P (Пропорциональная)** - реагирует на текущую ошибку
- **I (Интегральная)** - учитывает накопленную ошибку
- **D (Дифференциальная)** - предсказывает будущую ошибку

**Формула ПИД-регулятора:**
u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt

Где:
- Kp, Ki, Kd - коэффициенты настройки
- e(t) - ошибка регулирования
- u(t) - управляющий сигнал''',
            'provider': 'PLC Specialist'
        },
        'electrical': {
            'response': '''Электротехника изучает электрические явления и их практическое применение.

**Основные законы:**
- **Закон Ома:** U = I × R
- **Первый закон Кирхгофа:** ΣI = 0
- **Второй закон Кирхгофа:** ΣU = 0

**Компоненты:**
- Резисторы, конденсаторы, катушки индуктивности
- Транзисторы, диоды, операционные усилители
- Источники питания, измерительные приборы''',
            'provider': 'Electrical Specialist'
        }
    }
    
    return responses.get(category, get_general_response(message))
```

### 7. **Формирование HTTP ответа**

#### **JSON структура ответа:**
```python
def format_response(response_data, category, provider):
    """Форматирование ответа для отправки клиенту"""
    
    return {
        'response': response_data['response'],
        'category': category,
        'provider': provider,
        'timestamp': datetime.now().isoformat(),
        'search_results': response_data.get('search_results', []),
        'metadata': {
            'processing_time': response_data.get('processing_time', 0),
            'search_type': response_data.get('search_type', 'text'),
            'confidence': response_data.get('confidence', 0.8)
        },
        'status': 'success'
    }
```

## 🔄 Схема передачи пакетов

### **HTTP Request (Frontend → Backend):**
```http
POST /api/chat HTTP/1.1
Host: localhost:8084
Content-Type: application/json
Content-Length: 156

{
    "message": "Как работает ПИД-регулятор?",
    "timestamp": "2025-09-14T22:30:00.000Z",
    "session_id": "sess_12345"
}
```

### **HTTP Response (Backend → Frontend):**
```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 1024

{
    "response": "ПИД-регулятор (Пропорционально-Интегрально-Дифференциальный)...",
    "category": "controllers",
    "provider": "PLC Specialist",
    "timestamp": "2025-09-14T22:30:01.500Z",
    "search_results": [
        {
            "document_id": 1,
            "file_name": "PID_controller_manual.pdf",
            "similarity": 0.95
        }
    ],
    "metadata": {
        "processing_time": 1.2,
        "search_type": "hybrid",
        "confidence": 0.92
    },
    "status": "success"
}
```

## 🗄️ Работа с базой данных

### **Структура базы данных:**
```sql
-- Таблица документов
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    content TEXT,
    category TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица поискового индекса
CREATE TABLE document_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    keyword TEXT,
    position INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents (id)
);

-- Таблица векторных представлений (новое)
CREATE TABLE document_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    vector_data BLOB NOT NULL,
    vector_hash TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents (id)
);
```

### **Процесс индексации:**
```python
def index_document(file_path):
    """Индексация документа в базе данных"""
    
    # 1. Извлечение содержимого
    content = extract_content(file_path)
    
    # 2. Сохранение в таблицу documents
    cursor.execute('''
        INSERT INTO documents (file_path, file_name, content, category)
        VALUES (?, ?, ?, ?)
    ''', (file_path, filename, content, category))
    
    doc_id = cursor.lastrowid
    
    # 3. Создание текстового индекса
    words = content.lower().split()
    for i, word in enumerate(words):
        if len(word) > 3:
            cursor.execute('''
                INSERT INTO document_index (document_id, keyword, position)
                VALUES (?, ?, ?)
            ''', (doc_id, word, i))
    
    # 4. Создание векторного представления
    embedding = model.encode(content)
    vector_blob = embedding.tobytes()
    
    cursor.execute('''
        INSERT INTO document_vectors (document_id, vector_data, vector_hash)
        VALUES (?, ?, ?)
    ''', (doc_id, vector_blob, hashlib.md5(vector_blob).hexdigest()))
    
    conn.commit()
```

## 🚀 Производительность и оптимизация

### **Кэширование:**
```python
# Кэш для часто запрашиваемых ответов
response_cache = {}

def get_cached_response(message, category):
    """Получение кэшированного ответа"""
    cache_key = hashlib.md5(f"{message}_{category}".encode()).hexdigest()
    return response_cache.get(cache_key)

def cache_response(message, category, response):
    """Сохранение ответа в кэш"""
    cache_key = hashlib.md5(f"{message}_{category}".encode()).hexdigest()
    response_cache[cache_key] = response
```

### **Асинхронная обработка:**
```python
import asyncio
import aiohttp

async def process_request_async(message):
    """Асинхронная обработка запроса"""
    
    # Параллельный поиск в разных источниках
    tasks = [
        search_text_database(message),
        search_vector_database(message),
        get_specialist_response(message)
    ]
    
    results = await asyncio.gather(*tasks)
    return combine_results(results)
```

## 📊 Мониторинг и логирование

### **Метрики производительности:**
```python
def log_request_metrics(message, processing_time, search_type):
    """Логирование метрик запроса"""
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'message_length': len(message),
        'processing_time': processing_time,
        'search_type': search_type,
        'category': analyze_message(message)
    }
    
    logger.info(f"Request metrics: {json.dumps(metrics)}")
```

---

## 🎯 Заключение

**Rubin AI v2.0** использует многоуровневую архитектуру с четким разделением ответственности:

1. **Frontend** - веб-интерфейс для взаимодействия с пользователем
2. **API Gateway** - основной сервер для маршрутизации запросов
3. **Specialist Modules** - специализированные модули для разных областей
4. **Database Layer** - SQLite с текстовым и векторным поиском
5. **Search Engine** - гибридный поиск (текстовый + семантический)

**Поток данных** оптимизирован для быстрой обработки и высокого качества ответов, с поддержкой кэширования и асинхронной обработки.






















