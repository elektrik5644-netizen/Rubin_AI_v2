# Поисковик по базе данных - Архитектура работы

## 🔄 Общая схема работы поисковика

```
┌─────────────────────────────────────────────────────────────────┐
│                    RubinIDE.html (Frontend)                    │
├─────────────────────────────────────────────────────────────────┤
│  🔍 Поисковый запрос  │  📝 Ввод пользователя  │  🎯 Команды   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ HTTP POST Request
                      │ Content-Type: application/json
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              minimal_rubin_server.py (Backend)                 │
├─────────────────────────────────────────────────────────────────┤
│  📡 API Endpoint: /api/chat                                    │
│  🔍 Обработка запроса                                          │
│  🧠 Генерация ответа                                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ HTTP Response
                      │ Content-Type: application/json
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    RubinIDE.html (Frontend)                    │
├─────────────────────────────────────────────────────────────────┤
│  📊 Обработка ответа  │  💬 Отображение результатов  │  🎨 UI   │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Детальный процесс работы

### 1. **Инициация поиска (Frontend)**

#### **Пользовательский ввод:**
```javascript
// Функция поиска документов
function searchDocuments() {
    const query = prompt('Введите поисковый запрос:');
    if (query) {
        appendChatMessage('user', `🔍 Поиск документов: "${query}"`);
        searchDocumentsInDatabase(query);
    }
}
```

#### **Подготовка запроса:**
```javascript
async function searchDocumentsInDatabase(query) {
    try {
        // 1. ПОДГОТОВКА ДАННЫХ ЗАПРОСА
        const requestData = {
            query: query,        // Поисковый запрос
            limit: 5            // Ограничение количества результатов
        };
        
        // 2. ОТПРАВКА HTTP ЗАПРОСА
        const response = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
```

### 2. **Обработка на сервере (Backend)**

#### **Получение запроса:**
```python
def do_POST(self):
    parsed_path = urlparse(self.path)
    
    if parsed_path.path == '/api/chat':
        # 1. ЧТЕНИЕ ДАННЫХ ЗАПРОСА
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # 2. ПАРСИНГ JSON
            data = json.loads(post_data.decode('utf-8'))
            message = data.get('message', '')
            
            # 3. ОБРАБОТКА ЗАПРОСА
            response_text = self.generate_chat_response(message)
```

#### **Генерация ответа:**
```python
def generate_chat_response(self, message):
    """Генерация ответа чата"""
    message_lower = message.lower()
    
    # АНАЛИЗ КЛЮЧЕВЫХ СЛОВ И ГЕНЕРАЦИЯ ОТВЕТОВ
    if "привет" in message_lower or "hello" in message_lower:
        return "Привет! Я Rubin AI. Готов помочь с программированием и промышленной автоматизацией!"
    elif "python" in message_lower:
        return "Python - отличный язык программирования! Могу помочь с анализом кода, созданием скриптов или решением задач."
    elif "plc" in message_lower or "плц" in message_lower:
        return "PLC программирование - моя специализация! Помогу с Ladder Logic, Structured Text, диагностикой PMAC."
    elif "pmac" in message_lower:
        return "PMAC контроллеры - это моя область! Могу помочь с настройкой, программированием и диагностикой."
    else:
        return f"Понял ваш запрос: '{message}'. Я специализируюсь на промышленной автоматизации, программировании PLC, PMAC и анализе кода. Чем конкретно могу помочь?"
```

#### **Формирование ответа:**
```python
# 4. ФОРМИРОВАНИЕ HTTP ОТВЕТА
self.send_response(200)
self.send_header('Content-type', 'application/json')
self.send_header('Access-Control-Allow-Origin', '*')
self.end_headers()

response = {
    "response": response_text,
    "session_id": "default",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "processing_time": 0.1
}
self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
```

### 3. **Обработка ответа (Frontend)**

#### **Получение и парсинг ответа:**
```javascript
// 3. ОБРАБОТКА ОТВЕТА СЕРВЕРА
const result = await response.json();

if (result.status === 'success') {
    const documents = result.data.results;
    if (documents.length > 0) {
        // 4. ФОРМИРОВАНИЕ СООБЩЕНИЯ С РЕЗУЛЬТАТАМИ
        let message = `🔍 Найдено документов: ${documents.length}\n\n`;
        documents.forEach((doc, index) => {
            message += `${index + 1}. **${doc.filename}**\n`;
            message += `   📂 Категория: ${doc.category}\n`;
            message += `   🏷️ Теги: ${doc.tags.join(', ')}\n`;
            message += `   📊 Релевантность: ${doc.relevance}\n`;
            message += `   📄 Предварительный просмотр: ${doc.content.substring(0, 100)}...\n\n`;
        });
        appendChatMessage('assistant', message);
    } else {
        appendChatMessage('assistant', '🔍 Документы не найдены. Попробуйте другой запрос.');
    }
} else {
    appendChatMessage('error', `❌ Ошибка поиска: ${result.message}`);
}
```

#### **Отображение результатов:**
```javascript
function appendChatMessage(type, message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
```

## 🔍 Типы поисковых запросов

### 1. **Прямые поисковые запросы**
```javascript
// Пользователь вводит поисковый запрос
const query = "Python программирование";
searchDocumentsInDatabase(query);
```

### 2. **Команды в чате**
```javascript
// Автоматическое распознавание команд поиска
if (text.toLowerCase().includes('найди') || 
    text.toLowerCase().includes('поиск') ||
    text.toLowerCase().includes('ищи')) {
    // Извлечение поискового запроса из сообщения
    const searchQuery = extractSearchQuery(text);
    searchDocumentsInDatabase(searchQuery);
}
```

### 3. **Контекстные запросы**
```javascript
// Поиск на основе контекста разговора
if (message.includes('документы') || message.includes('файлы')) {
    // Поиск связанных документов
    searchDocumentsInDatabase(message);
}
```

## 📊 Структура данных

### **Запрос (Request):**
```json
{
    "query": "Python программирование",
    "limit": 5
}
```

### **Ответ (Response):**
```json
{
    "status": "success",
    "data": {
        "results": [
            {
                "filename": "python_tutorial.py",
                "category": "python_code",
                "tags": ["python", "tutorial", "programming"],
                "relevance": 0.95,
                "content": "def hello_world():\n    print('Hello, World!')",
                "document_id": "doc_123",
                "file_size": 1024,
                "content_length": 50
            }
        ],
        "total_found": 1,
        "search_time": 0.05
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🎯 Алгоритм поиска

### 1. **Анализ запроса**
```python
def analyze_search_query(query):
    """Анализ поискового запроса"""
    query_lower = query.lower()
    
    # Извлечение ключевых слов
    keywords = extract_keywords(query_lower)
    
    # Определение типа поиска
    search_type = determine_search_type(query_lower)
    
    # Определение приоритета
    priority = calculate_priority(keywords)
    
    return {
        "keywords": keywords,
        "search_type": search_type,
        "priority": priority
    }
```

### 2. **Поиск в базе данных**
```python
def search_in_database(query, limit=5):
    """Поиск документов в базе данных"""
    # 1. ПОДГОТОВКА ЗАПРОСА
    search_terms = prepare_search_terms(query)
    
    # 2. ВЫПОЛНЕНИЕ ПОИСКА
    results = database.search(
        terms=search_terms,
        limit=limit,
        fields=['content', 'filename', 'tags', 'category']
    )
    
    # 3. РАНЖИРОВАНИЕ РЕЗУЛЬТАТОВ
    ranked_results = rank_results(results, query)
    
    # 4. ФОРМАТИРОВАНИЕ ОТВЕТА
    formatted_results = format_search_results(ranked_results)
    
    return formatted_results
```

### 3. **Ранжирование результатов**
```python
def rank_results(results, query):
    """Ранжирование результатов поиска"""
    query_terms = query.lower().split()
    
    for result in results:
        score = 0
        
        # Поиск в названии файла
        if any(term in result['filename'].lower() for term in query_terms):
            score += 10
        
        # Поиск в содержимом
        content_matches = sum(1 for term in query_terms if term in result['content'].lower())
        score += content_matches * 5
        
        # Поиск в тегах
        tag_matches = sum(1 for term in query_terms if term in ' '.join(result['tags']).lower())
        score += tag_matches * 8
        
        # Поиск в категории
        if any(term in result['category'].lower() for term in query_terms):
            score += 6
        
        result['relevance'] = min(score / 20, 1.0)  # Нормализация до 0-1
    
    # Сортировка по релевантности
    return sorted(results, key=lambda x: x['relevance'], reverse=True)
```

## 🔄 Полный цикл работы

### **1. Инициация поиска**
```
Пользователь → Ввод запроса → searchDocuments() → searchDocumentsInDatabase()
```

### **2. Отправка запроса**
```
Frontend → HTTP POST → /api/chat → Backend
```

### **3. Обработка на сервере**
```
Backend → Парсинг JSON → generate_chat_response() → Анализ запроса → Поиск в БД
```

### **4. Формирование ответа**
```
Backend → Ранжирование → Форматирование → JSON Response → Frontend
```

### **5. Отображение результатов**
```
Frontend → Парсинг ответа → appendChatMessage() → UI обновление
```

## 🚨 Обработка ошибок

### **Ошибки сети:**
```javascript
try {
    const response = await fetch('http://localhost:8083/api/chat', {
        method: 'POST',
        body: JSON.stringify(requestData)
    });
} catch (error) {
    appendChatMessage('error', `❌ Ошибка поиска документов: ${error.message}`);
}
```

### **Ошибки сервера:**
```python
try:
    response_text = self.generate_chat_response(message)
except Exception as e:
    self.send_response(500)
    response = {"error": f"Chat processing failed: {str(e)}"}
    self.wfile.write(json.dumps(response).encode('utf-8'))
```

### **Ошибки парсинга:**
```javascript
try {
    const result = await response.json();
    if (result.status === 'success') {
        // Обработка успешного ответа
    } else {
        appendChatMessage('error', `❌ Ошибка поиска: ${result.message}`);
    }
} catch (error) {
    appendChatMessage('error', `❌ Ошибка парсинга ответа: ${error.message}`);
}
```

## 📈 Оптимизация поиска

### **1. Кэширование запросов**
```python
# Кэширование частых запросов
search_cache = {}

def cached_search(query):
    if query in search_cache:
        return search_cache[query]
    
    results = search_in_database(query)
    search_cache[query] = results
    return results
```

### **2. Индексация**
```python
# Создание индекса для быстрого поиска
def create_search_index():
    index = {
        'keywords': {},
        'categories': {},
        'tags': {}
    }
    
    for doc in database.get_all_documents():
        # Индексация по ключевым словам
        for word in doc['content'].split():
            if word not in index['keywords']:
                index['keywords'][word] = []
            index['keywords'][word].append(doc['id'])
    
    return index
```

### **3. Асинхронная обработка**
```javascript
// Асинхронный поиск с индикатором загрузки
async function searchDocumentsInDatabase(query) {
    // Показ индикатора загрузки
    showLoadingIndicator();
    
    try {
        const response = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            body: JSON.stringify({ query: query, limit: 5 })
        });
        
        const result = await response.json();
        // Обработка результата
    } finally {
        // Скрытие индикатора загрузки
        hideLoadingIndicator();
    }
}
```

## 🎨 Пользовательский интерфейс

### **Индикаторы состояния:**
- **Загрузка** - спиннер во время поиска
- **Результаты** - количество найденных документов
- **Ошибки** - красные сообщения об ошибках
- **Пустые результаты** - сообщение о том, что ничего не найдено

### **Форматирование результатов:**
- **Название файла** - жирным шрифтом
- **Категория** - с иконкой папки
- **Теги** - цветными метками
- **Релевантность** - процентным показателем
- **Предварительный просмотр** - первые 100 символов

## 🔧 Настройки поиска

### **Параметры запроса:**
- `limit` - максимальное количество результатов
- `category` - фильтр по категории
- `tags` - фильтр по тегам
- `date_range` - фильтр по дате
- `file_type` - фильтр по типу файла

### **Настройки ранжирования:**
- Вес названия файла: 10
- Вес содержимого: 5
- Вес тегов: 8
- Вес категории: 6
- Максимальная релевантность: 1.0
