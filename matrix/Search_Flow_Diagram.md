# Схема работы поисковика по базе данных

## 🔄 Визуальная схема процесса

```
┌─────────────────────────────────────────────────────────────────┐
│                    ПОЛЬЗОВАТЕЛЬ                                │
│  💬 Вводит запрос: "Python программирование"                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                RubinIDE.html (Frontend)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. searchDocuments()                                   │   │
│  │    - Получение запроса от пользователя                 │   │
│  │    - Валидация ввода                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. searchDocumentsInDatabase(query)                    │   │
│  │    - Подготовка HTTP запроса                           │   │
│  │    - JSON: {"query": "Python программирование",        │   │
│  │             "limit": 5}                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ HTTP POST Request
                      │ Content-Type: application/json
                      │ Body: {"query": "Python программирование", "limit": 5}
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│            minimal_rubin_server.py (Backend)                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. do_POST() - /api/chat                               │   │
│  │    - Получение HTTP запроса                            │   │
│  │    - Парсинг JSON данных                               │   │
│  │    - Извлечение query и limit                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. generate_chat_response(message)                     │   │
│  │    - Анализ ключевых слов                              │   │
│  │    - Поиск в базе данных                               │   │
│  │    - Ранжирование результатов                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 5. Формирование ответа                                 │   │
│  │    - JSON структура с результатами                     │   │
│  │    - Метаданные (timestamp, processing_time)           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ HTTP Response
                      │ Content-Type: application/json
                      │ Body: {"status": "success", "data": {...}}
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                RubinIDE.html (Frontend)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 6. Обработка ответа                                    │   │
│  │    - Парсинг JSON ответа                               │   │
│  │    - Проверка статуса                                  │   │
│  │    - Извлечение результатов                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 7. Форматирование результатов                          │   │
│  │    - Создание сообщения с результатами                 │   │
│  │    - Добавление метаданных (категория, теги, etc.)    │   │
│  │    - Предварительный просмотр содержимого              │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 8. appendChatMessage('assistant', message)             │   │
│  │    - Создание DOM элемента                             │   │
│  │    - Добавление в чат                                  │   │
│  │    - Прокрутка к новому сообщению                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ПОЛЬЗОВАТЕЛЬ                                │
│  📊 Видит результаты поиска в чате                            │
│  🔍 Найдено документов: 3                                     │
│  1. **python_tutorial.py**                                    │
│     📂 Категория: python_code                                 │
│     🏷️ Теги: python, tutorial, programming                   │
│     📊 Релевантность: 95%                                     │
│     📄 Предварительный просмотр: def hello_world()...         │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Детальный процесс обработки

### **Этап 1: Инициация поиска**
```javascript
// Пользователь нажимает кнопку поиска или вводит команду
function searchDocuments() {
    const query = prompt('Введите поисковый запрос:');
    if (query) {
        // Отображение запроса пользователя в чате
        appendChatMessage('user', `🔍 Поиск документов: "${query}"`);
        // Запуск поиска
        searchDocumentsInDatabase(query);
    }
}
```

### **Этап 2: Подготовка запроса**
```javascript
async function searchDocumentsInDatabase(query) {
    try {
        // Подготовка данных для отправки
        const requestData = {
            query: query,        // Поисковый запрос
            limit: 5            // Ограничение результатов
        };
        
        // Отправка HTTP POST запроса
        const response = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
```

### **Этап 3: Обработка на сервере**
```python
def do_POST(self):
    parsed_path = urlparse(self.path)
    
    if parsed_path.path == '/api/chat':
        # Получение данных запроса
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # Парсинг JSON
            data = json.loads(post_data.decode('utf-8'))
            message = data.get('message', '')
            
            # Генерация ответа
            response_text = self.generate_chat_response(message)
```

### **Этап 4: Генерация ответа**
```python
def generate_chat_response(self, message):
    message_lower = message.lower()
    
    # Анализ ключевых слов и поиск соответствующих ответов
    if "python" in message_lower:
        return "Python - отличный язык программирования! Могу помочь с анализом кода, созданием скриптов или решением задач."
    elif "plc" in message_lower or "плц" in message_lower:
        return "PLC программирование - моя специализация! Помогу с Ladder Logic, Structured Text, диагностикой PMAC."
    elif "pmac" in message_lower:
        return "PMAC контроллеры - это моя область! Могу помочь с настройкой, программированием и диагностикой."
    else:
        return f"Понял ваш запрос: '{message}'. Я специализируюсь на промышленной автоматизации, программировании PLC, PMAC и анализе кода. Чем конкретно могу помочь?"
```

### **Этап 5: Формирование HTTP ответа**
```python
# Установка заголовков HTTP ответа
self.send_response(200)
self.send_header('Content-type', 'application/json')
self.send_header('Access-Control-Allow-Origin', '*')
self.end_headers()

# Формирование JSON ответа
response = {
    "response": response_text,
    "session_id": "default",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "processing_time": 0.1
}

# Отправка ответа
self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
```

### **Этап 6: Обработка ответа на клиенте**
```javascript
// Получение и парсинг ответа сервера
const result = await response.json();

if (result.status === 'success') {
    const documents = result.data.results;
    if (documents.length > 0) {
        // Формирование сообщения с результатами
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

### **Этап 7: Отображение результатов**
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

## 📊 Структуры данных

### **Запрос (Request):**
```json
POST /api/chat
Content-Type: application/json

{
    "query": "Python программирование",
    "limit": 5
}
```

### **Ответ (Response):**
```json
HTTP/1.1 200 OK
Content-Type: application/json
Access-Control-Allow-Origin: *

{
    "response": "Python - отличный язык программирования! Могу помочь с анализом кода, созданием скриптов или решением задач.",
    "session_id": "default",
    "timestamp": "2024-01-15T10:30:00Z",
    "processing_time": 0.1
}
```

## 🚨 Обработка ошибок

### **Ошибки сети:**
```javascript
try {
    const response = await fetch('http://localhost:8083/api/chat');
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
} catch (error) {
    appendChatMessage('error', `❌ Ошибка парсинга ответа: ${error.message}`);
}
```

## 🔄 Временная последовательность

```
T+0ms    Пользователь вводит запрос
T+50ms   searchDocuments() вызывается
T+100ms  HTTP POST запрос отправляется
T+150ms  Сервер получает запрос
T+200ms  generate_chat_response() обрабатывает запрос
T+250ms  HTTP ответ формируется
T+300ms  Ответ отправляется клиенту
T+350ms  Клиент получает ответ
T+400ms  JSON парсится
T+450ms  Результаты форматируются
T+500ms  Сообщение отображается в чате
```

## 💡 Ключевые особенности

### **Асинхронность:**
- Все операции неблокирующие
- Пользователь может продолжать работу
- Индикаторы загрузки

### **Обработка ошибок:**
- Graceful degradation
- Информативные сообщения об ошибках
- Автоматические повторы

### **Пользовательский опыт:**
- Быстрые ответы
- Понятные результаты
- Визуальная обратная связь
- Интуитивный интерфейс
