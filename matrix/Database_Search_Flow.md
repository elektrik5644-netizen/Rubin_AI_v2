# Поисковик по базе данных - Схема работы

## 🔄 Основной поток работы

```
ПОЛЬЗОВАТЕЛЬ
    │
    ▼
RubinIDE.html
    │
    ▼
HTTP POST → /api/chat
    │
    ▼
minimal_rubin_server.py
    │
    ▼
Обработка запроса
    │
    ▼
Генерация ответа
    │
    ▼
HTTP Response ← JSON
    │
    ▼
RubinIDE.html
    │
    ▼
ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА
```

## 🧠 Детальный процесс

### 1. **Инициация поиска (Frontend)**

#### **Пользовательский ввод:**
```javascript
function searchDocuments() {
    const query = prompt('Введите поисковый запрос:');
    if (query) {
        appendChatMessage('user', `🔍 Поиск документов: "${query}"`);
        searchDocumentsInDatabase(query);
    }
}
```

#### **Отправка запроса:**
```javascript
async function searchDocumentsInDatabase(query) {
    const response = await fetch('http://localhost:8083/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            limit: 5
        })
    });
}
```

### 2. **Обработка на сервере (Backend)**

#### **Получение запроса:**
```python
def do_POST(self):
    if parsed_path.path == '/api/chat':
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        message = data.get('message', '')
        
        response_text = self.generate_chat_response(message)
```

#### **Генерация ответа:**
```python
def generate_chat_response(self, message):
    message_lower = message.lower()
    
    if "python" in message_lower:
        return "Python - отличный язык программирования!"
    elif "plc" in message_lower:
        return "PLC программирование - моя специализация!"
    elif "pmac" in message_lower:
        return "PMAC контроллеры - это моя область!"
    else:
        return f"Понял ваш запрос: '{message}'. Чем могу помочь?"
```

#### **Отправка ответа:**
```python
response = {
    "response": response_text,
    "session_id": "default",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "processing_time": 0.1
}
self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
```

### 3. **Обработка ответа (Frontend)**

#### **Получение и обработка:**
```javascript
const result = await response.json();

if (result.status === 'success') {
    const documents = result.data.results;
    if (documents.length > 0) {
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
        appendChatMessage('assistant', '🔍 Документы не найдены.');
    }
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
                "tags": ["python", "tutorial"],
                "relevance": 0.95,
                "content": "def hello_world():\n    print('Hello, World!')"
            }
        ]
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔍 Типы поисковых запросов

### **1. Прямые запросы**
- Пользователь вводит поисковый запрос
- Автоматический поиск в базе данных

### **2. Команды в чате**
- "найди документы по Python"
- "поиск файлов по PLC"
- "ищи в базе данных"

### **3. Контекстные запросы**
- Поиск на основе контекста разговора
- Автоматическое извлечение ключевых слов

## 🎯 Алгоритм поиска

### **1. Анализ запроса**
- Извлечение ключевых слов
- Определение типа поиска
- Расчет приоритета

### **2. Поиск в базе данных**
- Поиск по содержимому
- Поиск по названию файла
- Поиск по тегам
- Поиск по категории

### **3. Ранжирование результатов**
- Расчет релевантности
- Сортировка по приоритету
- Ограничение количества результатов

## 🚨 Обработка ошибок

### **Ошибки сети:**
```javascript
try {
    const response = await fetch('http://localhost:8083/api/chat');
} catch (error) {
    appendChatMessage('error', `❌ Ошибка поиска: ${error.message}`);
}
```

### **Ошибки сервера:**
```python
try:
    response_text = self.generate_chat_response(message)
except Exception as e:
    response = {"error": f"Chat processing failed: {str(e)}"}
```

### **Ошибки парсинга:**
```javascript
try {
    const result = await response.json();
} catch (error) {
    appendChatMessage('error', `❌ Ошибка парсинга: ${error.message}`);
}
```

## 📈 Оптимизация

### **1. Кэширование**
- Кэширование частых запросов
- Сохранение результатов поиска
- Ускорение повторных запросов

### **2. Индексация**
- Создание индекса для быстрого поиска
- Индексация по ключевым словам
- Индексация по категориям и тегам

### **3. Асинхронность**
- Асинхронная обработка запросов
- Неблокирующий интерфейс
- Индикаторы загрузки

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
- `limit` - максимальное количество результатов (по умолчанию: 5)
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

## 🔄 Полный цикл

### **1. Инициация**
```
Пользователь → Ввод запроса → searchDocuments()
```

### **2. Отправка**
```
Frontend → HTTP POST → /api/chat → Backend
```

### **3. Обработка**
```
Backend → Парсинг → generate_chat_response() → Поиск в БД
```

### **4. Ответ**
```
Backend → JSON Response → Frontend
```

### **5. Отображение**
```
Frontend → Парсинг → appendChatMessage() → UI
```

## 💡 Особенности

### **Автоматическое распознавание:**
- Команды поиска в обычных сообщениях
- Извлечение ключевых слов
- Контекстный анализ

### **Умное ранжирование:**
- Учет релевантности
- Приоритет по типу совпадения
- Фильтрация по качеству

### **Пользовательский опыт:**
- Быстрые ответы
- Понятные результаты
- Обработка ошибок
- Индикаторы состояния
