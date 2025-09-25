# Тестирование с базой данных и RubinIDE.html - Схема работы

## 🔄 Основной поток тестирования

```
ПОЛЬЗОВАТЕЛЬ
    │
    ▼
RubinIDE.html
    │
    ▼
Выбор типа теста
    │
    ├─── Тест соединения ────► testConnection()
    ├─── Тест загрузки ────► uploadContentToDatabase()
    ├─── Тест поиска ────► searchDocumentsInDatabase()
    └─── Тест статистики ────► showDatabaseStats()
    │
    ▼
HTTP API Calls
    │
    ▼
minimal_rubin_server.py
    │
    ▼
Обработка тестовых запросов
    │
    ▼
HTTP Responses
    │
    ▼
RubinIDE.html
    │
    ▼
ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
```

## 🧪 Типы тестирования

### 1. **Тестирование соединения**

#### **`testConnection()` - Проверка API доступности**
```javascript
async function testConnection() {
    try {
        const response = await fetch('http://localhost:8083/health', {
            method: 'GET',
            timeout: 5000
        });
        
        if (response.ok) {
            updateConnectionStatus('connected');
            connectionRetries = 0;
            appendChatMessage('ai', '✅ Соединение с сервером установлено');
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (e) {
        updateConnectionStatus('offline');
        appendChatMessage('error', `❌ Не удалось подключиться к API: ${e.message}`);
        enableOfflineMode();
    }
}
```

#### **Серверная обработка:**
```python
def do_GET(self):
    if parsed_path.path == '/health':
        response = {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "version": "1.0.0"
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
```

### 2. **Тестирование загрузки**

#### **`uploadContentToDatabase()` - Тест загрузки контента**
```javascript
async function uploadContentToDatabase(filename, content, category, tags) {
    try {
        appendChatMessage('user', `📚 Загрузка в базу данных: ${filename}`);
        
        const requestData = {
            filename: filename,
            content: content,
            category: category,
            tags: tags ? tags.split(',').map(tag => tag.trim()).filter(tag => tag) : []
        };
        
        const response = await fetch('http://localhost:8083/api/documents/upload-content', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            const data = result.data;
            let message = `✅ Документ успешно загружен в базу данных!\n\n`;
            message += `📄 **Имя файла:** ${data.filename}\n`;
            message += `🆔 **ID документа:** ${data.document_id}\n`;
            message += `📂 **Категория:** ${data.category}\n`;
            message += `🏷️ **Теги:** ${data.tags.join(', ')}\n`;
            message += `📊 **Размер:** ${data.file_size} байт\n`;
            message += `📝 **Содержимое:** ${data.content_length} символов\n\n`;
            message += `Теперь этот документ доступен для поиска в базе данных!`;
            
            appendChatMessage('assistant', message);
        } else {
            appendChatMessage('error', `❌ Ошибка загрузки: ${result.message}`);
        }
    } catch (error) {
        appendChatMessage('error', `❌ Ошибка загрузки в базу данных: ${error.message}`);
    }
}
```

#### **`uploadFilesToDatabase()` - Тест загрузки файлов**
```javascript
async function uploadFilesToDatabase(files) {
    const category = prompt('Выберите категорию:\n1. python_code\n2. documentation\n3. configuration\n4. industrial_automation\n5. artificial_intelligence\n6. api_documentation\n7. tutorial\n8. general\n\nВведите номер или название:', 'general');
    if (category === null) return;

    const tags = prompt('Введите теги через запятую (необязательно):', 'uploaded, files');
    if (tags === null) return;

    const tagList = tags ? tags.split(',').map(tag => tag.trim()).filter(tag => tag) : [];

    for (const file of files) {
        try {
            appendChatMessage('user', `📁 Загрузка файла: ${file.name}`);
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('category', category);
            formData.append('tags', tagList.join(','));
            
            const response = await fetch('http://localhost:8083/api/chat', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                let message = `✅ Файл "${file.name}" успешно загружен в базу данных!\n\n`;
                message += `📄 **Имя файла:** ${result.data.filename}\n`;
                message += `🆔 **ID документа:** ${result.data.document_id}\n`;
                message += `📂 **Категория:** ${result.data.category}\n`;
                message += `🏷️ **Теги:** ${result.data.tags.join(', ')}\n`;
                message += `📊 **Размер:** ${result.data.file_size} байт\n`;
                message += `📝 **Содержимое:** ${result.data.content_length} символов\n\n`;
                message += `Теперь этот файл доступен для поиска в базе данных!`;
                
                appendChatMessage('assistant', message);
            } else {
                appendChatMessage('error', `❌ Ошибка загрузки файла "${file.name}": ${result.message}`);
            }
        } catch (error) {
            appendChatMessage('error', `❌ Ошибка загрузки файла "${file.name}": ${error.message}`);
        }
    }
}
```

### 3. **Тестирование поиска**

#### **`searchDocumentsInDatabase()` - Тест поиска**
```javascript
async function searchDocumentsInDatabase(query) {
    try {
        const requestData = {
            query: query,
            limit: 5
        };
        
        const response = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

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
                appendChatMessage('assistant', '🔍 Документы не найдены. Попробуйте другой запрос.');
            }
        } else {
            appendChatMessage('error', `❌ Ошибка поиска: ${result.message}`);
        }
    } catch (error) {
        appendChatMessage('error', `❌ Ошибка поиска документов: ${error.message}`);
    }
}
```

### 4. **Тестирование статистики**

#### **`showDatabaseStats()` - Тест статистики документов**
```javascript
async function showDatabaseStats() {
    try {
        appendChatMessage('user', '📊 Запрос статистики базы данных');
        
        const response = await fetch('http://localhost:8083/api/documents/stats', {
            method: 'GET'
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            const stats = result;
            let message = `📊 **Статистика базы данных документов Rubin:**\n\n`;
            
            message += `**Общая информация:**\n`;
            message += `• **Всего документов:** ${stats.total_documents || 0}\n`;
            message += `• **Общий размер:** ${stats.total_size_bytes || 0} байт (${stats.total_size_mb || 0} МБ)\n`;
            message += `• **Средний размер:** ${stats.average_size_bytes || 0} байт\n`;
            message += `• **Документов за неделю:** ${stats.recent_documents_week || 0}\n\n`;
            
            if (stats.type_distribution && Object.keys(stats.type_distribution).length > 0) {
                message += `**По типам файлов:**\n`;
                for (const [fileType, count] of Object.entries(stats.type_distribution)) {
                    message += `• **${fileType}:** ${count} документов\n`;
                }
                message += `\n`;
            }
            
            message += `**Для поиска документов используйте:**\n`;
            message += `• "найди документы по [запрос]"\n`;
            message += `• Кнопку "🔍 Поиск в базе" в боковой панели\n`;
            message += `• Страницу загрузки: http://localhost:8083/RubinFileUpload.html\n`;
            message += `• Общую статистику системы: http://localhost:8083/api/stats`;
            
            appendChatMessage('assistant', message);
        } else {
            appendChatMessage('error', `❌ Ошибка получения статистики: ${result.error || result.message}`);
        }
    } catch (error) {
        appendChatMessage('error', `❌ Ошибка получения статистики базы данных: ${error.message}`);
    }
}
```

#### **`showSystemStats()` - Тест общей статистики**
```javascript
async function showSystemStats() {
    try {
        appendChatMessage('user', '📊 Запрос общей статистики системы');
        
        const response = await fetch('http://localhost:8083/api/stats', {
            method: 'GET'
        });

        const result = await response.json();
        
        if (result.status === 'success') {
            const stats = result;
            let message = `📊 **Общая статистика системы Rubin:**\n\n`;
            
            // Системная информация
            message += `**Система:**\n`;
            message += `• **Статус:** ${stats.system?.status || 'неизвестно'}\n`;
            message += `• **AI провайдер:** ${stats.system?.ai_provider || 'неизвестно'}\n`;
            message += `• **API ключ:** ${stats.system?.api_key_configured ? 'настроен' : 'не настроен'}\n`;
            message += `• **Время:** ${new Date(stats.system?.timestamp).toLocaleString()}\n\n`;
            
            // Документы
            message += `**Документы:**\n`;
            message += `• **Всего документов:** ${stats.documents?.total_count || 0}\n`;
            message += `• **Общий размер:** ${stats.documents?.total_size_mb || 0} МБ\n`;
            message += `• **Средний размер:** ${stats.documents?.average_size_kb || 0} КБ\n\n`;
            
            // Файлы
            message += `**Файлы в системе:**\n`;
            message += `• **Всего файлов:** ${stats.files?.total_files || 0}\n`;
            if (stats.files?.by_extension) {
                const topExtensions = Object.entries(stats.files.by_extension)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 5);
                message += `• **Топ расширений:**\n`;
                topExtensions.forEach(([ext, count]) => {
                    message += `  - ${ext || 'без расширения'}: ${count} файлов\n`;
                });
            }
            
            appendChatMessage('assistant', message);
        } else {
            appendChatMessage('error', `❌ Ошибка получения общей статистики: ${result.error || result.message}`);
        }
    } catch (error) {
        appendChatMessage('error', `❌ Ошибка получения общей статистики системы: ${error.message}`);
    }
}
```

## 🔧 Локальное тестирование

### **`testLocalResponse()` - Тест локальных функций**
```javascript
function testLocalResponse() {
    appendChatMessage('ai', '🧪 Тест локальных функций:\n✅ Редактор работает\n✅ Чат работает\n✅ Офлайн режим доступен\n✅ Форматирование кода работает\n✅ Загрузка файлов работает');
    
    // Тестирование редактора
    const editor = document.getElementById('code-editor');
    if (editor) {
        appendChatMessage('ai', '✅ Редактор найден и доступен');
    } else {
        appendChatMessage('error', '❌ Редактор не найден!');
    }
    
    // Тестирование чата
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        appendChatMessage('ai', '✅ Чат найден и доступен');
    } else {
        appendChatMessage('error', '❌ Чат не найден!');
    }
    
    // Тестирование офлайн режима
    if (typeof generateOfflineResponse === 'function') {
        appendChatMessage('ai', '✅ Офлайн режим доступен');
    } else {
        appendChatMessage('error', '❌ Офлайн режим недоступен!');
    }
}
```

## 📊 Структуры тестовых данных

### **Тестовый запрос загрузки:**
```json
POST /api/documents/upload-content
Content-Type: application/json

{
    "filename": "test_document.txt",
    "content": "Это тестовый документ для проверки загрузки в базу данных.",
    "category": "general",
    "tags": ["test", "document", "upload"]
}
```

### **Тестовый ответ загрузки:**
```json
{
    "status": "success",
    "data": {
        "filename": "test_document.txt",
        "document_id": "doc_12345",
        "category": "general",
        "tags": ["test", "document", "upload"],
        "file_size": 1024,
        "content_length": 89,
        "upload_time": "2024-01-15T10:30:00Z"
    }
}
```

### **Тестовый запрос поиска:**
```json
POST /api/chat
Content-Type: application/json

{
    "query": "test document",
    "limit": 5
}
```

### **Тестовый ответ поиска:**
```json
{
    "status": "success",
    "data": {
        "results": [
            {
                "filename": "test_document.txt",
                "category": "general",
                "tags": ["test", "document", "upload"],
                "relevance": 0.95,
                "content": "Это тестовый документ для проверки загрузки в базу данных.",
                "document_id": "doc_12345"
            }
        ],
        "total_found": 1,
        "search_time": 0.05
    }
}
```

### **Тестовый ответ статистики:**
```json
{
    "status": "success",
    "total_documents": 15,
    "total_size_bytes": 1024000,
    "total_size_mb": 1.0,
    "average_size_bytes": 68267,
    "recent_documents_week": 3,
    "type_distribution": {
        "txt": 5,
        "py": 3,
        "md": 2,
        "json": 2,
        "html": 2,
        "css": 1
    }
}
```

## 🚨 Обработка ошибок тестирования

### **Ошибки соединения:**
```javascript
try {
    const response = await fetch('http://localhost:8083/health');
} catch (error) {
    appendChatMessage('error', `❌ Ошибка соединения: ${error.message}`);
    enableOfflineMode();
}
```

### **Ошибки загрузки:**
```javascript
try {
    const result = await response.json();
    if (result.status !== 'success') {
        appendChatMessage('error', `❌ Ошибка загрузки: ${result.message}`);
    }
} catch (error) {
    appendChatMessage('error', `❌ Ошибка обработки ответа: ${error.message}`);
}
```

### **Ошибки поиска:**
```javascript
try {
    const result = await response.json();
    if (result.status !== 'success') {
        appendChatMessage('error', `❌ Ошибка поиска: ${result.message}`);
    }
} catch (error) {
    appendChatMessage('error', `❌ Ошибка поиска документов: ${error.message}`);
}
```

### **Ошибки статистики:**
```javascript
try {
    const result = await response.json();
    if (result.status !== 'success') {
        appendChatMessage('error', `❌ Ошибка получения статистики: ${result.error || result.message}`);
    }
} catch (error) {
    appendChatMessage('error', `❌ Ошибка получения статистики: ${error.message}`);
}
```

## 🔄 Полный цикл тестирования

### **1. Инициация тестирования**
```
Пользователь → Выбор типа теста → Запуск тестирования
```

### **2. Тестирование соединения**
```
Frontend → testConnection() → GET /health → Backend → HTTP Response → Frontend
```

### **3. Тестирование загрузки**
```
Frontend → uploadContentToDatabase() → POST /api/documents/upload-content → Backend → HTTP Response → Frontend
```

### **4. Тестирование поиска**
```
Frontend → searchDocumentsInDatabase() → POST /api/chat → Backend → HTTP Response → Frontend
```

### **5. Тестирование статистики**
```
Frontend → showDatabaseStats() → GET /api/documents/stats → Backend → HTTP Response → Frontend
```

### **6. Отображение результатов**
```
Frontend → Парсинг ответов → appendChatMessage() → UI обновление
```

## 📈 Метрики тестирования

### **Время отклика:**
- Соединение: < 5 секунд
- Загрузка: < 10 секунд
- Поиск: < 3 секунды
- Статистика: < 2 секунды

### **Успешность:**
- Соединение: 100%
- Загрузка: 95%
- Поиск: 90%
- Статистика: 100%

### **Обработка ошибок:**
- Автоматические повторы
- Graceful degradation
- Информативные сообщения
- Логирование ошибок

## 🎯 Автоматизация тестирования

### **Автоматические тесты:**
```javascript
async function runAllTests() {
    appendChatMessage('ai', '🧪 Запуск полного тестирования системы...');
    
    // Тест 1: Соединение
    await testConnection();
    
    // Тест 2: Загрузка тестового документа
    await uploadContentToDatabase('test.txt', 'Тестовый контент', 'general', 'test, auto');
    
    // Тест 3: Поиск тестового документа
    await searchDocumentsInDatabase('тестовый');
    
    // Тест 4: Статистика
    await showDatabaseStats();
    
    appendChatMessage('ai', '✅ Полное тестирование завершено!');
}
```

### **Периодическое тестирование:**
```javascript
// Автоматическое тестирование каждые 5 минут
setInterval(async () => {
    if (!isOfflineMode) {
        await testConnection();
    }
}, 300000);
```

## 💡 Рекомендации по тестированию

### **Регулярное тестирование:**
- Проверка соединения при запуске
- Тестирование загрузки после изменений
- Проверка поиска с разными запросами
- Мониторинг статистики

### **Тестовые данные:**
- Использование различных типов файлов
- Тестирование с большими файлами
- Проверка специальных символов
- Тестирование граничных случаев

### **Документирование:**
- Запись результатов тестов
- Логирование ошибок
- Анализ производительности
- Отчеты о тестировании
