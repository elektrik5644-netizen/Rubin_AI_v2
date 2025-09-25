# RubinIDE.html - Архитектура подключения к базе данных

## 🔗 Схема подключения

```
┌─────────────────────────────────────────────────────────────────┐
│                    RubinIDE.html (Frontend)                    │
├─────────────────────────────────────────────────────────────────┤
│  📝 Редактор кода    │  💬 Чат    │  📁 Файлы    │  🔧 Система  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ HTTP API Calls
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              minimal_rubin_server.py (Backend)                 │
├─────────────────────────────────────────────────────────────────┤
│  🌐 HTTP Server (Port 8083)                                    │
│  📡 API Endpoints                                               │
│  🗄️ Database Operations                                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ Database Queries
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    База данных                                  │
├─────────────────────────────────────────────────────────────────┤
│  📊 SQLite/PostgreSQL/MySQL                                    │
│  📄 Документы и файлы                                          │
│  🏷️ Категории и теги                                           │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ API Endpoints

### 1. **Health Check**
```javascript
GET http://localhost:8083/health
```
- Проверка доступности сервера
- Используется для мониторинга соединения

### 2. **Chat API**
```javascript
POST http://localhost:8083/api/chat
Content-Type: application/json

{
  "message": "текст сообщения"
}
```
- Основной API для общения с AI
- Обработка вопросов и ответов

### 3. **Upload Content**
```javascript
POST http://localhost:8083/api/documents/upload-content
Content-Type: application/json

{
  "filename": "имя_файла.txt",
  "content": "содержимое файла",
  "category": "python_code",
  "tags": ["tag1", "tag2"]
}
```
- Загрузка текстового содержимого в базу данных
- Создание документов с метаданными

### 4. **Upload Files**
```javascript
POST http://localhost:8083/api/chat
Content-Type: multipart/form-data

FormData:
- file: файл
- category: категория
- tags: теги через запятую
```
- Загрузка файлов в базу данных
- Поддержка различных форматов

### 5. **Search Documents**
```javascript
POST http://localhost:8083/api/chat
Content-Type: application/json

{
  "query": "поисковый запрос",
  "limit": 5
}
```
- Поиск документов в базе данных
- Возврат релевантных результатов

### 6. **Database Stats**
```javascript
GET http://localhost:8083/api/documents/stats
GET http://localhost:8083/api/stats
```
- Статистика базы данных
- Информация о системе

## 🔧 Функции подключения в RubinIDE.html

### 1. **Основные функции загрузки**

#### `uploadContentToDatabase(filename, content, category, tags)`
```javascript
async function uploadContentToDatabase(filename, content, category, tags) {
    const response = await fetch('http://localhost:8083/api/documents/upload-content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: filename,
            content: content,
            category: category,
            tags: tags ? tags.split(',').map(tag => tag.trim()).filter(tag => tag) : []
        })
    });
}
```

#### `uploadFilesToDatabase(files)`
```javascript
async function uploadFilesToDatabase(files) {
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('category', category);
        formData.append('tags', tagList.join(','));
        
        const response = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            body: formData
        });
    }
}
```

### 2. **Функции поиска**

#### `searchDocumentsInDatabase(query)`
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

### 3. **Функции статистики**

#### `showDatabaseStats()`
```javascript
async function showDatabaseStats() {
    const response = await fetch('http://localhost:8083/api/documents/stats', {
        method: 'GET'
    });
}
```

#### `showSystemStats()`
```javascript
async function showSystemStats() {
    const response = await fetch('http://localhost:8083/api/stats', {
        method: 'GET'
    });
}
```

## 📊 Категории документов

```javascript
const categories = [
    'python_code',           // Python код
    'documentation',         // Документация
    'configuration',         // Конфигурационные файлы
    'industrial_automation', // Промышленная автоматизация
    'artificial_intelligence', // Искусственный интеллект
    'api_documentation',     // API документация
    'tutorial',             // Обучающие материалы
    'general'               // Общие документы
];
```

## 🏷️ Система тегов

- **Автоматические теги**: `uploaded`, `files`, `folder`, `batch`
- **Пользовательские теги**: вводимые через запятую
- **Контекстные теги**: `chat`, `message`, `user`, `input`, `text`

## 🔄 Режимы работы

### 1. **Онлайн режим**
- Подключение к серверу `http://localhost:8083`
- Полный доступ к базе данных
- AI ответы через API

### 2. **Офлайн режим**
- Локальные функции
- Ограниченные возможности
- Базовые ответы без AI

## 🚨 Обработка ошибок

### 1. **Повторные попытки**
```javascript
const maxRetries = 3;
let connectionRetries = 0;

if (connectionRetries >= maxRetries) {
    enableOfflineMode();
} else {
    setTimeout(() => {
        sendChatMessage();
    }, 2000);
}
```

### 2. **Автоматическая проверка соединения**
```javascript
setInterval(() => {
    if (!isOfflineMode && connectionRetries < maxRetries) {
        testConnectionQuiet();
    }
}, 30000);
```

## 📁 Поддерживаемые форматы файлов

### Для редактирования:
`.py`, `.js`, `.html`, `.css`, `.json`, `.md`, `.txt`, `.xml`, `.yaml`, `.yml`, `.sql`, `.log`, `.csv`, `.plc`, `.st`, `.iec`, `.cfg`, `.ini`, `.conf`, `.config`, `.pmac`, `.gcode`, `.nc`

### Для загрузки в базу:
`.py`, `.md`, `.json`, `.html`, `.txt`, `.sql`, `.log`, `.yaml`, `.yml`, `.xml`, `.csv`

## 🔐 Безопасность

- **CORS**: Настроен для локального доступа
- **Валидация**: Проверка размера файлов (макс. 10MB)
- **Очистка**: Фильтрация тегов и категорий
- **Таймауты**: Ограничение времени запросов

## 📈 Мониторинг

- **Статус соединения**: Индикатор в интерфейсе
- **Счетчик попыток**: Отслеживание ошибок
- **Статистика**: Количество документов и размер
- **Логи**: Консольные сообщения для отладки
