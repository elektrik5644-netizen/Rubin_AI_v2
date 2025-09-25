# RubinIDE.html - Обработка информации (Краткая схема)

## 🔄 Основной поток обработки

```
ВХОДНАЯ ИНФОРМАЦИЯ
        │
        ▼
sendChatMessage()
        │
        ▼
ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА
        │
        ▼
ВЫБОР РЕЖИМА РАБОТЫ
        │
        ├─── ОНЛАЙН ────► API запросы ────► Серверная обработка
        │
        └─── ОФЛАЙН ────► generateOfflineResponse() ────► Локальные ответы
        │
        ▼
ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА
```

## 🧠 Ключевые функции обработки

### 1. **`sendChatMessage()` - Главная функция**
```javascript
async function sendChatMessage() {
    // 1. Получение и валидация ввода
    const text = input.value.trim();
    if (!text) return;
    
    // 2. Проверка команд
    if (text.includes('загрузи в базу данных')) {
        uploadToDatabase();
        return;
    }
    
    // 3. Выбор режима обработки
    if (isOfflineMode) {
        // Офлайн: локальные ответы
        const response = generateOfflineResponse(text);
        appendChatMessage('ai', response);
    } else {
        // Онлайн: API запросы
        const resp = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            body: JSON.stringify({ message: text })
        });
        const data = await resp.json();
        appendChatMessage('ai', data.response);
    }
}
```

### 2. **`generateOfflineResponse()` - Офлайн обработка**
```javascript
function generateOfflineResponse(message) {
    const lowerMessage = message.toLowerCase();
    
    // Анализ ключевых слов
    if (lowerMessage.includes('привет')) {
        return 'Привет! Я работаю в офлайн режиме.';
    }
    if (lowerMessage.includes('python')) {
        return 'Python - отличный язык программирования!';
    }
    if (lowerMessage.includes('ошибка')) {
        return 'Для решения ошибок:\n1. Проверьте синтаксис\n2. Убедитесь в правильности импортов';
    }
    
    return 'Извините, я работаю в офлайн режиме.';
}
```

### 3. **`uploadContentToDatabase()` - Загрузка файлов**
```javascript
async function uploadContentToDatabase(filename, content, category, tags) {
    const response = await fetch('http://localhost:8083/api/documents/upload-content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: filename,
            content: content,
            category: category,
            tags: tags.split(',').map(tag => tag.trim())
        })
    });
    
    const result = await response.json();
    if (result.status === 'success') {
        appendChatMessage('assistant', '✅ Документ загружен!');
    }
}
```

### 4. **`searchDocumentsInDatabase()` - Поиск**
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
    
    const result = await response.json();
    if (result.status === 'success') {
        const documents = result.data.results;
        // Отображение результатов поиска
    }
}
```

## 🎯 Типы обрабатываемой информации

### **1. Текстовые сообщения**
- **Вопросы** - "Как работает Python?"
- **Команды** - "загрузи в базу данных"
- **Утверждения** - "Python - хороший язык"
- **Запросы помощи** - "помощь", "help"

### **2. Файлы и документы**
- **Код** - Python, PLC, PMAC программы
- **Документация** - текстовые файлы
- **Конфигурации** - настройки систем
- **Обучающие материалы** - туториалы

### **3. Голосовой ввод**
- **Speech Recognition** - распознавание речи
- **Автоматическая отправка** - после распознавания
- **Поддержка русского языка** - ru-RU

### **4. TTS озвучивание**
- **Автоматическое озвучивание** - ответов AI
- **Настройки голоса** - скорость, высота
- **Очистка текста** - удаление markdown

## 🔄 Режимы работы

### **Онлайн режим**
```javascript
// API запросы к серверу
const resp = await fetch('http://localhost:8083/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message: text })
});
```

### **Офлайн режим**
```javascript
// Локальные ответы
const response = generateOfflineResponse(text);
appendChatMessage('ai', response);
```

## 🚨 Обработка ошибок

### **Повторные попытки**
```javascript
const maxRetries = 3;
if (connectionRetries >= maxRetries) {
    enableOfflineMode();
} else {
    setTimeout(() => sendChatMessage(), 2000);
}
```

### **Автоматическое переключение**
```javascript
function enableOfflineMode() {
    isOfflineMode = true;
    updateConnectionStatus('offline');
    // Показ уведомления об офлайн режиме
}
```

## 📊 Категории документов

| Категория | Описание |
|-----------|----------|
| `python_code` | Python код |
| `documentation` | Документация |
| `configuration` | Конфигурационные файлы |
| `industrial_automation` | Промышленная автоматизация |
| `artificial_intelligence` | Искусственный интеллект |
| `api_documentation` | API документация |
| `tutorial` | Обучающие материалы |
| `general` | Общие документы |

## 🎨 Визуализация

### **Отображение сообщений**
```javascript
function appendChatMessage(type, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
}
```

### **Индикаторы состояния**
- **Статус соединения** - подключено/подключение/офлайн
- **Режим работы** - онлайн/офлайн
- **TTS статус** - включено/выключено
- **Счетчик попыток** - количество ошибок

## 🔧 Специальные функции

### **Голосовой ввод**
```javascript
function toggleVoiceInput() {
    recognition = new SpeechRecognition();
    recognition.lang = 'ru-RU';
    recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        input.value = transcript;
        setTimeout(sendChatMessage, 50);
    };
    recognition.start();
}
```

### **TTS озвучивание**
```javascript
function speak(text) {
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = 'ru-RU';
    window.speechSynthesis.speak(utter);
}
```

### **Проверка соединения**
```javascript
async function testConnection() {
    const response = await fetch('http://localhost:8083/health');
    if (response.ok) {
        updateConnectionStatus('connected');
    } else {
        updateConnectionStatus('offline');
    }
}
```

## 📈 Мониторинг

### **Статистика**
- Количество обработанных сообщений
- Статус соединения
- Режим работы
- Количество ошибок

### **Логирование**
- Ошибки подключения
- Успешные запросы
- Переключения режимов
- Пользовательские действия

## 🚀 Оптимизация

### **Асинхронность**
- Все API запросы асинхронные
- Неблокирующий интерфейс
- Параллельная обработка

### **Кэширование**
- Сохранение состояния
- Кэширование ответов
- Оптимизация запросов

### **Таймауты**
- Ограничение времени запросов
- Автоматические повторы
- Graceful degradation
