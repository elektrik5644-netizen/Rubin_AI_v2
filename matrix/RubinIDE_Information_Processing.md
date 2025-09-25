# RubinIDE.html - Обработка информации

## 🔄 Схема обработки информации в RubinIDE.html

```
┌─────────────────────────────────────────────────────────────────┐
│                    ВХОДНАЯ ИНФОРМАЦИЯ                          │
├─────────────────────────────────────────────────────────────────┤
│  💬 Сообщения чата  │  📁 Файлы  │  💻 Код  │  🎤 Голосовой ввод │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА                   │
├─────────────────────────────────────────────────────────────────┤
│  🔍 Валидация  │  🧹 Очистка  │  📊 Нормализация  │  🏷️ Теги    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    АНАЛИЗ И КЛАССИФИКАЦИЯ                      │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Тип сообщения  │  🔑 Ключевые слова  │  📝 Команды         │
│  🌐 Режим работы   │  ⚡ Приоритет       │  🔄 Повторные попытки│
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ОБРАБОТКА ПО РЕЖИМАМ                        │
├─────────────────────────────────────────────────────────────────┤
│  🌐 ОНЛАЙН РЕЖИМ    │  📱 ОФЛАЙН РЕЖИМ                         │
│  ┌─────────────────┐ │  ┌─────────────────┐                    │
│  │ API запросы     │ │  │ Локальные ответы│                    │
│  │ База данных     │ │  │ Шаблонные ответы│                    │
│  │ AI обработка    │ │  │ Ограниченные    │                    │
│  └─────────────────┘ │  │ возможности     │                    │
│                      │  └─────────────────┘                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ГЕНЕРАЦИЯ ОТВЕТА                            │
├─────────────────────────────────────────────────────────────────┤
│  💬 Текстовый ответ  │  🔊 TTS озвучивание  │  📊 Статистика    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ВЫХОДНАЯ ИНФОРМАЦИЯ                         │
├─────────────────────────────────────────────────────────────────┤
│  💬 Ответы в чате  │  📁 Загруженные файлы  │  🔍 Результаты    │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Основные функции обработки

### 1. **Обработка сообщений чата**

#### **`sendChatMessage()` - Главная функция обработки**
```javascript
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    
    if (!text) return;
    
    // 1. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА
    // Проверяем команды загрузки
    if (text.toLowerCase().includes('загрузи в базу данных') || 
        text.toLowerCase().includes('загрузить в базу')) {
        uploadToDatabase();
        input.value = '';
        return;
    }
    
    // 2. ОТОБРАЖЕНИЕ СООБЩЕНИЯ ПОЛЬЗОВАТЕЛЯ
    appendChatMessage('user', text);
    input.value = '';

    // 3. ВЫБОР РЕЖИМА ОБРАБОТКИ
    if (isOfflineMode) {
        // ОФЛАЙН РЕЖИМ - локальные ответы
        const response = generateOfflineResponse(text);
        appendChatMessage('ai', response);
        if (autoTTS) speak(response);
        return;
    }

    // 4. ОНЛАЙН РЕЖИМ - API запросы
    try {
        const resp = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
            timeout: 10000
        });
        
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        
        const data = await resp.json();
        const answer = data.response || '[пустой ответ]';
        appendChatMessage('ai', answer);
        if (autoTTS) speak(answer);
        
        // Сброс счетчика попыток при успехе
        connectionRetries = 0;
        updateConnectionStatus('connected');
        
    } catch (e) {
        // 5. ОБРАБОТКА ОШИБОК
        connectionRetries++;
        
        if (connectionRetries >= maxRetries) {
            appendChatMessage('error', `Сервер недоступен после ${maxRetries} попыток. Переключение в офлайн режим.`);
            enableOfflineMode();
        } else {
            appendChatMessage('error', `Ошибка запроса (попытка ${connectionRetries}/${maxRetries}): ${e.message}`);
            
            // Автоматическая повторная попытка
            setTimeout(() => {
                appendChatMessage('ai', '🔄 Повторная попытка подключения...');
                sendChatMessage();
            }, 2000);
        }
    }
}
```

### 2. **Офлайн обработка**

#### **`generateOfflineResponse()` - Локальные ответы**
```javascript
function generateOfflineResponse(message) {
    const lowerMessage = message.toLowerCase();
    
    // Анализ ключевых слов и генерация ответов
    if (lowerMessage.includes('привет') || lowerMessage.includes('hello')) {
        return 'Привет! Я работаю в офлайн режиме. Могу помочь с базовыми вопросами по программированию.';
    }
    
    if (lowerMessage.includes('python') || lowerMessage.includes('код')) {
        return 'Python - отличный язык программирования! Вот простой пример:\n\n```python\nprint("Hello, World!")\n```';
    }
    
    if (lowerMessage.includes('ошибка') || lowerMessage.includes('error')) {
        return 'Для решения ошибок:\n1. Проверьте синтаксис\n2. Убедитесь в правильности импортов\n3. Проверьте переменные\n4. Используйте try-except для отладки';
    }
    
    if (lowerMessage.includes('помощь') || lowerMessage.includes('help')) {
        return 'Доступные команды:\n- 🔧 Система: информация о системе\n- ❓ Помощь: справка\n- 🧪 Тест: тестирование функций';
    }
    
    return 'Извините, я работаю в офлайн режиме. Попробуйте перезапустить сервер или используйте базовые команды.';
}
```

### 3. **Обработка файлов**

#### **`uploadContentToDatabase()` - Загрузка в базу данных**
```javascript
async function uploadContentToDatabase(filename, content, category, tags) {
    try {
        appendChatMessage('user', `📚 Загрузка в базу данных: ${filename}`);
        
        // 1. ПОДГОТОВКА ДАННЫХ
        const requestData = {
            filename: filename,
            content: content,
            category: category,
            tags: tags ? tags.split(',').map(tag => tag.trim()).filter(tag => tag) : []
        };
        
        // 2. API ЗАПРОС
        const response = await fetch('http://localhost:8083/api/documents/upload-content', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();
        
        // 3. ОБРАБОТКА РЕЗУЛЬТАТА
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

### 4. **Поиск в базе данных**

#### **`searchDocumentsInDatabase()` - Поиск документов**
```javascript
async function searchDocumentsInDatabase(query) {
    try {
        // 1. ПОДГОТОВКА ЗАПРОСА
        const requestData = {
            query: query,
            limit: 5
        };
        
        // 2. API ЗАПРОС
        const response = await fetch('http://localhost:8083/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();
        
        // 3. ОБРАБОТКА РЕЗУЛЬТАТОВ
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

## 🎯 Типы обработки информации

### 1. **Текстовые сообщения**

#### **Обработка команд:**
```javascript
// Проверка команд загрузки
if (text.toLowerCase().includes('загрузи в базу данных') || 
    text.toLowerCase().includes('загрузить в базу')) {
    uploadToDatabase();
    return;
}
```

#### **Анализ ключевых слов:**
```javascript
// Офлайн анализ
if (lowerMessage.includes('python') || lowerMessage.includes('код')) {
    return 'Python - отличный язык программирования!';
}

if (lowerMessage.includes('ошибка') || lowerMessage.includes('error')) {
    return 'Для решения ошибок:\n1. Проверьте синтаксис\n2. Убедитесь в правильности импортов';
}
```

### 2. **Файлы и документы**

#### **Загрузка файлов:**
```javascript
function uploadToDatabase() {
    const content = document.getElementById('code-editor').value;
    if (!content.trim()) {
        alert('Редактор пуст. Введите код или текст для загрузки в базу данных.');
        return;
    }

    // Диалог настройки загрузки
    const filename = prompt('Введите имя файла:', 'document.txt');
    const category = prompt('Выберите категорию:\n1. python_code\n2. documentation\n...');
    const tags = prompt('Введите теги через запятую (необязательно):', '');

    // Загрузка в базу данных
    uploadContentToDatabase(filename, content, category, tags);
}
```

#### **Категории документов:**
- `python_code` - Python код
- `documentation` - Документация
- `configuration` - Конфигурационные файлы
- `industrial_automation` - Промышленная автоматизация
- `artificial_intelligence` - Искусственный интеллект
- `api_documentation` - API документация
- `tutorial` - Обучающие материалы
- `general` - Общие документы

### 3. **Голосовой ввод**

#### **`toggleVoiceInput()` - Обработка голоса**
```javascript
function toggleVoiceInput() {
    try {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            alert('Голосовой ввод не поддерживается этим браузером');
            return;
        }
        
        recognition = new SR();
        recognition.lang = 'ru-RU';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        recognition.onresult = (e) => {
            const transcript = e.results[0][0].transcript;
            const input = document.getElementById('chat-input');
            input.value = transcript;
            setTimeout(sendChatMessage, 50);
        };
        
        recognition.start();
    } catch (e) {
        appendChatMessage('error', `Ошибка голосового ввода: ${e.message}`);
    }
}
```

### 4. **TTS озвучивание**

#### **`speak()` - Текстовое озвучивание**
```javascript
function speak(text) {
    try {
        if (!('speechSynthesis' in window)) return;
        const utter = new SpeechSynthesisUtterance(text.replace(/[#*_`>]/g, ''));
        utter.lang = 'ru-RU';
        utter.rate = 1.0;
        utter.pitch = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utter);
    } catch (e) {
        console.log('TTS недоступен:', e);
    }
}
```

## 🔄 Режимы работы

### 1. **Онлайн режим**
- **API запросы** к серверу `http://localhost:8083`
- **База данных** - загрузка и поиск документов
- **AI обработка** - интеллектуальные ответы
- **Полные возможности** - все функции доступны

### 2. **Офлайн режим**
- **Локальные ответы** - шаблонные ответы
- **Ограниченные возможности** - базовые функции
- **Автономная работа** - без сервера
- **Fallback механизм** - при недоступности сервера

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

### 2. **Автоматическое переключение**
```javascript
function enableOfflineMode() {
    isOfflineMode = true;
    updateConnectionStatus('offline');
    
    const offlineNotice = document.createElement('div');
    offlineNotice.className = 'offline-mode';
    offlineNotice.innerHTML = `
        🌐 Офлайн режим активирован<br>
        API недоступен, используем локальные функции<br>
        <button class="retry-button" onclick="testConnection()">🔄 Повторить подключение</button>
    `;
    
    document.getElementById('chat-messages').appendChild(offlineNotice);
}
```

### 3. **Проверка соединения**
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

## 📊 Мониторинг и статистика

### 1. **Отслеживание состояния**
```javascript
function updateConnectionStatus(status) {
    const indicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    
    indicator.className = `status-indicator ${status}`;
    
    switch (status) {
        case 'connected':
            statusText.textContent = 'Подключено';
            break;
        case 'connecting':
            statusText.textContent = 'Подключение...';
            break;
        case 'offline':
            statusText.textContent = 'Офлайн режим';
            break;
    }
}
```

### 2. **Системная информация**
```javascript
function sendSystemMessage() {
    const systemInfo = `
🔧 Информация о системе:
- Браузер: ${navigator.userAgent}
- Платформа: ${navigator.platform}
- Язык: ${navigator.language}
- Режим: ${isOfflineMode ? 'Офлайн' : 'Онлайн'}
- Попытки подключения: ${connectionRetries}/${maxRetries}
- Статус: ${document.getElementById('status-text').textContent}
    `;
    appendChatMessage('ai', systemInfo);
}
```

## 🎨 Визуализация обработки

### 1. **Отображение сообщений**
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

### 2. **Индикаторы состояния**
- **Статус соединения** - зеленый/желтый/красный
- **Режим работы** - онлайн/офлайн
- **Счетчик попыток** - количество ошибок
- **TTS статус** - включено/выключено

## 🚀 Оптимизация обработки

### 1. **Асинхронная обработка**
- Все API запросы асинхронные
- Неблокирующий интерфейс
- Параллельная обработка файлов

### 2. **Кэширование**
- Сохранение состояния соединения
- Кэширование ответов
- Оптимизация повторных запросов

### 3. **Таймауты**
- Ограничение времени запросов
- Автоматические повторы
- Graceful degradation
