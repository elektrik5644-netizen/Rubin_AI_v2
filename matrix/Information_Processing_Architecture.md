# Обработка информации в системе Rubin AI

## 🔄 Общая схема обработки информации

```
┌─────────────────────────────────────────────────────────────────┐
│                    ВХОДНАЯ ИНФОРМАЦИЯ                          │
├─────────────────────────────────────────────────────────────────┤
│  💬 Текстовые сообщения  │  📁 Файлы  │  💻 Код  │  🔍 Запросы  │
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
│                    АНАЛИЗ И ПОНИМАНИЕ                          │
├─────────────────────────────────────────────────────────────────┤
│  🧠 NLP Анализ  │  🎯 Классификация  │  🔑 Ключевые слова      │
│  📝 Извлечение тем  │  😊 Анализ настроения  │  ⚡ Сложность   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ГЕНЕРАЦИЯ ОТВЕТА                            │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Контекстный ответ  │  📚 База знаний  │  🔧 Специализация   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ВЫХОДНАЯ ИНФОРМАЦИЯ                         │
├─────────────────────────────────────────────────────────────────┤
│  💬 Ответы  │  📊 Анализ  │  🔍 Рекомендации  │  📈 Статистика  │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Уровни обработки информации

### 1. **Уровень ввода (Input Layer)**

#### **Типы входной информации:**
- **Текстовые сообщения** - вопросы, утверждения, команды
- **Файлы** - код, документация, конфигурации
- **Код** - Python, PLC, PMAC программы
- **Поисковые запросы** - запросы к базе данных

#### **Предварительная обработка:**
```javascript
// Валидация и очистка
function validateInput(input) {
    if (!input || input.trim().length === 0) {
        return false;
    }
    
    // Очистка от HTML тегов
    const cleanInput = input.replace(/<[^>]*>/g, '');
    
    // Нормализация пробелов
    return cleanInput.replace(/\s+/g, ' ').trim();
}
```

### 2. **Уровень анализа (Analysis Layer)**

#### **Анализ сообщений:**
```javascript
function analyzeMessage(message) {
    const messageLower = message.toLowerCase();
    
    return {
        // Тип сообщения
        isQuestion: message.includes('?') || message.includes('что') || 
                   message.includes('как') || message.includes('почему'),
        isStatement: message.includes('это') || message.includes('работает') || 
                    message.includes('принцип'),
        isFalse: messageLower.includes('неверно') || 
                messageLower.includes('неправильно') || 
                messageLower.includes('только'),
        
        // Извлечение ключевых слов
        keywords: extractKeywords(messageLower),
        
        // Определение тем
        topics: extractTopics(messageLower),
        
        // Анализ настроения
        sentiment: analyzeSentiment(messageLower),
        
        // Оценка сложности
        complexity: analyzeComplexity(message)
    };
}
```

#### **Извлечение ключевых слов:**
```javascript
function extractKeywords(message) {
    const keywords = [];
    const words = message.split(/\s+/);
    
    for (const word of words) {
        if (word.length > 3 && !isStopWord(word)) {
            keywords.push(word);
        }
    }
    
    return keywords.slice(0, 8); // Ограничиваем количество
}
```

#### **Определение тем:**
```javascript
function extractTopics(message) {
    const topics = [];
    
    // Программирование
    if (message.includes('python') || message.includes('программирование') || 
        message.includes('код') || message.includes('переменн')) {
        topics.push('programming');
    }
    
    // Датчики
    if (message.includes('термопар') || message.includes('датчик') || 
        message.includes('измерен') || message.includes('температур')) {
        topics.push('sensors');
    }
    
    // Электротехника
    if (message.includes('кирхгоф') || message.includes('электрическ') || 
        message.includes('цепь') || message.includes('ток')) {
        topics.push('electronics');
    }
    
    // Автоматизация
    if (message.includes('pmac') || message.includes('контроллер') || 
        message.includes('автоматизац') || message.includes('привод')) {
        topics.push('automation');
    }
    
    // Электрооборудование
    if (message.includes('контактор') || message.includes('реле') || 
        message.includes('переключател') || message.includes('коммутац')) {
        topics.push('electrical');
    }
    
    // ИИ
    if (message.includes('ии') || message.includes('ai') || 
        message.includes('машинное обучение') || message.includes('нейронн')) {
        topics.push('ai');
    }
    
    return topics;
}
```

### 3. **Уровень генерации ответов (Response Generation Layer)**

#### **Контекстная генерация:**
```javascript
function generateIntelligentResponse(message, analysis) {
    let response = '';
    
    // Генерация ответа на основе анализа
    if (analysis.topics.includes('programming')) {
        response += generateProgrammingResponse(message, analysis);
    } else if (analysis.topics.includes('sensors')) {
        response += generateSensorResponse(message, analysis);
    } else if (analysis.topics.includes('electronics')) {
        response += generateElectronicsResponse(message, analysis);
    } else if (analysis.topics.includes('automation')) {
        response += generateAutomationResponse(message, analysis);
    } else if (analysis.topics.includes('electrical')) {
        response += generateElectricalResponse(message, analysis);
    } else if (analysis.topics.includes('ai')) {
        response += generateAIResponse(message, analysis);
    } else {
        response += generateGeneralResponse(message, analysis);
    }
    
    return response;
}
```

#### **Специализированные ответы:**
```javascript
function generateProgrammingResponse(message, analysis) {
    let response = '🐍 **Программирование:**\n\n';
    
    if (message.toLowerCase().includes('python')) {
        response += 'Python - высокоуровневый язык программирования с динамической типизацией.\n\n';
        response += '**Особенности Python:**\n';
        response += '• Динамическая типизация - переменные создаются при присваивании\n';
        response += '• Автоматическое управление памятью\n';
        response += '• Простой и читаемый синтаксис\n';
        response += '• Богатая стандартная библиотека\n';
        response += '• Кроссплатформенность\n\n';
    }
    
    if (message.toLowerCase().includes('переменн')) {
        response += '**Переменные в Python:**\n';
        response += '• Создаются при присваивании (name = "value")\n';
        response += '• Не требуют объявления типа\n';
        response += '• Могут менять тип во время выполнения\n';
        response += '• Область видимости определяется местом создания\n';
        response += '• Автоматическое определение типа данных\n\n';
    }
    
    if (analysis.isFalse) {
        response += '**⚠️ Важно:** В Python переменные НЕ нужно объявлять перед использованием!\n\n';
    }
    
    return response;
}
```

## 🔧 Серверная обработка

### **Обработка чата:**
```python
def generate_chat_response(self, message):
    """Генерация ответа чата"""
    message_lower = message.lower()
    
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

### **Анализ кода:**
```python
def analyze_code(self, code, language):
    """Простой анализ кода"""
    issues = []
    recommendations = []
    quality_score = 85.0
    
    if language.lower() == "python":
        if "import *" in code:
            issues.append({
                "type": "warning",
                "message": "Использование 'import *' не рекомендуется",
                "severity": "medium"
            })
            recommendations.append("Используйте конкретные импорты")
        
        if "eval(" in code:
            issues.append({
                "type": "security",
                "message": "Использование eval() может быть небезопасно",
                "severity": "high"
            })
            recommendations.append("Избегайте использования eval()")
    
    quality_score = max(60, quality_score - len(issues) * 5)
    
    return {
        "issues": issues,
        "quality_score": quality_score,
        "recommendations": recommendations,
        "security_report": {"level": "low", "issues": []},
        "summary": {
            "total_issues": len(issues),
            "security_issues": len([i for i in issues if i.get("type") == "security"]),
            "code_length": len(code.split('\n')),
            "language": language
        }
    }
```

## 📊 Обработка файлов

### **Загрузка в базу данных:**
```javascript
async function uploadContentToDatabase(filename, content, category, tags) {
    try {
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

        const result = await response.json();
        
        if (result.status === 'success') {
            // Обработка успешной загрузки
            return result.data;
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        console.error('Ошибка загрузки:', error);
        throw error;
    }
}
```

### **Поиск в базе данных:**
```javascript
async function searchDocumentsInDatabase(query) {
    try {
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
            if (documents.length > 0) {
                let message = `🔍 Найдено документов: ${documents.length}\n\n`;
                documents.forEach((doc, index) => {
                    message += `${index + 1}. **${doc.filename}**\n`;
                    message += `   📂 Категория: ${doc.category}\n`;
                    message += `   🏷️ Теги: ${doc.tags.join(', ')}\n`;
                    message += `   📊 Релевантность: ${doc.relevance}\n`;
                    message += `   📄 Предварительный просмотр: ${doc.content.substring(0, 100)}...\n\n`;
                });
                return message;
            } else {
                return '🔍 Документы не найдены. Попробуйте другой запрос.';
            }
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        console.error('Ошибка поиска:', error);
        throw error;
    }
}
```

## 🎯 Специализации обработки

### 1. **Программирование**
- Анализ синтаксиса
- Выявление ошибок
- Рекомендации по улучшению
- Безопасность кода

### 2. **Промышленная автоматизация**
- PLC программирование
- PMAC контроллеры
- Диагностика оборудования
- Настройка систем

### 3. **Электротехника**
- Законы Кирхгофа
- Электрические цепи
- Контакторы и реле
- Датчики и измерения

### 4. **Искусственный интеллект**
- Компетентностная модель ИИ
- Профессиональные роли
- Навыки и компетенции
- Образовательные траектории

## 📈 Мониторинг и статистика

### **Отслеживание обработки:**
```javascript
function updateStats() {
    messageCount++;
    document.getElementById('totalMessages').textContent = messageCount;
    
    // Обновление статистики тем
    if (analysis.topics.length > 0) {
        totalTopics += analysis.topics.length;
        document.getElementById('totalTopics').textContent = totalTopics;
    }
    
    // Обновление статистики ключевых слов
    if (analysis.keywords.length > 0) {
        totalKeywords += analysis.keywords.length;
        document.getElementById('totalKeywords').textContent = totalKeywords;
    }
}
```

### **Анализ производительности:**
- Время обработки сообщений
- Количество обработанных запросов
- Успешность ответов
- Ошибки и исключения

## 🔄 Обработка ошибок

### **Повторные попытки:**
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

### **Офлайн режим:**
```javascript
function generateOfflineResponse(message) {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('привет') || lowerMessage.includes('hello')) {
        return 'Привет! Я работаю в офлайн режиме. Могу помочь с базовыми вопросами по программированию.';
    }
    
    if (lowerMessage.includes('python') || lowerMessage.includes('код')) {
        return 'Python - отличный язык программирования! Вот простой пример:\n\n```python\nprint("Hello, World!")\n```';
    }
    
    return 'Извините, я работаю в офлайн режиме. Попробуйте перезапустить сервер или используйте базовые команды.';
}
```

## 🎨 Визуализация анализа

### **Секция анализа:**
```javascript
function generateAnalysisSection(analysis) {
    let section = '<div class="analysis-section">';
    section += '<h4>🧠 Анализ сообщения</h4>';
    
    // Тип сообщения
    if (analysis.isQuestion) {
        section += '<p><strong>Тип:</strong> Вопрос</p>';
    } else if (analysis.isStatement) {
        section += '<p><strong>Тип:</strong> Утверждение</p>';
    }
    
    // Темы
    if (analysis.topics.length > 0) {
        section += '<p><strong>Темы:</strong></p>';
        section += '<div class="topics">';
        analysis.topics.forEach(topic => {
            section += `<span class="topic">${topic}</span>`;
        });
        section += '</div>';
    }
    
    // Ключевые слова
    if (analysis.keywords.length > 0) {
        section += '<p><strong>Ключевые слова:</strong></p>';
        section += '<div class="keywords">';
        analysis.keywords.forEach(keyword => {
            section += `<span class="keyword">${keyword}</span>`;
        });
        section += '</div>';
    }
    
    // Предупреждения
    if (analysis.isFalse) {
        section += '<div class="warning">⚠️ Обнаружено потенциально ложное утверждение!</div>';
    }
    
    section += '</div>';
    return section;
}
```

## 🚀 Оптимизация обработки

### **Кэширование:**
- Кэширование частых запросов
- Сохранение результатов анализа
- Оптимизация повторных вычислений

### **Параллельная обработка:**
- Асинхронная обработка файлов
- Параллельный анализ кода
- Многопоточная генерация ответов

### **Масштабирование:**
- Распределенная обработка
- Балансировка нагрузки
- Горизонтальное масштабирование
