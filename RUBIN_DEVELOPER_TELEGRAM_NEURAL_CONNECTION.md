# 🔗 КАК СВЯЗАНЫ РУБИН DEVELOPER, ТЕЛЕГРАМ БОТ И НЕЙРОННАЯ СЕТЬ

## 📊 ОБЗОР СВЯЗЕЙ

**RubinDeveloper** - это веб-интерфейс для разработчиков, **Telegram Bot** - интерфейс для пользователей, а **нейронная сеть** - это центральный мозг системы, который обеспечивает интеллектуальную обработку всех запросов.

---

## 🌐 СХЕМА СВЯЗЕЙ

```
Пользователь
    │
    ├── RubinDeveloper (веб-интерфейс)
    │   │
    │   └── intelligentRouting() → Smart Dispatcher
    │
    └── Telegram Bot
        │
        └── ask_dispatcher() → Smart Dispatcher
                │
                ▼
        Smart Dispatcher (порт 8080)
                │
                ├── categorize_message() → Neural Network
                │
                └── forward_request() → Специализированные модули
```

---

## 📱 TELEGRAM BOT → НЕЙРОННАЯ СЕТЬ

### **Поток данных:**
```python
# 1. Telegram Bot получает сообщение
def ask_dispatcher(message: str) -> str:
    payload = {"message": message}
    r = requests.post(SMART_DISPATCHER_URL, json=payload, timeout=60)
    # SMART_DISPATCHER_URL = "http://localhost:8080/api/chat"

# 2. Smart Dispatcher получает запрос
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    
    # 3. Категоризация через нейронную сеть
    category = categorize_message(message)
    
    # 4. Маршрутизация к модулю
    result, error = forward_request(category, message)
```

### **Ключевые функции:**
- **Прием сообщений** через long polling
- **Пересылка** в Smart Dispatcher
- **Обработка документов** и изображений
- **Возврат ответов** пользователю

---

## 🖥️ RUBINDEVELOPER → НЕЙРОННАЯ СЕТЬ

### **Веб-интерфейс для разработчиков:**
```javascript
// 1. Пользователь вводит сообщение в RubinDeveloper
async function sendMessage() {
    const message = messageInput.value.trim();
    
    // 2. Интеллектуальная маршрутизация
    const routingResult = await intelligentRouting(message);
    
    // 3. Отправка запроса к модулю
    response = await fetch(routingResult.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(routingResult.requestBody)
    });
}
```

### **Интеллектуальная маршрутизация:**
```javascript
async function intelligentRouting(message) {
    const lower = message.toLowerCase();
    
    // Анализ ключевых слов для определения модуля
    if (electricalKeywords.some(keyword => lower.includes(keyword))) {
        return {
            module: 'Электротехника',
            url: 'http://localhost:8087/api/solve',
            requestBody: { message: message }
        };
    }
    
    // По умолчанию - Smart Dispatcher
    return {
        module: 'Smart Dispatcher', 
        url: 'http://localhost:8080/api/chat',
        requestBody: { message: message }
    };
}
```

### **Тестирование нейронной сети:**
```javascript
async function testNeuralNetwork() {
    const response = await fetch('http://localhost:8090/api/neuro/status', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
    });
    
    if (response.ok) {
        addMessage('ai', '✅ Нейронная сеть работает!');
    }
}
```

---

## 🧠 SMART DISPATCHER - ЦЕНТРАЛЬНЫЙ УЗЕЛ

### **Основные функции:**
```python
def categorize_message(message):
    """Определение категории через нейронную сеть или ключевые слова"""
    
    # 1. Попытка использования нейронной сети
    if NEURAL_NETWORK_AVAILABLE:
        try:
            category, confidence = neural_categorize(message)
            if confidence > CONFIDENCE_THRESHOLD:
                return category
        except Exception as e:
            logger.warning(f"Нейронная сеть недоступна: {e}")
    
    # 2. Fallback на ключевые слова
    return categorize_by_keywords(message)

def forward_request(category, message):
    """Пересылка запроса к соответствующему модулю"""
    
    # 1. Получение конфигурации модуля
    config = SERVERS[category]
    url = f"http://localhost:{config['port']}{config['endpoint']}"
    
    # 2. Подготовка payload в зависимости от модуля
    if category in ['electrical', 'programming']:
        payload = {'message': message}
    elif category == 'neuro':
        payload = {'message': message}
    
    # 3. Отправка запроса
    response = requests.post(url, json=payload, timeout=15)
    return response.json()
```

---

## 🔄 НЕЙРОННАЯ СЕТЬ - МОЗГ СИСТЕМЫ

### **Архитектура:**
```python
class RubinNeuralNetwork(nn.Module):
    def __init__(self, input_size=384, hidden_sizes=[512, 256, 128], num_classes=10):
        # Входной слой: 384-мерные эмбеддинги (SentenceTransformer)
        # Скрытые слои: 512 → 256 → 128 нейронов + ReLU + Dropout
        # Выходной слой: 10 категорий
```

### **Процесс классификации:**
```python
def categorize_question(self, question: str) -> Tuple[str, float]:
    """Категоризация вопроса через нейронную сеть"""
    
    # 1. Создание эмбеддинга
    if self.sentence_model:
        embedding = self.sentence_model.encode([question])
        embedding_tensor = torch.FloatTensor(embedding).to(self.device)
        
        # 2. Прогон через нейронную сеть
        with torch.no_grad():
            outputs = self.neural_network(embedding_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 3. Определение категории и уверенности
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
            
            category = self.categories[predicted_idx]
            
        return category, confidence
```

---

## 🔗 СВЯЗЫВАЮЩИЕ КОМПОНЕНТЫ

### **1. Контекстная память:**
```python
CONVERSATION_HISTORY = {
    "sessions": {
        "session_id": {
            "messages": [...],
            "context": {
                "current_topic": None,
                "last_module": None
            }
        }
    }
}

def get_context_for_message(session_id, message):
    """Добавление контекста к сообщению"""
    session = CONVERSATION_HISTORY["sessions"][session_id]
    if session["messages"]:
        context_hint = f"[Контекст: {session['context']['last_module']}] "
        return context_hint + message
    return message
```

### **2. Этическая проверка:**
```python
def ethical_check(message, category):
    """Проверка безопасности запроса"""
    if ETHICAL_CORE_AVAILABLE:
        try:
            response = requests.post('http://localhost:8105/api/ethical/assess', 
                                   json={'message': message, 'category': category})
            result = response.json()
            return result.get('safe', True), result.get('reason', '')
        except:
            return True, "Этическая проверка недоступна"
    return True, "Этическая проверка отключена"
```

---

## 📊 КОНФИГУРАЦИЯ МОДУЛЕЙ

### **Серверы и их назначение:**
```python
SERVERS = {
    'mathematics': {
        'port': 8086, 
        'endpoint': '/api/solve',
        'keywords': ['уравнение', 'формула', 'вычисли', 'математика']
    },
    'electrical': {
        'port': 8087, 
        'endpoint': '/api/solve',
        'keywords': ['резистор', 'конденсатор', 'схема', 'ток']
    },
    'programming': {
        'port': 8088, 
        'endpoint': '/api/explain',
        'keywords': ['код', 'программа', 'алгоритм', 'функция']
    },
    'neuro': {
        'port': 8090, 
        'endpoint': '/api/analyze',
        'keywords': ['нейронная сеть', 'машинное обучение', 'AI']
    },
    'general': {
        'port': 8085, 
        'endpoint': '/api/chat',
        'keywords': ['общий', 'help', 'помощь']
    }
}
```

---

## 🎯 МАРШРУТИЗАЦИЯ ЗАПРОСОВ

### **RubinDeveloper маршрутизация:**
```javascript
// Ключевые слова для каждого модуля
const electricalKeywords = [
    'резистор', 'конденсатор', 'диод', 'транзистор',
    'напряжение', 'ток', 'схема', 'цепи'
];

const programmingKeywords = [
    'код', 'программа', 'алгоритм', 'функция',
    'переменная', 'цикл', 'условие', 'массив'
];

const mathematicsKeywords = [
    'уравнение', 'формула', 'интеграл', 'производная',
    'матрица', 'вектор', 'график', 'функция'
];
```

### **Smart Dispatcher категоризация:**
```python
def categorize_message(message):
    """Категоризация сообщения"""
    message_lower = message.lower()
    
    # Математика
    if any(word in message_lower for word in ['уравнение', 'формула', 'интеграл']):
        return 'mathematics'
    
    # Электротехника  
    if any(word in message_lower for word in ['резистор', 'ток', 'напряжение']):
        return 'electrical'
    
    # Программирование
    if any(word in message_lower for word in ['код', 'программа', 'алгоритм']):
        return 'programming'
    
    # Нейронные сети
    if any(word in message_lower for word in ['нейронная сеть', 'машинное обучение']):
        return 'neuro'
    
    return 'general'
```

---

## 📈 МОНИТОРИНГ И ДИАГНОСТИКА

### **RubinDeveloper диагностика:**
```javascript
// Проверка всех модулей
async function checkAllModulesStatus() {
    const results = [];
    
    for (const [name, config] of Object.entries(modules)) {
        try {
            const response = await fetch(`http://localhost:${config.port}/api/health`);
            results.push({ name, online: response.ok });
        } catch (e) {
            results.push({ name, online: false });
        }
    }
    
    return results;
}

// Тестирование API
async function runComprehensiveAPITest() {
    addMessage('ai', '🧪 Запускаю комплексное тестирование API...');
    
    const tests = [
        { name: 'Smart Dispatcher', test: testSmartDispatcher },
        { name: 'Neural Network', test: testNeuralNetwork },
        { name: 'Mathematics', test: testMathematicsModule },
        { name: 'Electrical', test: testElectricalModule }
    ];
    
    for (const { name, test } of tests) {
        const result = await test();
        const status = result ? '✅' : '❌';
        addMessage('ai', `${status} ${name}: ${result ? 'работает' : 'недоступен'}`);
    }
}
```

---

## 🔄 ПОЛНЫЙ ЦИКЛ ВЗАИМОДЕЙСТВИЯ

### **Scenario 1: Запрос через Telegram Bot**
```
1. Пользователь → Telegram Bot: "Объясни закон Ома"
2. Telegram Bot → Smart Dispatcher: POST /api/chat {"message": "Объясни закон Ома"}
3. Smart Dispatcher → Neural Network: categorize("Объясни закон Ома") → "electrical"
4. Smart Dispatcher → Electrical Module: POST /api/solve {"message": "Объясни закон Ома"}
5. Electrical Module → Smart Dispatcher: {"response": "Закон Ома: U = I * R..."}
6. Smart Dispatcher → Telegram Bot: {"success": true, "response": "Закон Ома..."}
7. Telegram Bot → Пользователь: "Закон Ома: U = I * R..."
```

### **Scenario 2: Запрос через RubinDeveloper**
```
1. Разработчик → RubinDeveloper: "напиши функцию сортировки"
2. RubinDeveloper → intelligentRouting: анализ ключевых слов → "programming"
3. RubinDeveloper → Programming Module: POST /api/explain {"message": "напиши функцию сортировки"}
4. Programming Module → RubinDeveloper: {"response": "def bubble_sort(arr):..."}
5. RubinDeveloper → Разработчик: отображение ответа в чате
```

### **Scenario 3: Прямой запрос к нейронной сети**
```
1. RubinDeveloper → Neural Network: GET /api/neuro/status
2. Neural Network → RubinDeveloper: {"success": true, "neural_available": true}
3. RubinDeveloper → Пользователь: "✅ Нейронная сеть работает!"
```

---

## 🎯 КЛЮЧЕВЫЕ ОСОБЕННОСТИ СВЯЗЕЙ

### **1. Двунаправленная связь:**
- **RubinDeveloper** ↔ **Smart Dispatcher** ↔ **специализированные модули**
- **Telegram Bot** ↔ **Smart Dispatcher** ↔ **специализированные модули**
- **Smart Dispatcher** ↔ **Neural Network** (для классификации)

### **2. Интеллектуальная маршрутизация:**
- **Frontend (RubinDeveloper)**: анализ ключевых слов в JavaScript
- **Backend (Smart Dispatcher)**: нейронная сеть + fallback на ключевые слова
- **Контекстная память**: сохранение истории диалогов

### **3. Отказоустойчивость:**
- **Fallback механизмы**: если нейронная сеть недоступна → ключевые слова
- **General модуль**: обработка неопознанных запросов
- **Проверка здоровья**: регулярная диагностика всех модулей

### **4. Расширяемость:**
- **Модульная архитектура**: легкое добавление новых серверов
- **API-based**: стандартизированные HTTP эндпоинты
- **Конфигурируемая маршрутизация**: настройка ключевых слов и приоритетов

---

## 🚀 БУДУЩИЕ УЛУЧШЕНИЯ

### **Планируемые функции:**
1. **WebSocket соединения** для real-time обновлений
2. **Персонализация** ответов на основе истории
3. **Мультимодальность** (текст + изображения + голос)
4. **Расширенная аналитика** использования модулей
5. **Автоматическое масштабирование** под нагрузкой

---

*Этот документ объясняет архитектуру связей между RubinDeveloper, Telegram Bot и нейронной сетью в системе Rubin AI v2.*





