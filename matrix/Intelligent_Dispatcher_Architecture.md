# Интеллектуальный диспетчер - Архитектура системы

## 🔄 Общая схема интеллектуального диспетчера

```
┌─────────────────────────────────────────────────────────────────┐
│                    ПОЛЬЗОВАТЕЛЬ                                │
│  💬 Запросы  │  📁 Файлы  │  🎤 Голос  │  🔍 Поиск           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                RubinIDE.html (Frontend)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Анализ запроса                                      │   │
│  │    - Определение типа запроса                          │   │
│  │    - Извлечение ключевых слов                          │   │
│  │    - Классификация сложности                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. Подготовка данных                                   │   │
│  │    - Форматирование запроса                            │   │
│  │    - Добавление контекста                              │   │
│  │    - Приоритизация                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ HTTP API Calls
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│            minimal_rubin_server.py (Backend)                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. Интеллектуальный диспетчер                          │   │
│  │    - Анализ типа задачи                                │   │
│  │    - Выбор оптимального AI провайдера                  │   │
│  │    - Балансировка нагрузки                             │   │
│  │    - Мониторинг производительности                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. Обработка запроса                                   │   │
│  │    - Генерация ответа                                  │   │
│  │    - Анализ кода                                       │   │
│  │    - Поиск в базе данных                               │   │
│  │    - Обучение системы                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ HTTP Responses
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                RubinIDE.html (Frontend)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 5. Обработка ответа                                    │   │
│  │    - Парсинг результата                                │   │
│  │    - Форматирование для отображения                    │   │
│  │    - Обновление интерфейса                             │   │
│  │    - Сохранение в историю                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ПОЛЬЗОВАТЕЛЬ                                │
│  📊 Результаты  │  📚 Обучение  │  🔄 Обратная связь          │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Компоненты интеллектуального диспетчера

### 1. **Анализ запроса (Frontend)**

#### **Определение типа запроса:**
```javascript
function analyzeRequestType(message) {
    const messageLower = message.toLowerCase();
    
    // Анализ ключевых слов для определения типа
    if (messageLower.includes('загрузи') || messageLower.includes('сохрани')) {
        return 'upload';
    } else if (messageLower.includes('найди') || messageLower.includes('поиск')) {
        return 'search';
    } else if (messageLower.includes('анализ') || messageLower.includes('проверь')) {
        return 'analysis';
    } else if (messageLower.includes('помощь') || messageLower.includes('help')) {
        return 'help';
    } else if (messageLower.includes('python') || messageLower.includes('код')) {
        return 'programming';
    } else if (messageLower.includes('plc') || messageLower.includes('pmac')) {
        return 'industrial';
    } else {
        return 'general';
    }
}
```

#### **Извлечение ключевых слов:**
```javascript
function extractKeywords(message) {
    const keywords = [];
    const messageLower = message.toLowerCase();
    
    // Технические термины
    const technicalTerms = [
        'python', 'plc', 'pmac', 'сервопривод', 'пид', 'лир',
        'автоматизация', 'программирование', 'код', 'анализ',
        'безопасность', 'оптимизация', 'диагностика'
    ];
    
    technicalTerms.forEach(term => {
        if (messageLower.includes(term)) {
            keywords.push(term);
        }
    });
    
    return keywords;
}
```

#### **Классификация сложности:**
```javascript
function classifyComplexity(message, keywords) {
    let complexity = 'simple';
    
    // Простые запросы
    if (message.length < 50 && keywords.length <= 2) {
        complexity = 'simple';
    }
    // Средние запросы
    else if (message.length < 200 && keywords.length <= 5) {
        complexity = 'medium';
    }
    // Сложные запросы
    else {
        complexity = 'complex';
    }
    
    return complexity;
}
```

### 2. **Интеллектуальный диспетчер (Backend)**

#### **Основной класс диспетчера:**
```python
class IntelligentDispatcher:
    def __init__(self):
        self.ai_providers = {
            'simple': SimpleAI(),
            'complex': ComplexAI(),
            'industrial': IndustrialAI(),
            'programming': ProgrammingAI()
        }
        self.performance_metrics = {}
        self.learning_data = []
        
    def dispatch_request(self, request_type, message, complexity, keywords):
        """Основная функция диспетчеризации"""
        
        # 1. Анализ типа задачи
        optimal_provider = self.select_optimal_provider(
            request_type, complexity, keywords
        )
        
        # 2. Балансировка нагрузки
        if self.is_provider_overloaded(optimal_provider):
            optimal_provider = self.get_alternative_provider(request_type)
        
        # 3. Обработка запроса
        start_time = time.time()
        response = self.process_request(optimal_provider, message)
        processing_time = time.time() - start_time
        
        # 4. Обновление метрик
        self.update_performance_metrics(
            optimal_provider, processing_time, response.quality
        )
        
        # 5. Обучение системы
        self.learn_from_interaction(
            request_type, complexity, keywords, 
            optimal_provider, processing_time, response.quality
        )
        
        return response
```

#### **Выбор оптимального AI провайдера:**
```python
def select_optimal_provider(self, request_type, complexity, keywords):
    """Выбор оптимального AI провайдера на основе анализа"""
    
    # Специализированные провайдеры для конкретных задач
    if 'plc' in keywords or 'pmac' in keywords:
        return 'industrial'
    elif 'python' in keywords or 'код' in keywords:
        return 'programming'
    elif complexity == 'simple':
        return 'simple'
    elif complexity == 'complex':
        return 'complex'
    else:
        return 'simple'  # По умолчанию
```

#### **Балансировка нагрузки:**
```python
def is_provider_overloaded(self, provider):
    """Проверка перегрузки провайдера"""
    
    if provider not in self.performance_metrics:
        return False
    
    metrics = self.performance_metrics[provider]
    
    # Проверка времени ответа
    if metrics.get('avg_response_time', 0) > 5.0:
        return True
    
    # Проверка количества активных запросов
    if metrics.get('active_requests', 0) > 10:
        return True
    
    return False

def get_alternative_provider(self, request_type):
    """Получение альтернативного провайдера"""
    
    alternatives = {
        'industrial': 'programming',
        'programming': 'complex',
        'complex': 'simple',
        'simple': 'complex'
    }
    
    return alternatives.get(request_type, 'simple')
```

### 3. **Обработка запроса**

#### **Генерация ответа:**
```python
def process_request(self, provider, message):
    """Обработка запроса через выбранный провайдер"""
    
    try:
        if provider == 'simple':
            return self.ai_providers['simple'].get_response(message)
        elif provider == 'complex':
            return self.ai_providers['complex'].analyze_complex_request(message)
        elif provider == 'industrial':
            return self.ai_providers['industrial'].analyze_industrial_code(message)
        elif provider == 'programming':
            return self.ai_providers['programming'].analyze_programming_code(message)
        else:
            return self.ai_providers['simple'].get_response(message)
            
    except Exception as e:
        # Fallback на простой провайдер при ошибке
        return self.ai_providers['simple'].get_response(message)
```

#### **Анализ кода:**
```python
def analyze_code(self, code, language):
    """Анализ кода с использованием интеллектуального диспетчера"""
    
    # Определение типа анализа
    analysis_type = self.determine_analysis_type(code, language)
    
    # Выбор специализированного анализатора
    if analysis_type == 'security':
        return self.security_analyzer.analyze(code, language)
    elif analysis_type == 'performance':
        return self.performance_analyzer.analyze(code, language)
    elif analysis_type == 'quality':
        return self.quality_analyzer.analyze(code, language)
    else:
        return self.general_analyzer.analyze(code, language)

def determine_analysis_type(self, code, language):
    """Определение типа анализа на основе кода"""
    
    if 'eval(' in code or 'exec(' in code:
        return 'security'
    elif 'for' in code and 'range(' in code:
        return 'performance'
    elif len(code.split('\n')) > 100:
        return 'quality'
    else:
        return 'general'
```

### 4. **Поиск в базе данных**

#### **Интеллектуальный поиск:**
```python
def intelligent_search(self, query, context=None):
    """Интеллектуальный поиск в базе данных"""
    
    # Анализ запроса
    query_analysis = self.analyze_search_query(query)
    
    # Поиск по релевантности
    results = self.search_by_relevance(query_analysis)
    
    # Фильтрация по контексту
    if context:
        results = self.filter_by_context(results, context)
    
    # Ранжирование результатов
    ranked_results = self.rank_results(results, query_analysis)
    
    return ranked_results

def analyze_search_query(self, query):
    """Анализ поискового запроса"""
    
    return {
        'keywords': self.extract_keywords(query),
        'intent': self.detect_intent(query),
        'complexity': self.assess_complexity(query),
        'domain': self.identify_domain(query)
    }
```

### 5. **Обучение системы**

#### **Машинное обучение:**
```python
def learn_from_interaction(self, request_type, complexity, keywords, 
                          provider, processing_time, quality):
    """Обучение системы на основе взаимодействий"""
    
    # Сохранение данных для обучения
    interaction_data = {
        'timestamp': time.time(),
        'request_type': request_type,
        'complexity': complexity,
        'keywords': keywords,
        'provider': provider,
        'processing_time': processing_time,
        'quality': quality
    }
    
    self.learning_data.append(interaction_data)
    
    # Обновление модели выбора провайдера
    self.update_provider_selection_model(interaction_data)
    
    # Обновление метрик производительности
    self.update_performance_metrics(provider, processing_time, quality)

def update_provider_selection_model(self, interaction_data):
    """Обновление модели выбора провайдера"""
    
    # Простая эвристическая модель
    if interaction_data['quality'] > 0.8:
        # Успешное взаимодействие - увеличиваем вес провайдера
        self.increase_provider_weight(
            interaction_data['provider'], 
            interaction_data['request_type']
        )
    else:
        # Неуспешное взаимодействие - уменьшаем вес
        self.decrease_provider_weight(
            interaction_data['provider'], 
            interaction_data['request_type']
        )
```

#### **Адаптация под пользователя:**
```python
def adapt_to_user(self, user_id, interaction_history):
    """Адаптация системы под конкретного пользователя"""
    
    # Анализ предпочтений пользователя
    user_preferences = self.analyze_user_preferences(interaction_history)
    
    # Обновление весов провайдеров для пользователя
    self.update_user_specific_weights(user_id, user_preferences)
    
    # Персонализация ответов
    self.personalize_responses(user_id, user_preferences)

def analyze_user_preferences(self, interaction_history):
    """Анализ предпочтений пользователя"""
    
    preferences = {
        'preferred_providers': {},
        'common_request_types': {},
        'average_complexity': 'medium',
        'response_time_preference': 'fast'
    }
    
    # Анализ истории взаимодействий
    for interaction in interaction_history:
        if interaction['quality'] > 0.8:
            provider = interaction['provider']
            preferences['preferred_providers'][provider] = \
                preferences['preferred_providers'].get(provider, 0) + 1
    
    return preferences
```

## 📊 Мониторинг и аналитика

### **Метрики производительности:**
```python
def update_performance_metrics(self, provider, processing_time, quality):
    """Обновление метрик производительности"""
    
    if provider not in self.performance_metrics:
        self.performance_metrics[provider] = {
            'total_requests': 0,
            'avg_response_time': 0,
            'avg_quality': 0,
            'success_rate': 0,
            'active_requests': 0
        }
    
    metrics = self.performance_metrics[provider]
    
    # Обновление метрик
    metrics['total_requests'] += 1
    metrics['avg_response_time'] = (
        (metrics['avg_response_time'] * (metrics['total_requests'] - 1) + 
         processing_time) / metrics['total_requests']
    )
    metrics['avg_quality'] = (
        (metrics['avg_quality'] * (metrics['total_requests'] - 1) + 
         quality) / metrics['total_requests']
    )
    
    if quality > 0.8:
        metrics['success_rate'] = (
            (metrics['success_rate'] * (metrics['total_requests'] - 1) + 
             1) / metrics['total_requests']
        )
    else:
        metrics['success_rate'] = (
            (metrics['success_rate'] * (metrics['total_requests'] - 1)) / 
            metrics['total_requests']
        )
```

### **Аналитика использования:**
```python
def generate_usage_analytics(self):
    """Генерация аналитики использования"""
    
    analytics = {
        'total_requests': sum(m['total_requests'] for m in self.performance_metrics.values()),
        'provider_performance': {},
        'request_type_distribution': {},
        'complexity_distribution': {},
        'quality_trends': {},
        'response_time_trends': {}
    }
    
    # Анализ производительности провайдеров
    for provider, metrics in self.performance_metrics.items():
        analytics['provider_performance'][provider] = {
            'avg_response_time': metrics['avg_response_time'],
            'avg_quality': metrics['avg_quality'],
            'success_rate': metrics['success_rate'],
            'total_requests': metrics['total_requests']
        }
    
    return analytics
```

## 🔄 Полный цикл работы

### **1. Инициация запроса:**
```
Пользователь → Ввод запроса → RubinIDE.html → Анализ запроса
```

### **2. Диспетчеризация:**
```
Frontend → HTTP API → Backend → IntelligentDispatcher → Выбор провайдера
```

### **3. Обработка:**
```
Выбранный провайдер → Обработка запроса → Генерация ответа
```

### **4. Обучение:**
```
Результат → Обновление метрик → Обучение модели → Адаптация
```

### **5. Ответ:**
```
Обработанный ответ → HTTP Response → Frontend → Отображение
```

## 🎯 Преимущества интеллектуального диспетчера

### **1. Оптимизация производительности:**
- Автоматический выбор оптимального AI провайдера
- Балансировка нагрузки между провайдерами
- Мониторинг производительности в реальном времени

### **2. Адаптивное обучение:**
- Обучение на основе взаимодействий пользователей
- Персонализация под конкретного пользователя
- Непрерывное улучшение качества ответов

### **3. Интеллектуальная маршрутизация:**
- Анализ типа запроса и сложности
- Выбор специализированного провайдера
- Fallback механизмы при ошибках

### **4. Мониторинг и аналитика:**
- Детальные метрики производительности
- Аналитика использования
- Тренды качества и времени ответа

### **5. Масштабируемость:**
- Легкое добавление новых AI провайдеров
- Горизонтальное масштабирование
- Поддержка множественных пользователей
