# 🧠 САМООБУЧЕНИЕ И ТЕСТИРОВАНИЕ В RUBIN AI v2

## 📚 **САМООБУЧЕНИЕ НЕЙРОННОЙ СЕТИ**

### **1. Механизм обучения на обратной связи:**

**🔄 Процесс обучения:**
```python
def learn_from_feedback(self, question, correct_category, user_rating):
    """Обучается на основе обратной связи"""
    # 1. Сохранение данных для обучения
    training_data = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'correct_category': correct_category,
        'user_rating': user_rating
    }
    
    # 2. Добавление в файл обучающих данных
    with open('neural_training_data.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(training_data, ensure_ascii=False) + '\n')
```

**📊 Этапы самообучения:**

1. **Сбор обратной связи:**
   - Пользователь оценивает ответ (1-5 звезд)
   - Система записывает правильную категорию
   - Данные сохраняются в `neural_training_data.jsonl`

2. **Подготовка данных:**
   - Создание эмбеддингов для вопросов
   - Преобразование категорий в числовые метки
   - Нормализация данных

3. **Обучение нейронной сети:**
   - Использование Adam оптимизатора
   - L2-регуляризация (weight_decay)
   - Dropout для предотвращения переобучения
   - Логирование процесса в CSV

### **2. Архитектура нейронной сети:**

**🧠 Структура сети:**
```python
class RubinNeuralNetwork(nn.Module):
    def __init__(self, input_size=384, hidden_sizes=[512, 256, 128], num_classes=10):
        # Скрытые слои с различными функциями активации
        # ReLU, Tanh, Softsign, Sigmoid, ELU, LeakyReLU
        # Dropout слои для регуляризации
        # Softmax для классификации
```

**⚙️ Параметры обучения:**
- **Эпохи:** 10 (настраивается)
- **Learning Rate:** 0.001 (настраивается)
- **Weight Decay:** 0.01 (L2-регуляризация)
- **Dropout:** 0.2 (предотвращение переобучения)

### **3. Категории для классификации:**

**📋 Поддерживаемые категории:**
- `математика` - уравнения, расчеты, алгебра
- `физика` - законы физики, механика
- `электротехника` - схемы, компоненты, законы
- `программирование` - алгоритмы, языки, код
- `геометрия` - фигуры, площади, углы
- `химия` - реакции, элементы
- `общие_вопросы` - приветствия, справка
- `техника` - инженерия, механизмы
- `наука` - научные концепции
- `time_series` - временные ряды, прогнозирование

## 🧪 **ТЕСТИРОВАНИЕ СИСТЕМЫ**

### **1. Автоматический тестировщик (`test_neural_integration.py`):**

**🔍 Функции тестирования:**

**A. Проверка статуса нейронной сети:**
```python
def test_neural_status():
    response = requests.get('http://localhost:8080/api/neural-status')
    # Проверяет:
    # - Доступность нейронной сети
    # - Статистику модели
    # - Количество параметров
    # - Историю диалогов
```

**B. Тестирование классификации:**
```python
test_cases = [
    "привет, как дела?",                    # → general
    "как решить квадратное уравнение...",   # → mathematics  
    "объясни закон Кирхгофа",              # → electrical
    "что такое транзистор?",               # → electrical
    "как работает антенна?",               # → radiomechanics
    "что такое ПЛК?",                      # → controllers
    "объясни алгоритм сортировки",         # → programming
    "как передаются данные с ЧПУ...",      # → controllers
    "ASCII-команды и протоколы",          # → controllers
    "сравни C++ и Python..."              # → programming
]
```

**C. Тестирование обучения:**
```python
def test_neural_learning():
    learning_data = [
        {
            'question': 'что такое резистор?',
            'correct_category': 'electrical',
            'rating': 5
        },
        # ... другие примеры
    ]
```

### **2. Ручное тестирование:**

**🌐 Через веб-интерфейс:**
- Отправка запросов на `http://localhost:8080/api/chat`
- Проверка категоризации
- Оценка качества ответов

**📊 Через API:**
```bash
# Тест классификации
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "закон Кирхгофа"}'

# Тест обучения
curl -X POST http://localhost:8080/api/neural-feedback \
  -H "Content-Type: application/json" \
  -d '{"question": "что такое резистор?", "correct_category": "electrical", "rating": 5}'
```

### **3. Мониторинг производительности:**

**📈 Метрики системы:**
- **Точность классификации** - процент правильных категорий
- **Уверенность нейронной сети** - confidence score
- **Время отклика** - скорость обработки запросов
- **Использование fallback** - частота keyword-based логики

**📊 Логирование:**
- **CSV лог обучения** - `training_log.csv`
- **JSONL данные** - `neural_training_data.jsonl`
- **История диалогов** - в памяти системы

## 🔄 **ПРОЦЕСС САМООБУЧЕНИЯ**

### **1. Инициализация:**
```python
# Загрузка предобученной модели (если есть)
self.load_model()

# Инициализация компонентов
self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
self.neural_network = RubinNeuralNetwork(...)
```

### **2. Обучение:**
```python
def train_neural_network(self, training_file, num_epochs=10):
    # 1. Загрузка данных
    training_data = self._load_training_data(training_file)
    
    # 2. Подготовка данных
    embeddings_tensor, labels_tensor = self._prepare_training_data(training_data)
    
    # 3. Обучение
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = self.neural_network(embeddings_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
    
    # 4. Сохранение модели
    self.save_model()
```

### **3. Адаптация:**
- **Новые категории** - автоматическое добавление
- **Улучшение точности** - обучение на новых данных
- **Оптимизация параметров** - настройка learning rate, dropout

## 🎯 **ПРЕИМУЩЕСТВА СИСТЕМЫ**

### **✅ Самообучение:**
- **Адаптивность** - система учится на ошибках
- **Улучшение точности** - со временем классификация становится лучше
- **Персонализация** - адаптация к конкретным задачам

### **✅ Тестирование:**
- **Автоматизация** - регулярные проверки работоспособности
- **Мониторинг** - отслеживание производительности
- **Отладка** - быстрое выявление проблем

### **✅ Надежность:**
- **Fallback механизмы** - keyword-based при сбоях нейронной сети
- **Обработка ошибок** - graceful degradation
- **Восстановление** - автоматическая загрузка сохраненных моделей

## 🚀 **ИСПОЛЬЗОВАНИЕ**

### **Запуск тестирования:**
```bash
python test_neural_integration.py
```

### **Обучение на новых данных:**
```bash
# Добавление данных в neural_training_data.jsonl
# Запуск обучения
python neural_rubin.py
```

### **Мониторинг:**
```bash
# Проверка статуса
curl http://localhost:8080/api/neural-status

# Просмотр логов
tail -f training_log.csv
```

**Система Rubin AI v2 с нейронной сетью обеспечивает непрерывное самообучение и автоматическое тестирование для поддержания высокой точности классификации!** 🧠✨











