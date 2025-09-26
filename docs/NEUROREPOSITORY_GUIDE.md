# NeuroRepository Guide

Руководство по настройке и использованию NeuroRepository для финансового анализа в Rubin AI v2.

## 🧬 Обзор

NeuroRepository - это модуль нейросетевых алгоритмов для финансового анализа, интегрированный в систему Rubin AI v2. Предоставляет возможности торгового анализа, прогнозирования цен и оценки рисков.

## 🚀 Возможности

- **Финансовый анализ** - анализ торговых данных и трендов
- **Прогнозирование цен** - предсказание движения цен
- **Кредитный анализ** - оценка кредитных рисков
- **Торговые стратегии** - разработка и тестирование стратегий
- **Максимальные сети** - использование продвинутых нейросетевых архитектур

## ⚙️ Настройка

### 1. Предварительные требования

```bash
# Установка зависимостей
pip install flask flask-cors requests numpy pandas scikit-learn
pip install tensorflow torch  # Для нейросетевых моделей
```

### 2. Настройка путей

```python
# В neuro_repository_api.py
NEURO_REPO_PATH = r"C:\Users\elekt\OneDrive\Desktop\NeuroRepository-master"
```

### 3. Запуск API сервера

```bash
python api/neuro_repository_api.py
```

## 🏗️ Архитектура

### Структура проектов

```
NeuroRepository-master/
├── NeuroProject-1/          # Базовый торговый анализ
│   ├── models/              # Обученные модели
│   ├── data/               # Торговые данные
│   └── scripts/            # Скрипты анализа
├── NeuroProject-2/         # Кредитный анализ
│   ├── credit_models/      # Модели кредитного анализа
│   └── risk_assessment/    # Оценка рисков
└── TradeEmulator/          # Торговый эмулятор
    ├── strategies/         # Торговые стратегии
    └── backtesting/       # Бэктестинг
```

### Конфигурация проектов

```python
NEURO_PROJECTS = {
    "neuro_project_1": {
        "path": "NeuroProject-1",
        "description": "Базовый нейросетевой проект для анализа торговых данных",
        "capabilities": ["торговый анализ", "прогнозирование цен", "анализ трендов"]
    },
    "neuro_project_2": {
        "path": "NeuroProject-2", 
        "description": "Расширенный проект с кредитными данными",
        "capabilities": ["кредитный анализ", "оценка рисков", "максимальные сети"]
    },
    "trade_emulator": {
        "path": "TradeEmulator",
        "description": "Торговый эмулятор для тестирования стратегий",
        "capabilities": ["симуляция торговли", "тестирование стратегий"]
    }
}
```

## 📡 API Endpoints

### Основные эндпоинты

- `POST /api/neuro/analyze` - Финансовый анализ
- `POST /api/neuro/predict` - Прогнозирование цен
- `POST /api/neuro/credit` - Кредитный анализ
- `POST /api/neuro/trade` - Торговые стратегии
- `GET /api/neuro/models` - Доступные модели
- `GET /api/neuro/projects` - Список проектов

### Примеры запросов

#### Финансовый анализ
```json
POST /api/neuro/analyze
{
    "project": "neuro_project_1",
    "data": {
        "symbol": "AAPL",
        "period": "1Y",
        "indicators": ["SMA", "RSI", "MACD"]
    }
}
```

#### Прогнозирование цен
```json
POST /api/neuro/predict
{
    "project": "neuro_project_1",
    "model": "lstm_price_predictor",
    "input_data": {
        "historical_prices": [...],
        "features": [...]
    }
}
```

#### Кредитный анализ
```json
POST /api/neuro/credit
{
    "project": "neuro_project_2",
    "applicant_data": {
        "income": 50000,
        "credit_score": 750,
        "debt_ratio": 0.3
    }
}
```

## 🧠 Нейросетевые модели

### Доступные модели

1. **LSTM Price Predictor** - прогнозирование цен
2. **CNN Pattern Recognition** - распознавание паттернов
3. **RNN Sequence Analysis** - анализ временных рядов
4. **GAN Synthetic Data** - генерация синтетических данных
5. **Transformer Attention** - внимание к важным признакам

### Обучение моделей

```python
def train_model(project_name, model_type, training_data):
    """Обучение нейросетевой модели"""
    project_path = NEURO_PROJECTS[project_name]["path"]
    
    # Загрузка данных
    data = load_training_data(project_path, training_data)
    
    # Создание модели
    model = create_model(model_type)
    
    # Обучение
    model.fit(data["X"], data["y"], epochs=100)
    
    # Сохранение
    save_model(model, project_path)
    
    return model
```

## 📊 Анализ данных

### Финансовые индикаторы

- **SMA (Simple Moving Average)** - простая скользящая средняя
- **RSI (Relative Strength Index)** - индекс относительной силы
- **MACD** - схождение-расхождение скользящих средних
- **Bollinger Bands** - полосы Боллинджера
- **Stochastic Oscillator** - стохастический осциллятор

### Технический анализ

```python
def calculate_technical_indicators(data):
    """Расчет технических индикаторов"""
    indicators = {}
    
    # SMA
    indicators['sma_20'] = data['close'].rolling(20).mean()
    indicators['sma_50'] = data['close'].rolling(50).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    indicators['rsi'] = 100 - (100 / (1 + rs))
    
    return indicators
```

## 🎯 Торговые стратегии

### Стратегии

1. **Trend Following** - следование тренду
2. **Mean Reversion** - возврат к среднему
3. **Momentum** - импульсные стратегии
4. **Arbitrage** - арбитражные стратегии
5. **Machine Learning** - ML-стратегии

### Бэктестинг

```python
def backtest_strategy(strategy, historical_data):
    """Бэктестинг торговой стратегии"""
    results = {
        "total_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "win_rate": 0
    }
    
    # Симуляция торговли
    for i in range(len(historical_data)):
        signal = strategy.generate_signal(historical_data[i])
        if signal == "BUY":
            # Логика покупки
            pass
        elif signal == "SELL":
            # Логика продажи
            pass
    
    return results
```

## 🔧 Интеграция с Rubin AI

### Маршрутизация запросов

```python
def route_neuro_request(request_data):
    """Маршрутизация запросов к NeuroRepository"""
    query = request_data.get("query", "").lower()
    
    if any(keyword in query for keyword in ["нейросеть", "нейронная сеть", "финансы"]):
        return "neuro_repository"
    elif any(keyword in query for keyword in ["торговля", "акции", "прогноз"]):
        return "neuro_project_1"
    elif any(keyword in query for keyword in ["кредит", "риск", "заем"]):
        return "neuro_project_2"
    
    return "general"
```

### Обработка ответов

```python
def process_neuro_response(response):
    """Обработка ответов от NeuroRepository"""
    if response.get("success"):
        return {
            "type": "financial_analysis",
            "data": response["data"],
            "confidence": response.get("confidence", 0.8),
            "explanation": response.get("explanation", "")
        }
    else:
        return {
            "type": "error",
            "message": response.get("error", "Unknown error")
        }
```

## 🛠️ Разработка

### Добавление новых проектов

1. Создайте папку проекта в NeuroRepository
2. Добавьте конфигурацию в `NEURO_PROJECTS`
3. Создайте API эндпоинты
4. Добавьте обработчики в Smart Dispatcher

### Создание новых моделей

```python
def create_custom_model(model_config):
    """Создание пользовательской модели"""
    if model_config["type"] == "LSTM":
        model = Sequential([
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
    elif model_config["type"] == "CNN":
        model = Sequential([
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(1)
        ])
    
    model.compile(optimizer='adam', loss='mse')
    return model
```

## 🐛 Отладка

### Общие проблемы

1. **Модели не загружаются**
   - Проверьте пути к файлам моделей
   - Убедитесь в совместимости версий TensorFlow/PyTorch

2. **Ошибки предсказания**
   - Проверьте формат входных данных
   - Убедитесь в нормализации данных

3. **Медленная работа**
   - Используйте GPU для обучения
   - Оптимизируйте размер батча

### Логирование

```python
import logging
logger = logging.getLogger("neuro_repository")

def log_model_performance(model_name, metrics):
    logger.info(f"Model {model_name} performance: {metrics}")
```

## 📊 Мониторинг

### Метрики производительности

- Точность предсказаний
- Время обработки запросов
- Использование памяти
- Загрузка GPU/CPU

### Health Check

```python
def check_neuro_health():
    """Проверка здоровья NeuroRepository"""
    health = {
        "models_loaded": check_models_status(),
        "memory_usage": get_memory_usage(),
        "gpu_available": check_gpu_availability()
    }
    return health
```

## 🚀 Развертывание

### Локальное развертывание

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск API
python api/neuro_repository_api.py
```

### Docker развертывание

```dockerfile
FROM python:3.9-slim
COPY api/neuro_repository_api.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8085
CMD ["python", "neuro_repository_api.py"]
```

### Production настройки

- Используйте GPU для обучения моделей
- Настройте мониторинг производительности
- Реализуйте кэширование предсказаний
- Добавьте автоматическое переобучение моделей

## 📚 Дополнительные ресурсы

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Financial Data Analysis](https://pandas.pydata.org/docs/)
- [Technical Analysis Library](https://ta-lib.org/)
