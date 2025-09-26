# Rubin AI v2 - Advanced AI System

Комплексная система искусственного интеллекта с умным диспетчером, интеграцией Telegram Bot, NeuroRepository, LocalAI и множеством специализированных модулей.

## 🚀 Основные компоненты

### 🧠 Smart Dispatcher (Центральный диспетчер)
- Умная маршрутизация запросов по модулям
- Проверка здоровья всех модулей
- История диалогов и контекст
- Этическая проверка запросов
- Система директив для настройки поведения

### 📱 Telegram Bot
- Интеграция с Telegram для взаимодействия через мессенджер
- Пересылка сообщений в Smart Dispatcher
- Поддержка документов и изображений
- Анализ электрических схем через Telegram

### 🧬 NeuroRepository
- Нейросетевые алгоритмы для финансового анализа
- Торговый анализ и прогнозирование цен
- Кредитный анализ и оценка рисков
- Торговый эмулятор для тестирования стратегий

### 🤖 LocalAI Integration
- Локальная интеграция с AI моделями
- Поддержка различных провайдеров (Google Cloud, HuggingFace, GPT)
- Умный выбор провайдера для задач
- Офлайн работа с локальными моделями

### 🗄️ Qdrant Vector Database
- Векторный поиск по документам
- Индексация знаний и документов
- Семантический поиск
- Интеграция с Cloudflare

### 🔧 Специализированные модули
- **Электрика** - анализ электрических схем, закон Ома
- **Математика** - решение математических задач, символьные вычисления
- **PLC анализ** - анализ программируемых логических контроллеров
- **Радиотехника** - модуляция и демодуляция сигналов
- **Программирование** - анализ кода, автоматическое исправление ошибок

## 🏗️ Архитектура системы

```
Rubin AI v2
├── Smart Dispatcher (Port 8080)
│   ├── API Endpoints
│   ├── Health Monitoring
│   └── Request Routing
├── Telegram Bot
│   ├── Message Processing
│   └── Document Analysis
├── NeuroRepository API (Port 8085)
│   ├── Financial Analysis
│   └── Trading Strategies
├── LocalAI Provider
│   ├── Model Management
│   └── Local Processing
├── Qdrant Vector DB
│   ├── Document Indexing
│   └── Semantic Search
└── Specialized Modules
    ├── Electrical (Port 8087)
    ├── Mathematics (Port 8086)
    ├── PLC Analysis (Port 8088)
    └── Programming (Port 8089)
```

## 🚀 Быстрый старт

### Предварительные требования
- Python 3.8+
- Docker (для Qdrant)
- Telegram Bot Token
- LocalAI сервер (опционально)

### Установка и запуск

1. **Клонирование репозитория**
```bash
git clone https://github.com/elektrik5644-netizen/Rubin_AI_v2.git
cd Rubin_AI_v2
```

2. **Установка зависимостей**
```bash
pip install -r requirements.txt
```

3. **Настройка переменных окружения**
```bash
# Создайте .env файл
cp env_example.txt .env

# Настройте переменные:
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
LOCALAI_URL=http://127.0.0.1:11434
QDRANT_URL=http://127.0.0.1:6333
```

4. **Запуск системы**
```bash
# Запуск основного диспетчера
python smart_dispatcher.py

# Запуск Telegram Bot (в отдельном терминале)
python telegram_bot.py

# Запуск NeuroRepository API (в отдельном терминале)
python api/neuro_repository_api.py
```

## 📡 API Endpoints

### Smart Dispatcher (Port 8080)
- `POST /api/chat` - Основной чат
- `GET /api/health` - Проверка здоровья системы
- `GET /api/chat/history` - История диалогов
- `GET /api/chat/context` - Контекст диалога

### NeuroRepository (Port 8085)
- `POST /api/neuro/analyze` - Финансовый анализ
- `POST /api/neuro/trade` - Торговые стратегии
- `GET /api/neuro/models` - Доступные модели

### Electrical Module (Port 8087)
- `POST /api/graph/analyze` - Анализ электрических схем
- `POST /api/graph/digitize` - Оцифровка схем

### Mathematics Module (Port 8086)
- `POST /api/math/solve` - Решение математических задач
- `POST /api/math/symbolic` - Символьные вычисления

## 🐳 Docker развертывание

```bash
# Запуск Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Запуск всей системы
docker-compose up -d
```

## 🌐 Интеграции

- **Cloudflare** - CDN и защита
- **Dash Cloudflare** - Мониторинг и аналитика
- **Qdrant.io** - Векторная база данных
- **Telegram API** - Мессенджер интеграция
- **LocalAI** - Локальные AI модели

## 📁 Структура проекта

```
Rubin_AI_v2/
├── smart_dispatcher.py          # Основной диспетчер
├── telegram_bot.py             # Telegram Bot
├── api/                        # API модули
│   ├── neuro_repository_api.py # NeuroRepository API
│   ├── electrical_api.py       # Электрический модуль
│   ├── mathematics_api.py      # Математический модуль
│   └── vector_search_api.py   # Векторный поиск
├── providers/                  # AI провайдеры
│   ├── localai_provider.py    # LocalAI интеграция
│   ├── google_cloud_provider.py
│   └── huggingface_provider.py
├── matrix/                     # Расширенная система
│   ├── RubinIDE.html          # Веб-интерфейс
│   ├── gateway_app/           # Gateway приложение
│   └── docker-compose.yml    # Docker конфигурация
├── docs/                       # Документация
├── requirements.txt           # Зависимости
└── DEPLOY_GUIDE.md           # Инструкция по развертыванию
```

## 🔧 Переменные окружения

- `PORT` - Порт сервера (по умолчанию 8080)
- `TELEGRAM_BOT_TOKEN` - Токен Telegram Bot
- `LOCALAI_URL` - URL LocalAI сервера
- `QDRANT_URL` - URL Qdrant сервера
- `FLASK_ENV` - Режим Flask (production/development)

## 📚 Документация модулей

- [Smart Dispatcher Guide](docs/SMART_DISPATCHER_GUIDE.md)
- [Telegram Bot Guide](docs/TELEGRAM_BOT_GUIDE.md)
- [NeuroRepository Guide](docs/NEUROREPOSITORY_GUIDE.md)
- [LocalAI Integration Guide](docs/LOCALAI_GUIDE.md)
- [Qdrant Setup Guide](docs/QDRANT_GUIDE.md)
- [Deployment Guide](DEPLOY_GUIDE.md)

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

MIT License

## 🆘 Поддержка

Для получения поддержки создайте Issue в репозитории или обратитесь к документации модулей.