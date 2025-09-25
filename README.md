# Rubin AI Smart Dispatcher

Центральный диспетчер для маршрутизации запросов в системе Rubin AI.

## Возможности

- 🧠 Умная маршрутизация запросов по модулям
- 🔄 Проверка здоровья всех модулей
- 📝 История диалогов и контекст
- 🛡️ Этическая проверка запросов
- 📊 Система директив для настройки поведения

## API Endpoints

- `POST /api/chat` - Основной чат
- `GET /api/health` - Проверка здоровья системы
- `GET /api/chat/history` - История диалогов
- `GET /api/chat/context` - Контекст диалога

## Быстрый старт

### Локальный запуск
```bash
pip install -r requirements.txt
python smart_dispatcher.py
```

### Деплой на Railway
1. Форкните репозиторий
2. Подключите к Railway
3. Автоматический деплой!

### Деплой на Fly.io
```bash
flyctl launch
flyctl deploy
```

## Структура проекта

```
├── smart_dispatcher.py      # Основной сервер
├── directives_manager.py    # Менеджер директив
├── api/                    # API модули
├── Dockerfile             # Контейнер
├── requirements.txt       # Зависимости
└── DEPLOY_GUIDE.md       # Подробная инструкция деплоя
```

## Переменные окружения

- `PORT` - Порт сервера (по умолчанию 8080)
- `FLASK_ENV` - Режим Flask (production/development)

## Лицензия

MIT License