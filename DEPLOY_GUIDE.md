# Rubin AI Smart Dispatcher - GitHub Deploy

## Быстрый деплой на Railway

### 1. Подготовка репозитория
```bash
# Инициализация Git (если еще не сделано)
git init
git add .
git commit -m "Initial commit - Rubin AI Smart Dispatcher"

# Создание репозитория на GitHub
# Перейдите на https://github.com/new
# Создайте репозиторий "rubin-ai-smart-dispatcher"
```

### 2. Загрузка на GitHub
```bash
# Добавление удаленного репозитория
git remote add origin https://github.com/YOUR_USERNAME/rubin-ai-smart-dispatcher.git

# Загрузка кода
git push -u origin main
```

### 3. Деплой на Railway
1. Перейдите на https://railway.app
2. Войдите через GitHub
3. Нажмите "New Project" → "Deploy from GitHub repo"
4. Выберите репозиторий "rubin-ai-smart-dispatcher"
5. Railway автоматически определит Dockerfile и запустит деплой
6. Получите публичный URL вида: `https://rubin-ai-smart-dispatcher-production.up.railway.app`

### 4. Настройка переменных окружения (опционально)
В Railway Dashboard → Settings → Variables:
```
FLASK_ENV=production
PORT=8080
```

### 5. Тестирование
```bash
# Проверка здоровья
curl https://YOUR_RAILWAY_URL/api/health

# Тест чата
curl -X POST https://YOUR_RAILWAY_URL/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "привет"}'
```

## Альтернативный деплой на Fly.io

### 1. Установка flyctl
```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# Или через winget
winget install flyio.flyctl
```

### 2. Логин и создание приложения
```bash
flyctl auth login
flyctl launch
# Следуйте инструкциям, выберите регион
```

### 3. Деплой
```bash
flyctl deploy
```

### 4. Получение URL
```bash
flyctl info
```

## Структура файлов для деплоя
```
rubin-ai-smart-dispatcher/
├── Dockerfile              # Контейнер для Railway/Fly.io
├── requirements.txt         # Python зависимости
├── smart_dispatcher.py     # Основной сервер
├── directives_manager.py   # Менеджер директив
├── api/                    # API модули
│   ├── general_api.py
│   ├── mathematics_api.py
│   └── ...
└── README.md               # Этот файл
```

## Настройка для продакшена

### Изменения в smart_dispatcher.py для продакшена:
```python
# В конце файла заменить:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### Переменные окружения:
- `PORT` - порт сервера (Railway/Fly.io автоматически)
- `FLASK_ENV=production` - режим продакшена
- `DEBUG=False` - отключение отладки

## Мониторинг и логи
- Railway: Dashboard → Deployments → View Logs
- Fly.io: `flyctl logs`

## Обновление деплоя
```bash
# После изменений в коде
git add .
git commit -m "Update smart dispatcher"
git push

# Railway и Fly.io автоматически пересоберут и перезапустят
```

