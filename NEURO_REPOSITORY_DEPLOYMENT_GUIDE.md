# 🧠 NeuroRepository API - Руководство по деплою

Полное руководство по развертыванию NeuroRepository API для финансового анализа на различных платформах.

## 📋 Содержание

1. [Обзор](#обзор)
2. [GitHub Actions](#github-actions)
3. [Railway](#railway)
4. [Fly.io](#flyio)
5. [Heroku](#heroku)
6. [Docker](#docker)
7. [Переменные окружения](#переменные-окружения)
8. [API Endpoints](#api-endpoints)

## 🎯 Обзор

NeuroRepository API предоставляет нейросетевые алгоритмы для:
- Финансового анализа
- Торговых стратегий
- Прогнозирования цен
- Оценки рисков
- Кредитного скоринга

## 🚀 GitHub Actions

### Автоматический деплой

Workflow файл: `.github/workflows/neuro-repository-deploy.yml`

**Триггеры:**
- Push в main ветку
- Изменения в `neuro_repository_server.py`
- Pull Request в main

**Платформы деплоя:**
- Railway
- Fly.io
- Heroku

## 🚂 Railway

### Быстрый деплой

1. **Подключите GitHub:**
   - Зайдите на [railway.app](https://railway.app)
   - Войдите через GitHub
   - Нажмите "New Project" → "Deploy from GitHub repo"

2. **Выберите репозиторий:**
   - Найдите `elektrik564-netizen/Rubin_AI_v2`
   - Нажмите "Deploy"

3. **Настройте переменные окружения:**
   ```
   PORT=8090
   NEURO_REPO_PATH=/app/NeuroRepository
   FLASK_ENV=production
   ```

### Через Railway CLI

```bash
# Установка CLI
npm install -g @railway/cli

# Логин
railway login

# Подключение к проекту
railway link

# Деплой
railway up
```

## ✈️ Fly.io

### Установка и настройка

1. **Установите Fly CLI:**
   ```bash
   # Windows
   powershell -c "iwr https://fly.io/install.ps1 -useb | iex"
   
   # Linux/Mac
   curl -L https://fly.io/install.sh | sh
   ```

2. **Логин:**
   ```bash
   flyctl auth login
   ```

3. **Инициализация:**
   ```bash
   flyctl launch --config neuro-repository-fly.toml
   ```

4. **Деплой:**
   ```bash
   flyctl deploy
   ```

### Настройка переменных

```bash
flyctl secrets set PORT=8090
flyctl secrets set NEURO_REPO_PATH=/app/NeuroRepository
flyctl secrets set FLASK_ENV=production
```

## 🟣 Heroku

### Через веб-интерфейс

1. **Создайте приложение:**
   - Зайдите на [heroku.com](https://heroku.com)
   - Нажмите "New" → "Create new app"
   - Выберите имя: `neuro-repository-api`

2. **Подключите GitHub:**
   - Deploy → GitHub
   - Подключите репозиторий
   - Включите "Automatic deploys"

3. **Настройте переменные:**
   - Settings → Config Vars
   - Добавьте:
     ```
     PORT=8090
     NEURO_REPO_PATH=/app/NeuroRepository
     FLASK_ENV=production
     ```

### Через Heroku CLI

```bash
# Установка CLI
# Windows: https://devcenter.heroku.com/articles/heroku-cli
# Linux/Mac: https://devcenter.heroku.com/articles/heroku-cli

# Логин
heroku login

# Создание приложения
heroku create neuro-repository-api

# Подключение к GitHub
heroku git:remote -a neuro-repository-api

# Деплой
git push heroku main

# Настройка переменных
heroku config:set PORT=8090
heroku config:set NEURO_REPO_PATH=/app/NeuroRepository
heroku config:set FLASK_ENV=production
```

## 🐳 Docker

### Локальный запуск

```bash
# Сборка образа
docker build -f neuro-repository-Dockerfile -t neuro-repository-api .

# Запуск контейнера
docker run -p 8090:8090 \
  -e PORT=8090 \
  -e NEURO_REPO_PATH=/app/NeuroRepository \
  -e FLASK_ENV=production \
  neuro-repository-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  neuro-repository:
    build:
      context: .
      dockerfile: neuro-repository-Dockerfile
    ports:
      - "8090:8090"
    environment:
      - PORT=8090
      - NEURO_REPO_PATH=/app/NeuroRepository
      - FLASK_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8090/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔧 Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|------------|----------|----------------------|
| `PORT` | Порт сервера | `8090` |
| `NEURO_REPO_PATH` | Путь к NeuroRepository | `/app/NeuroRepository` |
| `FLASK_ENV` | Режим Flask | `production` |

## 📡 API Endpoints

### Основные endpoints

- `POST /api/neuro/analyze` - Финансовый анализ
- `POST /api/neuro/trade` - Торговые стратегии
- `GET /api/neuro/models` - Доступные модели
- `POST /api/neuro/knowledge` - Знания о нейросетях
- `GET /api/neuro/status` - Статус сервиса
- `GET /api/health` - Проверка здоровья

### Примеры запросов

#### Финансовый анализ
```bash
curl -X POST http://localhost:8090/api/neuro/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Анализ цены акций Apple",
    "type": "price_prediction"
  }'
```

#### Торговые стратегии
```bash
curl -X POST http://localhost:8090/api/neuro/trade \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "Следование за трендом",
    "market": "bull"
  }'
```

#### Получение моделей
```bash
curl -X GET http://localhost:8090/api/neuro/models
```

#### Знания о нейросетях
```bash
curl -X POST http://localhost:8090/api/neuro/knowledge \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "нейросеть"
  }'
```

## 🔍 Мониторинг

### Health Check
```bash
curl -X GET http://localhost:8090/api/health
```

### Статус сервиса
```bash
curl -X GET http://localhost:8090/api/neuro/status
```

## 🚨 Troubleshooting

### Частые проблемы

1. **Порт занят:**
   ```bash
   # Проверить занятые порты
   netstat -tulpn | grep :8090
   
   # Изменить порт
   export PORT=8091
   ```

2. **Ошибки зависимостей:**
   ```bash
   # Переустановить зависимости
   pip install -r requirements.txt --force-reinstall
   ```

3. **Проблемы с путями:**
   ```bash
   # Проверить переменные окружения
   echo $NEURO_REPO_PATH
   
   # Установить правильный путь
   export NEURO_REPO_PATH=/correct/path/to/NeuroRepository
   ```

## 📊 Производительность

### Рекомендации

- **CPU:** Минимум 1 ядро
- **RAM:** Минимум 512 MB
- **Диск:** 1 GB для моделей
- **Сеть:** Стабильное соединение

### Масштабирование

- Горизонтальное масштабирование через load balancer
- Вертикальное масштабирование через увеличение ресурсов
- Кэширование результатов анализа

## 🔐 Безопасность

### Рекомендации

- Используйте HTTPS в продакшене
- Настройте CORS для нужных доменов
- Ограничьте доступ к API
- Регулярно обновляйте зависимости

## 📈 Аналитика

### Метрики

- Количество запросов
- Время ответа
- Использование CPU/RAM
- Ошибки и исключения

### Логирование

- Структурированные логи
- Уровни логирования
- Ротация логов
- Централизованное логирование

## 🎯 Заключение

NeuroRepository API готов к деплою на все основные платформы. Выберите подходящую платформу в зависимости от ваших требований:

- **Railway** - для быстрого старта
- **Fly.io** - для глобального развертывания
- **Heroku** - для простоты использования
- **Docker** - для полного контроля

Все конфигурационные файлы готовы к использованию!





