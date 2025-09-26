# 🚀 Руководство по деплою Rubin AI v2

Полное руководство по развертыванию системы Rubin AI v2 на различных платформах.

## 📋 Содержание

1. [GitHub Pages](#github-pages)
2. [Railway](#railway)
3. [Fly.io](#flyio)
4. [Heroku](#heroku)
5. [Vercel](#vercel)
6. [Netlify](#netlify)
7. [Docker Hub](#docker-hub)

## 🌐 GitHub Pages

### Автоматический деплой

1. **Включите GitHub Pages:**
   - Перейдите в Settings → Pages
   - Source: GitHub Actions
   - Сохраните настройки

2. **Workflow уже настроен** в `.github/workflows/deploy.yml`

3. **Настройте домен (опционально):**
   - Создайте файл `CNAME` в корне репозитория
   - Добавьте ваш домен: `yourdomain.com`

### Ручной деплой

```bash
# Установка gh-pages
npm install -g gh-pages

# Деплой
gh-pages -d docs/_build/html
```

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
   TELEGRAM_BOT_TOKEN=your_bot_token
   QDRANT_URL=your_qdrant_url
   LOCALAI_URL=your_localai_url
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
   flyctl launch
   ```

4. **Деплой:**
   ```bash
   flyctl deploy
   ```

### Настройка переменных

```bash
flyctl secrets set TELEGRAM_BOT_TOKEN=your_token
flyctl secrets set QDRANT_URL=your_url
flyctl secrets set LOCALAI_URL=your_url
```

## 🟣 Heroku

### Через веб-интерфейс

1. **Создайте приложение:**
   - Зайдите на [heroku.com](https://heroku.com)
   - Нажмите "New" → "Create new app"
   - Выберите имя и регион

2. **Подключите GitHub:**
   - Deploy → GitHub
   - Подключите репозиторий
   - Включите "Automatic deploys"

3. **Настройте переменные:**
   - Settings → Config Vars
   - Добавьте необходимые переменные

### Через Heroku CLI

```bash
# Установка CLI
# Windows: https://devcenter.heroku.com/articles/heroku-cli
# Linux/Mac: https://devcenter.heroku.com/articles/heroku-cli

# Логин
heroku login

# Создание приложения
heroku create rubin-ai-v2

# Подключение к GitHub
heroku git:remote -a rubin-ai-v2

# Деплой
git push heroku main

# Настройка переменных
heroku config:set TELEGRAM_BOT_TOKEN=your_token
heroku config:set QDRANT_URL=your_url
```

## ▲ Vercel

### Быстрый деплой

1. **Подключите GitHub:**
   - Зайдите на [vercel.com](https://vercel.com)
   - Войдите через GitHub
   - Нажмите "New Project"

2. **Выберите репозиторий:**
   - Найдите `elektrik564-netizen/Rubin_AI_v2`
   - Нажмите "Import"

3. **Настройте проект:**
   - Framework Preset: Other
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: `.`

### Через Vercel CLI

```bash
# Установка CLI
npm install -g vercel

# Логин
vercel login

# Деплой
vercel

# Продакшн деплой
vercel --prod
```

## 🟢 Netlify

### Через веб-интерфейс

1. **Подключите GitHub:**
   - Зайдите на [netlify.com](https://netlify.com)
   - Войдите через GitHub
   - Нажмите "New site from Git"

2. **Выберите репозиторий:**
   - GitHub → `elektrik564-netizen/Rubin_AI_v2`

3. **Настройте сборку:**
   - Build command: `pip install -r requirements.txt && python smart_dispatcher.py`
   - Publish directory: `.`

### Через Netlify CLI

```bash
# Установка CLI
npm install -g netlify-cli

# Логин
netlify login

# Деплой
netlify deploy

# Продакшн деплой
netlify deploy --prod
```

## 🐳 Docker Hub

### Сборка и публикация

1. **Создайте аккаунт на Docker Hub**

2. **Сборка образа:**
   ```bash
   docker build -t yourusername/rubin-ai-v2 .
   ```

3. **Публикация:**
   ```bash
   docker push yourusername/rubin-ai-v2
   ```

4. **Запуск:**
   ```bash
   docker run -p 8080:8080 yourusername/rubin-ai-v2
   ```

## 🔧 Настройка переменных окружения

### Обязательные переменные

```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Базы данных
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# AI провайдеры
LOCALAI_URL=http://localhost:11434
LOCALAI_API_KEY=your_localai_api_key

# Google Cloud (опционально)
GOOGLE_CLOUD_PROJECT_ID=your_project_id
GOOGLE_CLOUD_CREDENTIALS_PATH=path/to/credentials.json

# HuggingFace (опционально)
HUGGINGFACE_API_TOKEN=your_huggingface_token

# Cloudflare
CLOUDFLARE_API_TOKEN=your_cloudflare_token
CLOUDFLARE_ZONE_ID=your_zone_id
CLOUDFLARE_DOMAIN=yourdomain.com

# Сервер
PORT=8080
FLASK_ENV=production
SECRET_KEY=your_secret_key
```

## 📊 Мониторинг деплоя

### Health Check

```bash
# Проверка статуса
curl https://your-app.railway.app/api/health
curl https://your-app.fly.dev/api/health
curl https://your-app.herokuapp.com/api/health
```

### Логи

```bash
# Railway
railway logs

# Fly.io
flyctl logs

# Heroku
heroku logs --tail
```

## 🚨 Troubleshooting

### Общие проблемы

1. **Ошибки зависимостей:**
   - Проверьте `requirements.txt`
   - Убедитесь в совместимости версий Python

2. **Проблемы с портами:**
   - Используйте переменную `PORT`
   - Проверьте настройки платформы

3. **Ошибки переменных окружения:**
   - Проверьте правильность токенов
   - Убедитесь в доступности внешних сервисов

### Логи отладки

```bash
# Включение debug режима
export FLASK_ENV=development
export FLASK_DEBUG=1

# Запуск с подробными логами
python -u smart_dispatcher.py
```

## 📚 Дополнительные ресурсы

- [Railway Documentation](https://docs.railway.app/)
- [Fly.io Documentation](https://fly.io/docs/)
- [Heroku Dev Center](https://devcenter.heroku.com/)
- [Vercel Documentation](https://vercel.com/docs)
- [Netlify Documentation](https://docs.netlify.com/)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)

## 🎯 Рекомендации

1. **Для разработки:** Railway или Fly.io
2. **Для продакшна:** Fly.io или Heroku
3. **Для статики:** GitHub Pages или Netlify
4. **Для контейнеров:** Docker Hub + любой хостинг

Выберите платформу в зависимости от ваших потребностей и бюджета! 🚀
