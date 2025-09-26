# Deployment Guide - Rubin AI v2

Полное руководство по развертыванию системы Rubin AI v2 в различных средах.

## 🚀 Обзор развертывания

Система Rubin AI v2 состоит из множества компонентов, которые можно развертывать как единое целое или по отдельности в зависимости от требований.

## 🏗️ Архитектура развертывания

### Компоненты системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │    │ Smart Dispatcher│    │ NeuroRepository │
│   (Port: N/A)   │◄──►│   (Port: 8080)  │◄──►│   (Port: 8085)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LocalAI       │    │   Qdrant DB     │    │ Specialized     │
│   (Port: 11434) │◄──►│   (Port: 6333)  │◄──►│   Modules       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Порты системы

- **8080** - Smart Dispatcher (основной)
- **8085** - NeuroRepository API
- **8086** - Mathematics Module
- **8087** - Electrical Module
- **8088** - PLC Analysis Module
- **8089** - Programming Module
- **11434** - LocalAI Server
- **6333** - Qdrant Vector Database

## 🐳 Docker развертывание

### Docker Compose конфигурация

```yaml
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  # LocalAI Server
  localai:
    image: localai/localai:latest
    ports:
      - "11434:11434"
    volumes:
      - localai_models:/models
    environment:
      - OLLAMA_HOST=0.0.0.0
    restart: unless-stopped

  # Smart Dispatcher
  smart-dispatcher:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - QDRANT_URL=http://qdrant:6333
      - LOCALAI_URL=http://localai:11434
      - FLASK_ENV=production
    depends_on:
      - qdrant
      - localai
    restart: unless-stopped

  # NeuroRepository API
  neuro-repository:
    build: .
    ports:
      - "8085:8085"
    environment:
      - PORT=8085
      - NEURO_REPO_PATH=/app/NeuroRepository
    volumes:
      - ./NeuroRepository:/app/NeuroRepository
    restart: unless-stopped

  # Telegram Bot
  telegram-bot:
    build: .
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - SMART_DISPATCHER_URL=http://smart-dispatcher:8080
    depends_on:
      - smart-dispatcher
    restart: unless-stopped

volumes:
  qdrant_storage:
  localai_models:
```

### Dockerfile

```dockerfile
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY . /app
WORKDIR /app

# Создание пользователя
RUN useradd -m -u 1000 rubin && chown -R rubin:rubin /app
USER rubin

# Экспорт портов
EXPOSE 8080 8085 8086 8087 8088 8089

# Команда по умолчанию
CMD ["python", "smart_dispatcher.py"]
```

## 🌐 Cloud развертывание

### Railway развертывание

1. **Подготовка репозитория**
```bash
# Создание railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python smart_dispatcher.py",
    "healthcheckPath": "/api/health"
  }
}
```

2. **Переменные окружения**
```bash
# В Railway Dashboard
PORT=8080
QDRANT_URL=https://your-qdrant-instance.com
LOCALAI_URL=https://your-localai-instance.com
TELEGRAM_BOT_TOKEN=your_bot_token
```

3. **Деплой**
```bash
# Установка Railway CLI
npm install -g @railway/cli

# Логин и деплой
railway login
railway link
railway up
```

### Fly.io развертывание

1. **Создание fly.toml**
```toml
app = "rubin-ai-v2"
primary_region = "fra"

[build]

[env]
  PORT = "8080"
  QDRANT_URL = "https://your-qdrant.fly.dev"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 256
```

2. **Деплой**
```bash
# Установка flyctl
curl -L https://fly.io/install.sh | sh

# Логин и деплой
flyctl auth login
flyctl launch
flyctl deploy
```

### Heroku развертывание

1. **Создание Procfile**
```
web: python smart_dispatcher.py
worker: python telegram_bot.py
```

2. **requirements.txt** (уже существует)

3. **Деплой**
```bash
# Установка Heroku CLI
# Логин и деплой
heroku login
heroku create rubin-ai-v2
git push heroku main
```

## 🔧 Локальное развертывание

### Быстрый старт

```bash
# 1. Клонирование репозитория
git clone https://github.com/elektrik5644-netizen/Rubin_AI_v2.git
cd Rubin_AI_v2

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Настройка переменных окружения
cp env_example.txt .env
# Отредактируйте .env файл

# 4. Запуск Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:latest

# 5. Запуск LocalAI (опционально)
docker run -d -p 11434:11434 localai/localai:latest

# 6. Инициализация базы данных
python setup_qdrant.py

# 7. Запуск системы
python smart_dispatcher.py &
python telegram_bot.py &
python api/neuro_repository_api.py &
```

### Пошаговое развертывание

#### Шаг 1: Подготовка окружения
```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

#### Шаг 2: Настройка баз данных
```bash
# Запуск Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Проверка доступности
curl http://localhost:6333/collections
```

#### Шаг 3: Настройка AI провайдеров
```bash
# Запуск LocalAI (если используется)
docker run -d --name localai -p 11434:11434 localai/localai:latest

# Загрузка модели
docker exec localai ollama pull llama2
```

#### Шаг 4: Инициализация системы
```bash
# Создание коллекций в Qdrant
python setup_qdrant.py

# Индексация документов
python index_documents_for_vector_search.py
```

#### Шаг 5: Запуск сервисов
```bash
# Основной диспетчер
python smart_dispatcher.py &

# Telegram Bot
python telegram_bot.py &

# NeuroRepository API
python api/neuro_repository_api.py &

# Специализированные модули
python electrical_server.py &
python math_server.py &
python plc_analysis_api_server.py &
```

## 🔒 Безопасность

### Переменные окружения

```bash
# Создайте .env файл с секретными данными
TELEGRAM_BOT_TOKEN=your_secret_token
QDRANT_API_KEY=your_qdrant_key
LOCALAI_API_KEY=your_localai_key
CLOUDFLARE_API_TOKEN=your_cloudflare_token
```

### Настройка файрвола

```bash
# UFW (Ubuntu)
sudo ufw allow 8080/tcp
sudo ufw allow 8085/tcp
sudo ufw allow 6333/tcp
sudo ufw enable

# iptables
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
iptables -A INPUT -p tcp --dport 8085 -j ACCEPT
iptables -A INPUT -p tcp --dport 6333 -j ACCEPT
```

### HTTPS настройка

```bash
# Использование Let's Encrypt
sudo apt install certbot
sudo certbot --nginx -d yourdomain.com

# Или с Cloudflare
# Настройте SSL/TLS в Cloudflare Dashboard
```

## 📊 Мониторинг

### Health Check скрипт

```python
#!/usr/bin/env python3
import requests
import sys

def check_service(url, name):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

def main():
    services = [
        ("http://localhost:8080/api/health", "Smart Dispatcher"),
        ("http://localhost:8085/api/health", "NeuroRepository"),
        ("http://localhost:6333/collections", "Qdrant"),
        ("http://localhost:11434/api/tags", "LocalAI")
    ]
    
    all_healthy = True
    for url, name in services:
        if not check_service(url, name):
            all_healthy = False
    
    if all_healthy:
        print("\n🎉 Все сервисы работают!")
        sys.exit(0)
    else:
        print("\n⚠️ Некоторые сервисы недоступны!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Логирование

```python
# Настройка логирования для всех компонентов
import logging
import logging.handlers

def setup_logging():
    # Создание форматтера
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Настройка root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Файловый хендлер с ротацией
    file_handler = logging.handlers.RotatingFileHandler(
        'rubin_ai.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
```

## 🚀 Production настройки

### Оптимизация производительности

```python
# Настройки для production
PRODUCTION_CONFIG = {
    "gunicorn": {
        "workers": 4,
        "worker_class": "sync",
        "worker_connections": 1000,
        "max_requests": 1000,
        "max_requests_jitter": 100,
        "timeout": 30,
        "keepalive": 2
    },
    "flask": {
        "DEBUG": False,
        "TESTING": False,
        "SECRET_KEY": "your-secret-key"
    },
    "qdrant": {
        "timeout": 10,
        "retry_attempts": 3,
        "connection_pool_size": 10
    }
}
```

### Масштабирование

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  smart-dispatcher:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

### Load Balancer

```nginx
# nginx.conf
upstream rubin_backend {
    server localhost:8080;
    server localhost:8081;
    server localhost:8082;
}

server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://rubin_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🔧 Troubleshooting

### Общие проблемы

1. **Порты заняты**
```bash
# Проверка занятых портов
netstat -tulpn | grep :8080
# или
lsof -i :8080

# Освобождение порта
sudo kill -9 <PID>
```

2. **Ошибки подключения к базе данных**
```bash
# Проверка статуса Qdrant
docker ps | grep qdrant
docker logs qdrant

# Перезапуск
docker restart qdrant
```

3. **Проблемы с Telegram Bot**
```bash
# Проверка токена
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe"

# Проверка webhook
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getWebhookInfo"
```

### Логи и отладка

```bash
# Просмотр логов всех сервисов
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs -f smart-dispatcher

# Отладка Python приложений
python -u smart_dispatcher.py
```

## 📚 Дополнительные ресурсы

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Railway Documentation](https://docs.railway.app/)
- [Fly.io Documentation](https://fly.io/docs/)
- [Heroku Dev Center](https://devcenter.heroku.com/)
- [Nginx Configuration](https://nginx.org/en/docs/)