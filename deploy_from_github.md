# 🚀 Развертывание Rubin AI v2 с GitHub

## 📋 Инструкция по развертыванию

### 1. **Клонирование репозитория**

```bash
# Клонируем репозиторий
git clone https://github.com/elektrik5644-netizen/Rubin_AI_v2.git
cd Rubin_AI_v2
```

### 2. **Установка зависимостей**

```bash
# Установка Python зависимостей
pip install -r requirements.txt

# Или для этического ядра
pip install -r requirements.ethical.txt
```

### 3. **Запуск через Docker (Рекомендуется)**

#### **Основная система:**
```bash
# Запуск всех сервисов
docker-compose up -d

# Или только основные компоненты
docker-compose -f docker-compose.smart-dispatcher.yml up -d
```

#### **Отдельные сервисы:**
```bash
# Умный диспетчер
docker-compose -f docker-compose.smart-dispatcher.yml up -d

# Telegram бот
docker-compose -f docker-compose.telegram.yml up -d

# Этическое ядро
docker-compose -f docker-compose.ethical.yml up -d

# Математический сервер
docker-compose -f docker-compose.math.yml up -d

# Программирование
docker-compose -f docker-compose.programming.yml up -d

# Электротехника
docker-compose -f docker-compose.electrical.yml up -d

# GAI сервер
docker-compose -f docker-compose.gai.yml up -d
```

### 4. **Запуск без Docker**

#### **Основные сервисы:**
```bash
# Умный диспетчер (порт 8081)
python enhanced_smart_dispatcher.py

# Telegram бот
python telegram_bot.py

# Нейронная сеть
python neural_rubin.py
```

#### **Специализированные серверы:**
```bash
# Математический сервер (порт 8086)
python math_server.py

# Программирование (порт 8088)
python programming_server.py

# Электротехника (порт 8087)
python electrical_server.py

# GAI сервер (порт 8104)
python gai_server.py

# Этическое ядро (порт 8105)
python ethical_core_server.py
```

### 5. **Быстрый старт**

#### **Windows PowerShell:**
```powershell
# Запуск всех серверов
.\start_all_servers.ps1

# Запуск Docker системы
.\start_docker_system.ps1

# Быстрый перезапуск
.\quick_restart.ps1
```

#### **Linux/Mac:**
```bash
# Запуск основных компонентов
python start_essential.py

# Запуск полной системы
python start_rubin_complete.py

# Стабильная версия
python start_rubin_stable_v2.py
```

### 6. **Проверка статуса**

```bash
# Проверка API
curl http://localhost:8081/api/health

# Проверка статуса
curl http://localhost:8081/api/status

# Список серверов
curl http://localhost:8081/api/servers
```

### 7. **Настройка переменных окружения**

Создайте файл `.env`:
```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Ollama
OLLAMA_URL=http://localhost:11434

# База данных
DATABASE_URL=postgresql://user:password@localhost:5433/rubin_ai

# Redis
REDIS_URL=redis://localhost:6380

# Qdrant
QDRANT_URL=http://localhost:6333
```

### 8. **Мониторинг и отладка**

```bash
# Мониторинг системы
python system_monitor.py

# Отладка и оптимизация
python debug_optimizer.py

# Тестирование API
python test_api.py

# Анализ нейросети
python neural_analysis_report.py
```

### 9. **Обновление с GitHub**

```bash
# Получение обновлений
git pull origin main

# Перезапуск сервисов
docker-compose down
docker-compose up -d --build
```

### 10. **Порты и сервисы**

| Сервис | Порт | Описание |
|--------|------|----------|
| Smart Dispatcher | 8081 | Основной API |
| Telegram Bot | 8080 | Telegram интерфейс |
| Math Server | 8086 | Математические задачи |
| Programming | 8088 | Программирование |
| Electrical | 8087 | Электротехника |
| GAI Server | 8104 | Генеративный AI |
| Ethical Core | 8105 | Этическое ядро |
| Ollama | 11434 | LLM сервер |
| PostgreSQL | 5433 | База данных |
| Redis | 6380 | Кэш |
| Qdrant | 6333 | Векторная БД |

### 11. **Устранение проблем**

#### **Проблемы с портами:**
```bash
# Проверка занятых портов
netstat -an | findstr :8081

# Остановка всех контейнеров
docker-compose down

# Очистка Docker
docker system prune -a
```

#### **Проблемы с зависимостями:**
```bash
# Переустановка зависимостей
pip install -r requirements.txt --force-reinstall

# Обновление pip
pip install --upgrade pip
```

### 12. **Логи и отладка**

```bash
# Просмотр логов Docker
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs -f smart-dispatcher

# Отладка Python
python -m pdb enhanced_smart_dispatcher.py
```

## 🎯 Быстрый старт (1 команда)

```bash
git clone https://github.com/elektrik5644-netizen/Rubin_AI_v2.git && cd Rubin_AI_v2 && docker-compose up -d
```

## 📞 Поддержка

- **GitHub Issues**: [Создать issue](https://github.com/elektrik5644-netizen/Rubin_AI_v2/issues)
- **Документация**: См. файлы `*.md` в репозитории
- **Тестирование**: `python test_api.py`

---

**Rubin AI v2** - Интеллектуальная система с нейронной сетью для решения технических задач.
