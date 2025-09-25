# 🚀 Rubin AI Matrix - Быстрый старт

## 📋 Что это?

Rubin AI Matrix - это модульная система искусственного интеллекта для промышленной автоматизации, состоящая из 5 специализированных узлов:

1. **Gateway** - Центральный шлюз (порт 8083)
2. **Compute Core** - Вычислительное ядро (порт 5000)
3. **PostgreSQL** - База данных (порт 5432)
4. **Qdrant** - Векторная база (порт 6333)
5. **Ollama** - AI сервис (порт 11434)

## ⚡ Быстрый запуск

### 1. Предварительные требования:
- Docker Desktop
- 8GB+ RAM
- 20GB+ свободного места

### 2. Запуск системы:

**Windows:**
```cmd
start_matrix.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/start_matrix.sh
./scripts/start_matrix.sh
```

### 3. Проверка работы:

**Windows:**
```cmd
test_matrix.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/test_matrix.sh
./scripts/test_matrix.sh
```

## 🌐 Доступ к сервисам

После запуска система будет доступна по адресам:

- **Веб-интерфейс:** http://localhost:8083
- **API документация:** http://localhost:8083/docs
- **Qdrant:** http://localhost:6333
- **Ollama:** http://localhost:11434

## 💬 Примеры использования

### AI Чат:
```bash
curl -X POST "http://localhost:8083/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Привет, как дела?"}'
```

### Анализ кода:
```bash
curl -X POST "http://localhost:8083/api/code/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "code": "print(\"Hello, World!\")",
       "language": "python"
     }'
```

### Диагностика PMAC:
```bash
curl -X POST "http://localhost:8083/api/diagnostics/pmac" \
     -H "Content-Type: application/json" \
     -d '{
       "command": "status"
     }'
```

## 🔧 Управление

### Остановка системы:
```bash
docker-compose down
```

### Просмотр логов:
```bash
docker-compose logs -f gateway
docker-compose logs -f compute_core
docker-compose logs -f ollama_service
```

### Перезапуск сервиса:
```bash
docker-compose restart gateway
```

## 📊 Мониторинг

### Статус системы:
```bash
curl http://localhost:8083/api/matrix/status
```

### Метрики:
```bash
curl http://localhost:8083/metrics
```

## 🐛 Устранение неполадок

### Проблема: Сервис не запускается
**Решение:**
```bash
# Проверьте логи
docker-compose logs <service_name>

# Пересоберите образы
docker-compose build --no-cache

# Очистите систему
docker system prune -a
```

### Проблема: Медленная работа
**Решение:**
- Увеличьте лимиты памяти в docker-compose.yml
- Проверьте использование ресурсов: `docker stats`
- Убедитесь, что GPU доступен для Ollama

### Проблема: Ошибки подключения
**Решение:**
```bash
# Проверьте сеть
docker network ls
docker network inspect rubin_matrix_network

# Проверьте статус контейнеров
docker-compose ps
```

## 📚 Дополнительная информация

- **Полная документация:** README.md
- **API документация:** http://localhost:8083/docs
- **Исходный код:** gateway_app/, cpp_core/
- **Конфигурация:** docker-compose.yml, .env

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте логи системы
2. Изучите документацию
3. Создайте issue в репозитории

---

**Rubin AI Matrix v2.0** - Модульная система ИИ для промышленной автоматизации 🚀
