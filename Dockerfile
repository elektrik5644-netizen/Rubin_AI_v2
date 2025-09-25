FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY smart_dispatcher.py .
COPY directives_manager.py .
COPY api/ ./api/

# Создание пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Переменные окружения
ENV FLASK_APP=smart_dispatcher.py
ENV FLASK_ENV=production
ENV PORT=8080

# Экспорт порта
EXPOSE 8080

# Команда запуска
CMD ["python", "smart_dispatcher.py"]

