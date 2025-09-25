FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание пользователя
RUN useradd -m -u 1000 rubin && chown -R rubin:rubin /app
USER rubin

# Создание директории для логов
RUN mkdir -p /app/logs

# Экспорт порта
EXPOSE 80

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
