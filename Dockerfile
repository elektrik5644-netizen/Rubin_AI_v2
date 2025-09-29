# Dockerfile для сервера контроллеров
FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка PyTorch (CPU-версия для простоты)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

# Копирование исходного кода
COPY controllers_server.py .
COPY enhanced_smart_dispatcher.py .
COPY neural_rubin.py .
COPY mathematical_problem_solver.py .
COPY rubin_time_series_processor.py .
COPY rubin_data_preprocessor.py .
COPY handlers/ ./handlers/
COPY enhanced_qdrant_adapter.py .
COPY test_model_loader.py .

# Создание пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Открытие порта
EXPOSE 8080

# Переменные окружения
ENV FLASK_APP=enhanced_smart_dispatcher.py
ENV FLASK_ENV=production

# Команда запуска
CMD ["python", "enhanced_smart_dispatcher.py"]