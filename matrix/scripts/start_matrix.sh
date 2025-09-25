#!/bin/bash

echo "🚀 Запуск Rubin AI Matrix..."

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker и попробуйте снова."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен. Установите Docker Compose и попробуйте снова."
    exit 1
fi

# Создание необходимых директорий
echo "📁 Создание директорий..."
mkdir -p data/{qdrant,postgres,ollama,logs,cpp}

# Проверка файла .env
if [ ! -f .env ]; then
    echo "❌ Файл .env не найден. Создайте его на основе env_example.txt"
    exit 1
fi

# Запуск сервисов
echo "🐳 Запуск Docker контейнеров..."
docker-compose up -d

# Ожидание готовности сервисов
echo "⏳ Ожидание готовности сервисов..."
sleep 30

# Проверка статуса
echo "📊 Проверка статуса сервисов..."
docker-compose ps

# Загрузка моделей в Ollama
echo "🤖 Загрузка AI моделей..."
docker-compose exec ollama_service ollama pull phi3

echo "✅ Rubin AI Matrix запущена!"
echo "🌐 Веб-интерфейс: http://localhost:8083"
echo "📊 API документация: http://localhost:8083/docs"
echo "🔍 Qdrant: http://localhost:6333"
echo "🤖 Ollama: http://localhost:11434"
