#!/bin/bash
# Скрипт для сборки и запуска Ethical Core в Docker

echo "🚀 Сборка Ethical Core Docker контейнера..."

# Остановка существующих контейнеров
echo "🛑 Остановка существующих контейнеров..."
docker-compose -f docker-compose.ethical.yml down

# Сборка образа
echo "🔨 Сборка Docker образа..."
docker-compose -f docker-compose.ethical.yml build --no-cache

# Запуск контейнеров
echo "▶️ Запуск Ethical Core..."
docker-compose -f docker-compose.ethical.yml up -d

# Ожидание запуска
echo "⏳ Ожидание запуска сервисов..."
sleep 10

# Проверка статуса
echo "📊 Проверка статуса..."
docker-compose -f docker-compose.ethical.yml ps

# Проверка здоровья
echo "🏥 Проверка здоровья сервисов..."
curl -f http://localhost:8105/api/health || echo "❌ Ethical Core не отвечает"

echo "✅ Готово! Ethical Core доступен на http://localhost:8105"


