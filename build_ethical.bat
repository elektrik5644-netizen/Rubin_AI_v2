@echo off
REM Скрипт для сборки и запуска Ethical Core в Docker (Windows)

echo 🚀 Сборка Ethical Core Docker контейнера...

REM Остановка существующих контейнеров
echo 🛑 Остановка существующих контейнеров...
docker-compose -f docker-compose.ethical.yml down

REM Сборка образа
echo 🔨 Сборка Docker образа...
docker-compose -f docker-compose.ethical.yml build --no-cache

REM Запуск контейнеров
echo ▶️ Запуск Ethical Core...
docker-compose -f docker-compose.ethical.yml up -d

REM Ожидание запуска
echo ⏳ Ожидание запуска сервисов...
timeout /t 10 /nobreak > nul

REM Проверка статуса
echo 📊 Проверка статуса...
docker-compose -f docker-compose.ethical.yml ps

REM Проверка здоровья
echo 🏥 Проверка здоровья сервисов...
curl -f http://localhost:8105/api/health || echo ❌ Ethical Core не отвечает

echo ✅ Готово! Ethical Core доступен на http://localhost:8105
pause


