@echo off
echo 🚀 Запуск Rubin AI Matrix...

REM Проверка наличия Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker не установлен. Установите Docker и попробуйте снова.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose не установлен. Установите Docker Compose и попробуйте снова.
    pause
    exit /b 1
)

REM Создание необходимых директорий
echo 📁 Создание директорий...
if not exist "data" mkdir data
if not exist "data\qdrant" mkdir data\qdrant
if not exist "data\postgres" mkdir data\postgres
if not exist "data\ollama" mkdir data\ollama
if not exist "data\logs" mkdir data\logs
if not exist "data\cpp" mkdir data\cpp

REM Проверка файла .env
if not exist ".env" (
    echo ❌ Файл .env не найден. Создайте его на основе env_example.txt
    pause
    exit /b 1
)

REM Запуск сервисов
echo 🐳 Запуск Docker контейнеров...
docker-compose up -d

REM Ожидание готовности сервисов
echo ⏳ Ожидание готовности сервисов...
timeout /t 30 /nobreak >nul

REM Проверка статуса
echo 📊 Проверка статуса сервисов...
docker-compose ps

REM Загрузка моделей в Ollama
echo 🤖 Загрузка AI моделей...
docker-compose exec ollama_service ollama pull phi3

echo ✅ Rubin AI Matrix запущена!
echo 🌐 Веб-интерфейс: http://localhost:8083
echo 📊 API документация: http://localhost:8083/docs
echo 🔍 Qdrant: http://localhost:6333
echo 🤖 Ollama: http://localhost:11434
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
