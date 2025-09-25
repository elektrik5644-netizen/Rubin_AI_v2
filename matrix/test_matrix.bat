@echo off
echo 🧪 Тестирование Rubin AI Matrix...

REM Функция для тестирования эндпоинта
:test_endpoint
set url=%1
set expected_status=%2
set description=%3

echo Тестирование: %description%
for /f %%i in ('curl -s -o nul -w "%%{http_code}" "%url%"') do set response=%%i

if "%response%"=="%expected_status%" (
    echo ✅ %description% - OK (%response%)
) else (
    echo ❌ %description% - FAILED (%response%, ожидалось %expected_status%)
)
goto :eof

REM Тестирование основных эндпоинтов
call :test_endpoint "http://localhost:8083/health" "200" "Gateway Health Check"
call :test_endpoint "http://localhost:8083/" "200" "Gateway Root"
call :test_endpoint "http://localhost:6333/health" "200" "Qdrant Health Check"
call :test_endpoint "http://localhost:11434/api/tags" "200" "Ollama API"

REM Тестирование базы данных
echo Тестирование подключения к PostgreSQL...
docker-compose exec postgres_db pg_isready -U rubin

REM Тестирование AI чата
echo Тестирование AI чата...
curl -X POST "http://localhost:8083/api/chat" -H "Content-Type: application/json" -d "{\"message\": \"Привет, как дела?\"}" -s

REM Тестирование анализа кода
echo Тестирование анализа кода...
curl -X POST "http://localhost:8083/api/code/analyze" -H "Content-Type: application/json" -d "{\"code\": \"print(\\\"Hello, World!\\\")\", \"language\": \"python\"}" -s

REM Тестирование статуса матрицы
echo Тестирование статуса матрицы...
curl -X GET "http://localhost:8083/api/matrix/status" -s

echo 🎉 Тестирование завершено!
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
