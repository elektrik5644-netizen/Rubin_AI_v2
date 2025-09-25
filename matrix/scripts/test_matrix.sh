#!/bin/bash

echo "🧪 Тестирование Rubin AI Matrix..."

# Функция для тестирования эндпоинта
test_endpoint() {
    local url=$1
    local expected_status=$2
    local description=$3
    
    echo "Тестирование: $description"
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$response" = "$expected_status" ]; then
        echo "✅ $description - OK ($response)"
    else
        echo "❌ $description - FAILED ($response, ожидалось $expected_status)"
    fi
}

# Тестирование основных эндпоинтов
test_endpoint "http://localhost:8083/health" "200" "Gateway Health Check"
test_endpoint "http://localhost:8083/" "200" "Gateway Root"
test_endpoint "http://localhost:6333/health" "200" "Qdrant Health Check"
test_endpoint "http://localhost:11434/api/tags" "200" "Ollama API"

# Тестирование базы данных
echo "Тестирование подключения к PostgreSQL..."
docker-compose exec postgres_db pg_isready -U rubin

# Тестирование AI чата
echo "Тестирование AI чата..."
curl -X POST "http://localhost:8083/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Привет, как дела?"}' \
     -s | jq . || echo "Ошибка тестирования чата"

# Тестирование анализа кода
echo "Тестирование анализа кода..."
curl -X POST "http://localhost:8083/api/code/analyze" \
     -H "Content-Type: application/json" \
     -d '{"code": "print(\"Hello, World!\")", "language": "python"}' \
     -s | jq . || echo "Ошибка тестирования анализа кода"

# Тестирование статуса матрицы
echo "Тестирование статуса матрицы..."
curl -X GET "http://localhost:8083/api/matrix/status" \
     -s | jq . || echo "Ошибка тестирования статуса матрицы"

echo "🎉 Тестирование завершено!"
