# PowerShell скрипт для остановки всех серверов Rubin AI
# Автор: Rubin AI Assistant
# Дата: 28.09.2025

Write-Host "🛑 Остановка системы Rubin AI..." -ForegroundColor Red

# Остановка Python процессов (если есть)
Write-Host "`n🐍 Остановка Python серверов..." -ForegroundColor Yellow

$pythonProcesses = Get-Process python3.13 -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "Найдено $($pythonProcesses.Count) Python процессов" -ForegroundColor Cyan
    
    foreach ($process in $pythonProcesses) {
        try {
            Write-Host "🔄 Остановка процесса PID: $($process.Id)" -ForegroundColor Yellow
            Stop-Process -Id $process.Id -Force
            Write-Host "✅ Процесс $($process.Id) остановлен" -ForegroundColor Green
        }
        catch {
            Write-Host "❌ Не удалось остановить процесс $($process.Id)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "ℹ️ Python процессы не найдены" -ForegroundColor Cyan
}

# Остановка всех Docker контейнеров Rubin AI
Write-Host "`n🐳 Остановка Docker контейнеров..." -ForegroundColor Yellow

# Smart Dispatcher
Write-Host "Stopping Smart Dispatcher..." -ForegroundColor Yellow
docker-compose -f docker-compose.smart-dispatcher.yml down

# Math Server
Write-Host "Stopping Math Server..." -ForegroundColor Yellow
docker-compose -f docker-compose.math.yml down

# General Server
Write-Host "Stopping General Server..." -ForegroundColor Yellow
docker-compose -f docker-compose.general.yml down

       # Programming Server
       Write-Host "Stopping Programming Server..." -ForegroundColor Yellow
       docker-compose -f docker-compose.programming.yml down
       
       # GAI Server
       Write-Host "Stopping GAI Server..." -ForegroundColor Yellow
       docker-compose -f docker-compose.gai.yml down

# Electrical Server
Write-Host "Stopping Electrical Server..." -ForegroundColor Yellow
docker-compose -f docker-compose.electrical.yml down

# Controllers Server
Write-Host "Stopping Controllers Server..." -ForegroundColor Yellow
docker-compose down

# Telegram Bot
Write-Host "Stopping Telegram Bot..." -ForegroundColor Yellow
docker-compose -f docker-compose.telegram.yml down

# Проверка остановленных контейнеров
$containers = docker ps -a --format "table {{.Names}}\t{{.Status}}" | Select-String "rubin"
if ($containers) {
    Write-Host "`n📋 Статус контейнеров Rubin AI:" -ForegroundColor Cyan
    Write-Host $containers -ForegroundColor White
} else {
    Write-Host "✅ Все контейнеры Rubin AI остановлены" -ForegroundColor Green
}

# Очистка неиспользуемых ресурсов
Write-Host "`n🧹 Очистка неиспользуемых ресурсов..." -ForegroundColor Yellow
docker system prune -f

Write-Host "`n✅ Все серверы остановлены!" -ForegroundColor Green
Write-Host "🔄 Для запуска всех серверов запустите start_docker_system.ps1" -ForegroundColor Cyan
