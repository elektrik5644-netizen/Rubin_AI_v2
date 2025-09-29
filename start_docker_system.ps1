# PowerShell script for starting the entire Rubin AI system with Docker Compose
# Author: Rubin AI Assistant
# Date: 28.09.2025

Write-Host "🚀 Starting Rubin AI System (Docker Only)..." -ForegroundColor Green

# Set environment variables
$env:TELEGRAM_BOT_TOKEN = "8126465863:AAHDrnHWaDGzwmwDDWTOYBw8TLcxpr6jOns"
Write-Host "TELEGRAM_BOT_TOKEN set" -ForegroundColor Green

# Ensure Docker network exists
Write-Host "🌐 Checking/Creating Docker network 'rubin-network'..." -ForegroundColor Yellow
try {
    docker network create rubin-network | Out-Null
    Write-Host "✅ Docker network 'rubin-network' created or already exists." -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not create Docker network 'rubin-network'. It might already exist or Docker is not running." -ForegroundColor Yellow
}

# Start Dockerized services
Write-Host "`n🐳 Starting Dockerized services..." -ForegroundColor Cyan

# Smart Dispatcher
Write-Host "Starting Smart Dispatcher (Docker)..." -ForegroundColor Yellow
docker-compose -f docker-compose.smart-dispatcher.yml up -d --build
Start-Sleep -Seconds 5

# Math Server
Write-Host "Starting Math Server (Docker)..." -ForegroundColor Yellow
docker-compose -f docker-compose.math.yml up -d --build
Start-Sleep -Seconds 5

# General Server
Write-Host "Starting General Server (Docker)..." -ForegroundColor Yellow
docker-compose -f docker-compose.general.yml up -d --build
Start-Sleep -Seconds 5

       # Programming Server
       Write-Host "Starting Programming Server (Docker)..." -ForegroundColor Yellow
       docker-compose -f docker-compose.programming.yml up -d --build
       Start-Sleep -Seconds 5
       
       # GAI Server
       Write-Host "Starting GAI Server (Docker)..." -ForegroundColor Yellow
       docker-compose -f docker-compose.gai.yml up -d --build
       Start-Sleep -Seconds 5

# Electrical Server
Write-Host "Starting Electrical Server (Docker)..." -ForegroundColor Yellow
docker-compose -f docker-compose.electrical.yml up -d --build
Start-Sleep -Seconds 5

# Controllers Server
Write-Host "Starting Controllers Server (Docker)..." -ForegroundColor Yellow
docker-compose up -d --build
Start-Sleep -Seconds 5

# Telegram Bot
Write-Host "Starting Telegram Bot (Docker)..." -ForegroundColor Yellow
docker-compose -f docker-compose.telegram.yml up -d --build
Start-Sleep -Seconds 5

Write-Host "`nChecking Docker container status..." -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-String "rubin"

Write-Host "`n✅ All Rubin AI services should now be running!" -ForegroundColor Green
Write-Host "ℹ️ You can check individual server health via their /api/health endpoints." -ForegroundColor Cyan
Write-Host "ℹ️ To stop all services, run 'powershell -ExecutionPolicy Bypass -File stop_all_servers.ps1'" -ForegroundColor Cyan