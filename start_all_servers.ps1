# PowerShell script for starting all Rubin AI servers
# Author: Rubin AI Assistant
# Date: 27.09.2025

Write-Host "Starting Rubin AI system..." -ForegroundColor Green

# Set environment variables
$env:TELEGRAM_BOT_TOKEN = "8126465863:AAHDrnHWaDGzwmwDDWTOYBw8TLcxpr6jOns"
Write-Host "TELEGRAM_BOT_TOKEN set" -ForegroundColor Green

# Function to start server in background
function Start-Server {
    param(
        [string]$Name,
        [string]$Command,
        [string]$Port
    )
    
    Write-Host "Starting $Name on port $Port..." -ForegroundColor Yellow
    
    # Start in background
    Start-Process -FilePath "python" -ArgumentList $Command -WindowStyle Hidden
    
    # Wait for startup
    Start-Sleep -Seconds 3
    
    # Check status
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$Port/api/health" -Method GET -TimeoutSec 5
        Write-Host "$Name started successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "$Name failed to start" -ForegroundColor Red
        return $false
    }
}

# Start servers in order
Write-Host "`nStarting main servers..." -ForegroundColor Cyan

# 1. Math Server (port 8086)
Start-Server -Name "Math Server" -Command "math_server.py" -Port "8086"

# 2. General Server (port 8085)
Start-Server -Name "General Server" -Command "general_server.py" -Port "8085"

# 3. Programming Server (port 8088)
Start-Server -Name "Programming Server" -Command "programming_server.py" -Port "8088"

# 4. Smart Dispatcher (Docker)
Write-Host "Starting Smart Dispatcher (Docker)..." -ForegroundColor Yellow
docker-compose -f docker-compose.smart-dispatcher.yml up -d
Start-Sleep -Seconds 5
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/api/health" -Method GET -TimeoutSec 5
    Write-Host "Smart Dispatcher (Docker) started successfully" -ForegroundColor Green
}
catch {
    Write-Host "Smart Dispatcher (Docker) failed to start" -ForegroundColor Red
}

# 5. Telegram Bot
Write-Host "`nStarting Telegram Bot..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "telegram_bot.py" -WindowStyle Hidden
Start-Sleep -Seconds 5
Write-Host "Telegram Bot started" -ForegroundColor Green

# Check Docker containers
Write-Host "`nChecking Docker containers..." -ForegroundColor Cyan

# Check Electrical Server
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8087/api/health" -Method GET -TimeoutSec 5
    Write-Host "Electrical Server (Docker) is working" -ForegroundColor Green
}
catch {
    Write-Host "Electrical Server (Docker) is not working" -ForegroundColor Red
    Write-Host "Starting Electrical Server..." -ForegroundColor Yellow
    docker-compose -f docker-compose.electrical.yml up -d
}

# Check Controllers Server
try {
    $response = Invoke-RestMethod -Uri "http://localhost:9000/api/health" -Method GET -TimeoutSec 5
    Write-Host "Controllers Server (Docker) is working" -ForegroundColor Green
}
catch {
    Write-Host "Controllers Server (Docker) is not working" -ForegroundColor Red
    Write-Host "Starting Controllers Server..." -ForegroundColor Yellow
    docker-compose up -d
}

# Final check
Write-Host "`nFinal system check..." -ForegroundColor Cyan

$servers = @(
    @{Name="Smart Dispatcher"; Port="8080"},
    @{Name="General Server"; Port="8085"},
    @{Name="Math Server"; Port="8086"},
    @{Name="Electrical Server"; Port="8087"},
    @{Name="Programming Server"; Port="8088"},
    @{Name="Controllers Server"; Port="9000"}
)

$workingServers = 0
foreach ($server in $servers) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$($server.Port)/api/health" -Method GET -TimeoutSec 3
        Write-Host "$($server.Name) (port $($server.Port)) - working" -ForegroundColor Green
        $workingServers++
    }
    catch {
        Write-Host "$($server.Name) (port $($server.Port)) - not working" -ForegroundColor Red
    }
}

Write-Host "`nStatistics: $workingServers out of $($servers.Count) servers are working" -ForegroundColor Cyan

if ($workingServers -eq $servers.Count) {
    Write-Host "All servers started successfully!" -ForegroundColor Green
    Write-Host "`nTelegram Bot is ready!" -ForegroundColor Green
    Write-Host "Try writing to the bot in Telegram: 'hello'" -ForegroundColor Yellow
} else {
    Write-Host "Some servers failed to start. Check logs." -ForegroundColor Yellow
}

Write-Host "`nTo restart all servers, run this script again." -ForegroundColor Cyan
Write-Host "To stop all servers, run stop_all_servers.ps1" -ForegroundColor Cyan