# PowerShell script for starting all new Rubin AI servers
# Author: Rubin AI Assistant
# Date: 28.09.2025

Write-Host "🚀 Starting all new Rubin AI servers..." -ForegroundColor Green

# Set environment variables
$env:TELEGRAM_BOT_TOKEN = "8126465863:AAHDrnHWaDGzwmwDDWTOYBw8TLcxpr6jOns"
Write-Host "TELEGRAM_BOT_TOKEN set" -ForegroundColor Green

# Function to start a server
function Start-Server {
    param(
        [string]$Name,
        [string]$Command,
        [string]$Port
    )
    Write-Host "Starting $Name on port $Port..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList $Command -WindowStyle Hidden
    Start-Sleep -Seconds 3
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

Write-Host "`nStarting new servers..." -ForegroundColor Cyan

$workingServers = 0

# Start all new servers
$servers = @(
    @{ Name = "PLC Analysis Server"; Command = "plc_analysis_server.py"; Port = "8099" },
    @{ Name = "Advanced Math Server"; Command = "advanced_math_server.py"; Port = "8100" },
    @{ Name = "Data Processing Server"; Command = "data_processing_server.py"; Port = "8101" },
    @{ Name = "Search Engine Server"; Command = "search_engine_server.py"; Port = "8102" },
    @{ Name = "System Utils Server"; Command = "system_utils_server.py"; Port = "8103" },
    @{ Name = "GAI Server"; Command = "gai_server.py"; Port = "8104" },
    @{ Name = "Ethical Core Server"; Command = "ethical_core_server.py"; Port = "8105" }
)

foreach ($server in $servers) {
    if (Start-Server -Name $server.Name -Command $server.Command -Port $server.Port) { 
        $workingServers++ 
    }
}

Write-Host "`nFinal system check..." -ForegroundColor Cyan

$allServers = @(
    @{ Name = "Smart Dispatcher"; Port = 8080; Type = "Docker" },
    @{ Name = "General Server"; Port = 8085; Type = "Docker" },
    @{ Name = "Math Server"; Port = 8086; Type = "Docker" },
    @{ Name = "Electrical Server"; Port = 8087; Type = "Docker" },
    @{ Name = "Programming Server"; Port = 8088; Type = "Docker" },
    @{ Name = "Neuro Server"; Port = 8090; Type = "Docker" },
    @{ Name = "Controllers Server"; Port = 9000; Type = "Docker" },
    @{ Name = "PLC Analysis Server"; Port = 8099; Type = "Local" },
    @{ Name = "Advanced Math Server"; Port = 8100; Type = "Local" },
    @{ Name = "Data Processing Server"; Port = 8101; Type = "Local" },
    @{ Name = "Search Engine Server"; Port = 8102; Type = "Local" },
    @{ Name = "System Utils Server"; Port = 8103; Type = "Local" },
    @{ Name = "GAI Server"; Port = 8104; Type = "Local" },
    @{ Name = "Ethical Core Server"; Port = 8105; Type = "Local" }
)

foreach ($server in $allServers) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$($server.Port)/api/health" -Method GET -TimeoutSec 3
        Write-Host "✅ $($server.Name) (порт $($server.Port)) - работает" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ $($server.Name) (порт $($server.Port)) - не работает" -ForegroundColor Red
    }
}

Write-Host "`nStatistics: $workingServers out of $($servers.Count) new servers are working" -ForegroundColor Yellow
Write-Host "Total system servers: $($allServers.Count)" -ForegroundColor Cyan

if ($workingServers -lt $servers.Count) {
    Write-Host "Some new servers failed to start. Check logs." -ForegroundColor Red
} else {
    Write-Host "All new servers are running!" -ForegroundColor Green
}

Write-Host "`n🎉 Rubin AI system is now complete with all servers!" -ForegroundColor Green
Write-Host "📊 Total servers: $($allServers.Count)" -ForegroundColor Cyan
Write-Host "🐳 Docker servers: $(($allServers | Where-Object { $_.Type -eq 'Docker' }).Count)" -ForegroundColor Blue
Write-Host "💻 Local servers: $(($allServers | Where-Object { $_.Type -eq 'Local' }).Count)" -ForegroundColor Magenta
