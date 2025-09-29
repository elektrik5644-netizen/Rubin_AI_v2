# Stop all Python processes
Get-Process python3.13 -ErrorAction SilentlyContinue | Stop-Process -Force

# Stop Docker containers
docker-compose -f docker-compose.smart-dispatcher.yml down
docker-compose -f docker-compose.electrical.yml down
docker-compose down

# Set token
$env:TELEGRAM_BOT_TOKEN = "8126465863:AAHDrnHWaDGzwmwDDWTOYBw8TLcxpr6jOns"

# Start Docker containers
docker-compose -f docker-compose.smart-dispatcher.yml up -d
docker-compose -f docker-compose.electrical.yml up -d
docker-compose up -d

# Start local servers
Start-Process -FilePath "python" -ArgumentList "math_server.py" -WindowStyle Hidden
Start-Process -FilePath "python" -ArgumentList "general_server.py" -WindowStyle Hidden  
Start-Process -FilePath "python" -ArgumentList "programming_server.py" -WindowStyle Hidden
Start-Process -FilePath "python" -ArgumentList "telegram_bot.py" -WindowStyle Hidden

Write-Host "All servers restarted!" -ForegroundColor Green
Write-Host "Telegram bot should be working now." -ForegroundColor Cyan
