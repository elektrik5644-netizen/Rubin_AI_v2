@echo off
title Rubin AI Server
color 0A
echo.
echo ========================================
echo    🚀 RUBIN AI SERVER LAUNCHER
echo ========================================
echo.
echo 📁 Директория: %CD%
echo 🌐 Порт: 8083
echo.
echo ⏳ Запуск сервера...
echo.
python minimal_rubin_server.py
echo.
echo 🛑 Сервер остановлен
echo.
pause
