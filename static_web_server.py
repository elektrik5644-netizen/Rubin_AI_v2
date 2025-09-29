#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой статический веб-сервер для Rubin AI интерфейсов
"""

import http.server
import socketserver
import os
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="matrix", **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_static_server():
    """Запуск статического веб-сервера"""
    PORT = 8085
    
    print(f"Запуск статического веб-сервера на порту {PORT}")
    print(f"Корневая папка: matrix/")
    print(f"Доступные интерфейсы:")
    print(f"  - RubinIDE: http://localhost:{PORT}/RubinIDE.html")
    print(f"  - RubinDeveloper: http://localhost:{PORT}/RubinDeveloper.html")
    print(f"\nНажмите Ctrl+C для остановки")
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nСтатический сервер остановлен")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    start_static_server()























