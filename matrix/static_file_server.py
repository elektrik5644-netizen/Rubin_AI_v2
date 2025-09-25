#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой HTTP сервер для обслуживания статических файлов
"""

import http.server
import socketserver
import os
import sys

class StaticFileHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def run_static_server(port=8084):
    """Запуск статического сервера"""
    print(f"🌐 Запуск статического сервера на порту {port}...")
    print(f"📁 Директория: {os.path.dirname(os.path.abspath(__file__))}")
    print("=" * 60)
    
    try:
        with socketserver.TCPServer(("", port), StaticFileHandler) as httpd:
            print(f"🎉 Статический сервер запущен на http://localhost:{port}")
            print(f"📋 Доступные файлы:")
            print(f"   http://localhost:{port}/Smart_Rubin_AI_Advanced_Interface.html")
            print(f"   http://localhost:{port}/rubin_code_diagnostic_ui.html")
            print(f"   http://localhost:{port}/rubin_editor_control_ui.html")
            print(f"   http://localhost:{port}/rubin_code_auto_fixer_ui.html")
            print("\n🛑 Для остановки нажмите Ctrl+C")
            print("=" * 60)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Остановка статического сервера...")
        print("✅ Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка сервера: {e}")

if __name__ == "__main__":
    run_static_server()
