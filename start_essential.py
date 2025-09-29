#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск основных модулей Rubin AI v2.0
"""

import os
import sys
import subprocess
import time
import signal
import atexit

class RubinStarter:
    def __init__(self):
        self.processes = []
        
    def start_module(self, name, command, port):
        """Запуск модуля"""
        try:
            print(f"🚀 Запуск {name} на порту {port}...")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='ignore',
                env=env
            )
            
            self.processes.append(process)
            print(f"✅ {name} запущен (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка запуска {name}: {e}")
            return False
    
    def cleanup(self):
        """Очистка при выходе"""
        print("\n🛑 Остановка всех модулей...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print("✅ Все модули остановлены")
    
    def start_all(self):
        """Запуск всех модулей"""
        print("🎯 ЗАПУСК ОСНОВНЫХ МОДУЛЕЙ RUBIN AI v2.0")
        print("=" * 50)
        
        # Регистрируем обработчик для корректного завершения
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup())
        signal.signal(signal.SIGTERM, lambda s, f: self.cleanup())
        
        modules = [
            {
                'name': 'AI Чат (Основной сервер)',
                'command': [sys.executable, 'api/rubin_ai_v2_server.py'],
                'port': 8084
            },
            {
                'name': 'Электротехника',
                'command': [sys.executable, 'api/electrical_api.py'],
                'port': 8087
            },
            {
                'name': 'Радиомеханика',
                'command': [sys.executable, 'api/radiomechanics_api.py'],
                'port': 8089
            },
            {
                'name': 'Контроллеры',
                'command': [sys.executable, 'api/controllers_api.py'],
                'port': 8090
            }
        ]
        
        success_count = 0
        
        for module in modules:
            if self.start_module(module['name'], module['command'], module['port']):
                success_count += 1
            time.sleep(2)
        
        print("\n" + "=" * 50)
        print(f"📊 Результат: {success_count}/{len(modules)} модулей запущено")
        
        if success_count > 0:
            print("\n🌐 Доступные интерфейсы:")
            print("  - AI Чат: http://localhost:8084/RubinIDE.html")
            print("  - Developer: http://localhost:8084/RubinDeveloper.html")
            print("  - Проверка статуса: http://localhost:8084/status_check.html")
            print("  - Электротехника: http://localhost:8087/api/electrical/status")
            print("  - Радиомеханика: http://localhost:8089/api/radiomechanics/status")
            print("  - Контроллеры: http://localhost:8090/api/controllers/status")
            print("\n⏳ Нажмите Ctrl+C для остановки")
            
            try:
                # Ждем завершения всех процессов
                for process in self.processes:
                    process.wait()
            except KeyboardInterrupt:
                pass
        else:
            print("❌ Не удалось запустить ни одного модуля")

if __name__ == "__main__":
    starter = RubinStarter()
    starter.start_all()























