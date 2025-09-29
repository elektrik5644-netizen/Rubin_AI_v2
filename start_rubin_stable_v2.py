#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Стабильный запуск Rubin AI v2.0 с исправлением проблем
"""

import os
import sys
import time
import subprocess
import signal
import psutil
import socket
from pathlib import Path

class RubinStarter:
    def __init__(self):
        self.processes = []
        self.ports = {
            'ai_chat': 8084,
            'static_web': 8085,
            'electrical': 8087,
            'documents': 8088,
            'radiomechanics': 8089,
            'controllers': 8090
        }
    
    def is_port_in_use(self, port):
        """Проверяет, занят ли порт"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except:
            return False
    
    def kill_process_on_port(self, port):
        """Убивает процесс на указанном порту"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    if proc.info['connections']:
                        for conn in proc.info['connections']:
                            if conn.laddr.port == port:
                                print(f"Найден процесс {proc.info['name']} (PID: {proc.info['pid']}) на порту {port}")
                                proc.kill()
                                print(f"Процесс {proc.info['pid']} завершен")
                                time.sleep(1)
                                return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            print(f"Ошибка при освобождении порта {port}: {e}")
        return False
    
    def kill_all_python_processes(self):
        """Убивает все процессы Python (осторожно!)"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        print(f"Завершение процесса Python: {proc.info['name']} (PID: {proc.info['pid']})")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            time.sleep(2)
        except Exception as e:
            print(f"Ошибка при завершении процессов Python: {e}")
    
    def start_server(self, name, script_path, port):
        """Запускает сервер"""
        try:
            if self.is_port_in_use(port):
                print(f"Порт {port} занят, освобождаем...")
                self.kill_process_on_port(port)
                time.sleep(1)
            
            if self.is_port_in_use(port):
                print(f"Не удалось освободить порт {port}, пропускаем {name}")
                return False
            
            print(f"Запуск {name} на порту {port}...")
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.processes.append((name, process, port))
            time.sleep(2)  # Даем время на запуск
            
            if self.is_port_in_use(port):
                print(f"✅ {name} запущен (PID: {process.pid})")
                return True
            else:
                print(f"❌ {name} не запустился")
                return False
                
        except Exception as e:
            print(f"Ошибка запуска {name}: {e}")
            return False
    
    def start_all_servers(self):
        """Запускает все серверы"""
        print("🎯 СТАБИЛЬНЫЙ ЗАПУСК RUBIN AI v2.0")
        print("=" * 60)
        
        # Сначала убиваем все Python процессы
        print("🔧 Очистка процессов...")
        self.kill_all_python_processes()
        time.sleep(3)
        
        # Запускаем серверы
        servers = [
            ("AI Чат (Основной)", "api/rubin_ai_v2_server.py", 8084),
            ("Электротехника", "api/electrical_api.py", 8087),
            ("Радиомеханика", "api/radiomechanics_api.py", 8089),
            ("Контроллеры", "api/controllers_api.py", 8090),
            ("Документы", "api/documents_api.py", 8088),
            ("Статический Веб-сервер", "static_web_server.py", 8085)
        ]
        
        successful_servers = []
        
        for name, script, port in servers:
            if self.start_server(name, script, port):
                successful_servers.append((name, port))
        
        return successful_servers
    
    def check_server_status(self, port):
        """Проверяет статус сервера"""
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def show_status(self, successful_servers):
        """Показывает статус серверов"""
        print("\n" + "=" * 60)
        print("📊 СТАТУС ЗАПУЩЕННЫХ СЕРВЕРОВ:")
        print("=" * 60)
        
        for name, port in successful_servers:
            if self.check_server_status(port):
                print(f"✅ {name}: ОНЛАЙН (порт {port})")
            else:
                print(f"❌ {name}: ОФФЛАЙН (порт {port})")
        
        print("\n🌐 ДОСТУПНЫЕ ИНТЕРФЕЙСЫ:")
        print("   🤖 AI Чат: http://localhost:8084/RubinIDE.html")
        print("   ⚙️ Developer: http://localhost:8084/RubinDeveloper.html")
        print("   📊 Статус: http://localhost:8084/status_check.html")
        print("   📚 Документы: http://localhost:8088/DocumentsManager.html")
        print("   🌐 Статический: http://localhost:8085/RubinIDE.html")
        
        print("\n🔍 СПЕЦИАЛИЗИРОВАННЫЕ API:")
        print("   ⚡ Электротехника: http://localhost:8087/api/electrical/status")
        print("   📡 Радиомеханика: http://localhost:8089/api/radiomechanics/status")
        print("   🎛️ Контроллеры: http://localhost:8090/api/controllers/status")
        print("   📚 Документы API: http://localhost:8088/health")
    
    def cleanup(self):
        """Очистка при завершении"""
        print("\n🛑 Завершение работы...")
        for name, process, port in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} завершен")
            except:
                try:
                    process.kill()
                    print(f"🔨 {name} принудительно завершен")
                except:
                    print(f"❌ Не удалось завершить {name}")
    
    def run(self):
        """Основной метод запуска"""
        try:
            successful_servers = self.start_all_servers()
            
            if not successful_servers:
                print("❌ Не удалось запустить ни одного сервера!")
                return False
            
            self.show_status(successful_servers)
            
            print("\n⏳ Нажмите Ctrl+C для остановки всех серверов")
            
            # Ожидание завершения
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Главная функция"""
    starter = RubinStarter()
    success = starter.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()






















