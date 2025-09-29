#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 MINIMAL STABLE SYSTEM для Rubin AI v2
Минимальный набор серверов для стабильной работы
"""

import subprocess
import time
import requests
import os
import signal
import sys

class MinimalStableSystem:
    def __init__(self):
        self.processes = {}
        # Только критически важные серверы
        self.minimal_servers = {
            'smart_dispatcher': {
                'command': ['python', 'simple_dispatcher.py'],
                'port': 8080,
                'critical': True
            },
            'general_api': {
                'command': ['python', 'api/general_api.py'],
                'port': 8085,
                'critical': True
            },
            'mathematics_api': {
                'command': ['python', 'api/mathematics_api.py'],
                'port': 8086,
                'critical': True
            },
            'electrical_api': {
                'command': ['python', 'api/electrical_api.py'],
                'port': 8087,
                'critical': True
            },
            'programming_api': {
                'command': ['python', 'api/programming_api.py'],
                'port': 8088,
                'critical': True
            },
            'radiomechanics_api': {
                'command': ['python', 'api/radiomechanics_api.py'],
                'port': 8089,
                'critical': True
            },
            'neuro_api': {
                'command': ['python', 'simple_neuro_api_server.py'],
                'port': 8090,
                'critical': True
            }
        }
    
    def kill_all_python_processes(self):
        """Убиваем все Python процессы"""
        print("🔄 Останавливаем все Python процессы...")
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                             capture_output=True, text=True)
            else:  # Linux/Mac
                subprocess.run(['pkill', '-f', 'python'], 
                             capture_output=True, text=True)
            time.sleep(3)
            print("✅ Все процессы остановлены")
        except Exception as e:
            print(f"⚠️ Ошибка при остановке процессов: {e}")
    
    def start_server(self, name, config):
        """Запускаем один сервер"""
        try:
            print(f"🚀 Запускаем {name} на порту {config['port']}...")
            process = subprocess.Popen(
                config['command'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes[name] = process
            time.sleep(2)  # Даем время на запуск
            print(f"✅ {name} запущен (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"❌ Ошибка запуска {name}: {e}")
            return False
    
    def check_server_health(self, name, config):
        """Проверяем здоровье сервера"""
        try:
            if name == 'smart_dispatcher':
                url = f"http://localhost:{config['port']}/api/chat"
                response = requests.post(url, json={'message': 'test'}, timeout=3)
            else:
                url = f"http://localhost:{config['port']}/api/health"
                response = requests.get(url, timeout=3)
            
            return response.status_code in [200, 201]
        except:
            return False
    
    def run(self):
        """Основной метод запуска"""
        print("🎯 MINIMAL STABLE SYSTEM для Rubin AI v2")
        print("=" * 50)
        
        # Останавливаем все процессы
        self.kill_all_python_processes()
        
        # Запускаем минимальный набор серверов
        print("🔥 Запускаем минимальный набор серверов...")
        success_count = 0
        
        for name, config in self.minimal_servers.items():
            if self.start_server(name, config):
                success_count += 1
        
        print(f"✅ Запущено {success_count}/{len(self.minimal_servers)} серверов")
        
        # Ждем стабилизации
        print("⏳ Ждем стабилизации системы...")
        time.sleep(5)
        
        # Проверяем здоровье
        print("🔍 Проверяем здоровье системы...")
        healthy_count = 0
        
        for name, config in self.minimal_servers.items():
            if self.check_server_health(name, config):
                healthy_count += 1
                print(f"✅ {name} - здоров")
            else:
                print(f"❌ {name} - не отвечает")
        
        print(f"📊 Результат: {healthy_count}/{len(self.minimal_servers)} серверов здоровы")
        
        if healthy_count >= len(self.minimal_servers) * 0.8:
            print("🎉 МИНИМАЛЬНАЯ СИСТЕМА ГОТОВА К РАБОТЕ!")
            return True
        else:
            print("⚠️ СИСТЕМА ТРЕБУЕТ ВНИМАНИЯ!")
            return False

def main():
    system = MinimalStableSystem()
    
    try:
        success = system.run()
        if success:
            print("🚀 Система запущена. Нажмите Ctrl+C для остановки.")
            # Держим систему запущенной
            while True:
                time.sleep(1)
        else:
            print("❌ Запуск системы не удался")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Остановка системы...")
        system.kill_all_python_processes()
        sys.exit(0)

if __name__ == "__main__":
    main()








