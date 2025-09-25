#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ULTRA SIMPLE STABLE LAUNCHER для Rubin AI v2
Максимально упрощенный запуск только критически важных серверов
"""

import subprocess
import time
import requests
import os
import sys

class UltraSimpleLauncher:
    def __init__(self):
        # Только самые критически важные серверы
        self.critical_servers = [
            {
                'name': 'smart_dispatcher',
                'command': ['python', 'simple_dispatcher.py'],
                'port': 8080,
                'health_url': 'http://localhost:8080/api/chat'
            },
            {
                'name': 'general_api',
                'command': ['python', 'api/general_api.py'],
                'port': 8085,
                'health_url': 'http://localhost:8085/api/health'
            },
            {
                'name': 'mathematics_api',
                'command': ['python', 'api/mathematics_api.py'],
                'port': 8086,
                'health_url': 'http://localhost:8086/health'
            },
            {
                'name': 'electrical_api',
                'command': ['python', 'api/electrical_api.py'],
                'port': 8087,
                'health_url': 'http://localhost:8087/api/electrical/status'
            },
            {
                'name': 'programming_api',
                'command': ['python', 'api/programming_api.py'],
                'port': 8088,
                'health_url': 'http://localhost:8088/api/health'
            },
            {
                'name': 'radiomechanics_api',
                'command': ['python', 'api/radiomechanics_api.py'],
                'port': 8089,
                'health_url': 'http://localhost:8089/api/radiomechanics/status'
            },
            {
                'name': 'neuro_api',
                'command': ['python', 'simple_neuro_api_server.py'],
                'port': 8090,
                'health_url': 'http://localhost:8090/api/health'
            }
        ]
        
        self.processes = {}
    
    def kill_all_python(self):
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
            print(f"⚠️ Ошибка: {e}")
    
    def start_server(self, server):
        """Запускаем один сервер"""
        try:
            print(f"🚀 Запускаем {server['name']} на порту {server['port']}...")
            process = subprocess.Popen(
                server['command'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes[server['name']] = process
            time.sleep(2)  # Даем время на запуск
            print(f"✅ {server['name']} запущен (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"❌ Ошибка запуска {server['name']}: {e}")
            return False
    
    def check_health(self, server):
        """Проверяем здоровье сервера"""
        try:
            if server['name'] == 'smart_dispatcher':
                response = requests.post(server['health_url'], 
                                       json={'message': 'test'}, timeout=3)
            else:
                response = requests.get(server['health_url'], timeout=3)
            
            return response.status_code in [200, 201]
        except:
            return False
    
    def run(self):
        """Основной метод запуска"""
        print("🎯 ULTRA SIMPLE STABLE LAUNCHER для Rubin AI v2")
        print("=" * 60)
        
        # Останавливаем все процессы
        self.kill_all_python()
        
        # Запускаем критически важные серверы
        print("🔥 Запускаем критически важные серверы...")
        success_count = 0
        
        for server in self.critical_servers:
            if self.start_server(server):
                success_count += 1
        
        print(f"✅ Запущено {success_count}/{len(self.critical_servers)} серверов")
        
        # Ждем стабилизации
        print("⏳ Ждем стабилизации системы...")
        time.sleep(5)
        
        # Проверяем здоровье
        print("🔍 Проверяем здоровье системы...")
        healthy_count = 0
        
        for server in self.critical_servers:
            if self.check_health(server):
                healthy_count += 1
                print(f"✅ {server['name']} - здоров")
            else:
                print(f"❌ {server['name']} - не отвечает")
        
        print(f"📊 Результат: {healthy_count}/{len(self.critical_servers)} серверов здоровы")
        
        if healthy_count >= len(self.critical_servers) * 0.7:  # 70% серверов здоровы
            print("🎉 СИСТЕМА ГОТОВА К РАБОТЕ!")
            return True
        else:
            print("⚠️ СИСТЕМА ТРЕБУЕТ ВНИМАНИЯ!")
            return False

def main():
    launcher = UltraSimpleLauncher()
    
    try:
        success = launcher.run()
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
        launcher.kill_all_python()
        sys.exit(0)

if __name__ == "__main__":
    main()



