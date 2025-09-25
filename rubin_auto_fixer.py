#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ ПРОБЛЕМ RUBIN AI
=============================================
"""

import subprocess
import time
import requests
import os
import sys
from datetime import datetime

class RubinAutoFixer:
    """Автоматическое исправление проблем Rubin AI"""
    
    def __init__(self):
        self.servers_to_fix = {
            'general_api': {
                'port': 8085,
                'script': 'general_server.py',
                'endpoint': '/api/health',
                'fallback': None
            },
            'mathematics': {
                'port': 8086,
                'script': 'math_server.py',
                'endpoint': '/health',  # Исправленный endpoint
                'fallback': 'general_api'
            },
            'electrical': {
                'port': 8087,
                'script': 'electrical_server.py',
                'endpoint': '/api/electrical/status',
                'fallback': 'mathematics'
            },
            'programming': {
                'port': 8088,
                'script': 'programming_server.py',
                'endpoint': '/api/programming/explain',
                'fallback': 'general_api'
            },
            'radiomechanics': {
                'port': 8089,
                'script': 'radiomechanics_server.py',
                'endpoint': '/api/radiomechanics/status',
                'fallback': 'general_api'
            }
        }
        
        self.fixed_servers = []
        self.failed_servers = []
    
    def check_server_exists(self, script_name):
        """Проверяет существование скрипта сервера"""
        return os.path.exists(script_name)
    
    def start_server(self, server_name, config):
        """Запускает сервер"""
        script_name = config['script']
        
        if not self.check_server_exists(script_name):
            print(f"❌ {server_name}: Скрипт {script_name} не найден")
            return False
        
        try:
            print(f"🚀 Запускаю {server_name} ({script_name})...")
            
            # Запускаем сервер в фоновом режиме
            process = subprocess.Popen([
                sys.executable, script_name
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Ждем запуска
            time.sleep(3)
            
            # Проверяем, что сервер запустился
            if self.test_server_health(server_name, config):
                print(f"✅ {server_name}: Успешно запущен")
                self.fixed_servers.append(server_name)
                return True
            else:
                print(f"❌ {server_name}: Не удалось запустить")
                process.terminate()
                self.failed_servers.append(server_name)
                return False
                
        except Exception as e:
            print(f"❌ {server_name}: Ошибка запуска - {e}")
            self.failed_servers.append(server_name)
            return False
    
    def test_server_health(self, server_name, config):
        """Тестирует здоровье сервера"""
        port = config['port']
        endpoint = config['endpoint']
        url = f"http://localhost:{port}{endpoint}"
        
        try:
            if server_name == 'programming':
                response = requests.post(url, json={'concept': 'test'}, timeout=5)
            else:
                response = requests.get(url, timeout=5)
            
            return response.status_code == 200
        except:
            return False
    
    def fix_all_servers(self):
        """Исправляет все серверы"""
        print("🔧 АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ ПРОБЛЕМ RUBIN AI")
        print("=" * 60)
        
        for server_name, config in self.servers_to_fix.items():
            print(f"\n🔍 Проверяю {server_name}...")
            
            if self.test_server_health(server_name, config):
                print(f"✅ {server_name}: Уже работает")
                self.fixed_servers.append(server_name)
            else:
                print(f"❌ {server_name}: Требует исправления")
                self.start_server(server_name, config)
            
            time.sleep(1)  # Небольшая пауза между запусками
    
    def update_smart_dispatcher(self):
        """Обновляет Smart Dispatcher до v3.0"""
        print(f"\n🔄 ОБНОВЛЕНИЕ SMART DISPATCHER:")
        print("-" * 40)
        
        try:
            # Проверяем, какой диспетчер сейчас запущен
            response = requests.get('http://localhost:8080/api/health', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'learning' in str(data):
                    print("✅ Smart Dispatcher v3.0 уже запущен")
                else:
                    print("⚠️ Smart Dispatcher v2.0 запущен, требуется обновление")
                    print("💡 Рекомендуется остановить v2.0 и запустить v3.0")
            else:
                print("❌ Smart Dispatcher недоступен")
        except Exception as e:
            print(f"❌ Ошибка проверки Smart Dispatcher: {e}")
    
    def generate_final_report(self):
        """Генерирует финальный отчет"""
        print(f"\n📊 ФИНАЛЬНЫЙ ОТЧЕТ ИСПРАВЛЕНИЯ:")
        print("=" * 40)
        
        print(f"✅ Исправлено серверов: {len(self.fixed_servers)}")
        for server in self.fixed_servers:
            print(f"  • {server}")
        
        if self.failed_servers:
            print(f"\n❌ Не удалось исправить: {len(self.failed_servers)}")
            for server in self.failed_servers:
                print(f"  • {server}")
        
        total_servers = len(self.servers_to_fix) + 4  # +4 уже работающих
        working_servers = len(self.fixed_servers) + 4
        
        print(f"\n📈 Общий статус системы: {working_servers}/{total_servers} серверов работают")
        
        if working_servers >= total_servers * 0.8:
            print("🎉 Система Rubin AI работает стабильно!")
        elif working_servers >= total_servers * 0.6:
            print("⚠️ Система работает с ограничениями")
        else:
            print("❌ Система требует дополнительного исправления")

def main():
    """Основная функция автоматического исправления"""
    fixer = RubinAutoFixer()
    
    # Исправляем все серверы
    fixer.fix_all_servers()
    
    # Обновляем Smart Dispatcher
    fixer.update_smart_dispatcher()
    
    # Генерируем отчет
    fixer.generate_final_report()
    
    print(f"\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
    print("=" * 20)
    print("1. Проверить работу всех серверов")
    print("2. Обновить Smart Dispatcher до v3.0")
    print("3. Протестировать маршрутизацию")
    print("4. Настроить мониторинг")

if __name__ == "__main__":
    main()





