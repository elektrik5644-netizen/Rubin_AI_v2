#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 STABLE SYSTEM LAUNCHER для Rubin AI v2
Автоматический запуск всех серверов с проверкой стабильности
"""

import subprocess
import time
import requests
import threading
import os
import signal
import sys
from datetime import datetime

class StableSystemLauncher:
    def __init__(self):
        self.processes = {}
        self.servers = {
            'smart_dispatcher': {
                'command': ['python', 'simple_dispatcher.py'],
                'port': 8080,
                'health_endpoint': '/api/chat',
                'critical': True
            },
            'general_api': {
                'command': ['python', 'api/general_api.py'],
                'port': 8085,
                'health_endpoint': '/api/health',
                'critical': True
            },
            'mathematics_api': {
                'command': ['python', 'api/mathematics_api.py'],
                'port': 8086,
                'health_endpoint': '/health',
                'critical': True
            },
            'electrical_api': {
                'command': ['python', 'api/electrical_api.py'],
                'port': 8087,
                'health_endpoint': '/api/electrical/status',
                'critical': True
            },
            'programming_api': {
                'command': ['python', 'api/programming_api.py'],
                'port': 8088,
                'health_endpoint': '/api/health',
                'critical': True
            },
            'radiomechanics_api': {
                'command': ['python', 'api/radiomechanics_api.py'],
                'port': 8089,
                'health_endpoint': '/api/radiomechanics/status',
                'critical': True
            },
            'neuro_api': {
                'command': ['python', 'simple_neuro_api_server.py'],
                'port': 8090,
                'health_endpoint': '/api/health',
                'critical': True
            },
            'controllers_api': {
                'command': ['python', 'api/controllers_api.py'],
                'port': 9000,
                'health_endpoint': '/api/controllers/status',
                'critical': False
            },
            'plc_analysis': {
                'command': ['python', 'plc_analysis_api_server.py'],
                'port': 8099,
                'health_endpoint': '/api/plc/health',
                'critical': False
            },
            'advanced_math': {
                'command': ['python', 'advanced_math_api_server.py'],
                'port': 8100,
                'health_endpoint': '/api/advanced_math/health',
                'critical': False
            },
            'data_processing': {
                'command': ['python', 'data_processing_api_server.py'],
                'port': 8101,
                'health_endpoint': '/api/data_processing/health',
                'critical': False
            },
            'search_engine': {
                'command': ['python', 'search_engine_api_server.py'],
                'port': 8102,
                'health_endpoint': '/api/search/health',
                'critical': False
            },
            'system_utils': {
                'command': ['python', 'system_utils_api_server.py'],
                'port': 8103,
                'health_endpoint': '/api/system/utils/health',
                'critical': False
            },
            'gai_server': {
                'command': ['python', 'enhanced_gai_server.py'],
                'port': 8104,
                'health_endpoint': '/api/gai/health',
                'critical': False
            },
            'unified_manager': {
                'command': ['python', 'unified_system_manager.py'],
                'port': 8084,
                'health_endpoint': '/api/system/health',
                'critical': False
            },
            'ethical_core': {
                'command': ['python', 'ethical_core_api_server.py'],
                'port': 8105,
                'health_endpoint': '/api/ethical/health',
                'critical': False
            }
        }
        
    def kill_existing_processes(self):
        """Убиваем все существующие Python процессы"""
        print("🔄 Останавливаем существующие процессы...")
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                             capture_output=True, text=True)
            else:  # Linux/Mac
                subprocess.run(['pkill', '-f', 'python'], 
                             capture_output=True, text=True)
            time.sleep(2)
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
            time.sleep(3)  # Даем время на запуск
            
            # Проверяем здоровье
            if self.check_server_health(name, config):
                print(f"✅ {name} успешно запущен")
                return True
            else:
                print(f"❌ {name} не отвечает на проверку здоровья")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка запуска {name}: {e}")
            return False
    
    def check_server_health(self, name, config):
        """Проверяем здоровье сервера"""
        try:
            url = f"http://localhost:{config['port']}{config['health_endpoint']}"
            
            if config['health_endpoint'] == '/api/chat':
                # Для Smart Dispatcher используем POST
                response = requests.post(url, json={'message': 'test'}, timeout=5)
            else:
                # Для остальных используем GET
                response = requests.get(url, timeout=5)
            
            return response.status_code in [200, 201]
        except:
            return False
    
    def start_critical_servers(self):
        """Запускаем критически важные серверы"""
        print("🔥 Запускаем критически важные серверы...")
        critical_servers = {k: v for k, v in self.servers.items() if v['critical']}
        
        for name, config in critical_servers.items():
            if not self.start_server(name, config):
                print(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: Не удалось запустить {name}")
                return False
        
        return True
    
    def start_additional_servers(self):
        """Запускаем дополнительные серверы"""
        print("⚡ Запускаем дополнительные серверы...")
        additional_servers = {k: v for k, v in self.servers.items() if not v['critical']}
        
        for name, config in additional_servers.items():
            self.start_server(name, config)
    
    def monitor_system(self):
        """Мониторинг системы"""
        print("📊 Запускаем мониторинг системы...")
        
        while True:
            try:
                online_count = 0
                total_count = len(self.servers)
                
                for name, config in self.servers.items():
                    if self.check_server_health(name, config):
                        online_count += 1
                
                print(f"📈 Статус: {online_count}/{total_count} серверов онлайн")
                
                if online_count < total_count * 0.8:  # Менее 80% серверов онлайн
                    print("⚠️ ВНИМАНИЕ: Много серверов оффлайн!")
                
                time.sleep(30)  # Проверяем каждые 30 секунд
                
            except KeyboardInterrupt:
                print("🛑 Мониторинг остановлен")
                break
            except Exception as e:
                print(f"❌ Ошибка мониторинга: {e}")
                time.sleep(10)
    
    def run(self):
        """Основной метод запуска"""
        print("🚀 STABLE SYSTEM LAUNCHER для Rubin AI v2")
        print("=" * 50)
        
        # Останавливаем существующие процессы
        self.kill_existing_processes()
        
        # Запускаем критически важные серверы
        if not self.start_critical_servers():
            print("🚨 КРИТИЧЕСКАЯ ОШИБКА: Не удалось запустить критически важные серверы")
            return False
        
        # Ждем стабилизации
        print("⏳ Ждем стабилизации системы...")
        time.sleep(5)
        
        # Запускаем дополнительные серверы
        self.start_additional_servers()
        
        # Финальная проверка
        print("🔍 Финальная проверка системы...")
        time.sleep(3)
        
        online_count = 0
        for name, config in self.servers.items():
            if self.check_server_health(name, config):
                online_count += 1
        
        print(f"✅ Система запущена: {online_count}/{len(self.servers)} серверов онлайн")
        
        if online_count >= len(self.servers) * 0.8:
            print("🎉 СИСТЕМА ГОТОВА К РАБОТЕ!")
            return True
        else:
            print("⚠️ СИСТЕМА ТРЕБУЕТ ВНИМАНИЯ!")
            return False

def main():
    launcher = StableSystemLauncher()
    
    try:
        success = launcher.run()
        if success:
            print("🚀 Запуск мониторинга...")
            launcher.monitor_system()
        else:
            print("❌ Запуск системы не удался")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Остановка системы...")
        launcher.kill_existing_processes()
        sys.exit(0)

if __name__ == "__main__":
    main()



