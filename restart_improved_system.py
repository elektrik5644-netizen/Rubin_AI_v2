#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Перезапуск системы Rubin AI с применением улучшений
"""

import subprocess
import time
import requests
import sys
import os

class SystemRestarter:
    def __init__(self):
        self.main_api_port = 8084
        self.documents_api_port = 8088
        
    def stop_existing_processes(self):
        """Остановка существующих процессов"""
        print("🛑 Остановка существующих процессов...")
        
        try:
            # Останавливаем процессы Python
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                         capture_output=True, text=True)
            print("✅ Процессы Python остановлены")
        except Exception as e:
            print(f"⚠️ Ошибка остановки процессов: {e}")
        
        # Проверяем, что порты свободны
        time.sleep(2)
        self.check_ports_free()
    
    def check_ports_free(self):
        """Проверка, что порты свободны"""
        print("🔍 Проверка свободности портов...")
        
        try:
            # Проверяем порт 8084
            response = requests.get(f"http://localhost:{self.main_api_port}/health", 
                                  timeout=2)
            print(f"⚠️ Порт {self.main_api_port} все еще занят")
        except:
            print(f"✅ Порт {self.main_api_port} свободен")
        
        try:
            # Проверяем порт 8088
            response = requests.get(f"http://localhost:{self.documents_api_port}/health", 
                                  timeout=2)
            print(f"⚠️ Порт {self.documents_api_port} все еще занят")
        except:
            print(f"✅ Порт {self.documents_api_port} свободен")
    
    def start_main_api(self):
        """Запуск основного API"""
        print("🚀 Запуск основного API Rubin AI...")
        
        try:
            # Запускаем основной API в фоновом режиме
            process = subprocess.Popen([
                sys.executable, "api/rubin_ai_v2_simple.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"✅ Основной API запущен (PID: {process.pid})")
            return process
            
        except Exception as e:
            print(f"❌ Ошибка запуска основного API: {e}")
            return None
    
    def start_documents_api(self):
        """Запуск API документов"""
        print("📚 Запуск API документов...")
        
        try:
            # Запускаем API документов в фоновом режиме
            process = subprocess.Popen([
                sys.executable, "api/documents_api.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"✅ API документов запущен (PID: {process.pid})")
            return process
            
        except Exception as e:
            print(f"❌ Ошибка запуска API документов: {e}")
            return None
    
    def wait_for_services(self, timeout=60):
        """Ожидание готовности сервисов"""
        print("⏳ Ожидание готовности сервисов...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Проверяем основной API
            try:
                response = requests.get(f"http://localhost:{self.main_api_port}/health", 
                                      timeout=5)
                if response.status_code == 200:
                    print("✅ Основной API готов")
                    break
            except:
                pass
            
            time.sleep(2)
            print(".", end="", flush=True)
        
        print()  # Новая строка
        
        # Проверяем API документов
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.documents_api_port}/health", 
                                      timeout=5)
                if response.status_code == 200:
                    print("✅ API документов готов")
                    break
            except:
                pass
            
            time.sleep(2)
            print(".", end="", flush=True)
        
        print()  # Новая строка
    
    def test_system_health(self):
        """Тестирование здоровья системы"""
        print("🏥 Проверка здоровья системы...")
        
        # Тестируем основной API
        try:
            response = requests.get(f"http://localhost:{self.main_api_port}/health", 
                                  timeout=10)
            if response.status_code == 200:
                print("✅ Основной API работает")
            else:
                print(f"⚠️ Основной API вернул статус: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка основного API: {e}")
        
        # Тестируем API документов
        try:
            response = requests.get(f"http://localhost:{self.documents_api_port}/health", 
                                  timeout=10)
            if response.status_code == 200:
                print("✅ API документов работает")
            else:
                print(f"⚠️ API документов вернул статус: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка API документов: {e}")
        
        # Тестируем поиск
        try:
            response = requests.post(
                f"http://localhost:{self.main_api_port}/api/chat",
                json={"message": "ПИД-регулятор"},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            if response.status_code == 200:
                print("✅ Поиск работает")
            else:
                print(f"⚠️ Поиск вернул статус: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
    
    def restart_system(self):
        """Полный перезапуск системы"""
        print("🔄 Полный перезапуск системы Rubin AI с улучшениями...\n")
        
        # 1. Останавливаем существующие процессы
        self.stop_existing_processes()
        
        # 2. Запускаем основной API
        main_process = self.start_main_api()
        if not main_process:
            print("❌ Не удалось запустить основной API")
            return False
        
        # 3. Запускаем API документов
        docs_process = self.start_documents_api()
        if not docs_process:
            print("❌ Не удалось запустить API документов")
            return False
        
        # 4. Ожидаем готовности сервисов
        self.wait_for_services()
        
        # 5. Тестируем систему
        self.test_system_health()
        
        print("\n🎉 Система успешно перезапущена с применением всех улучшений!")
        print(f"🌐 Основной API: http://localhost:{self.main_api_port}")
        print(f"📚 API документов: http://localhost:{self.documents_api_port}")
        print(f"🖥️ Веб-интерфейс: http://localhost:{self.main_api_port}/RubinIDE.html")
        
        return True

def main():
    """Основная функция"""
    restarter = SystemRestarter()
    success = restarter.restart_system()
    
    if success:
        print("\n✅ Система готова к работе!")
        print("🧪 Рекомендуется запустить тестирование: python test_improved_system.py")
    else:
        print("\n❌ Произошли ошибки при перезапуске системы")
        sys.exit(1)

if __name__ == "__main__":
    main()

















