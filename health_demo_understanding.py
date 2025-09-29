#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏥 HEALTH & DEMO UNDERSTANDING MODULE
====================================
Модуль для понимания вопросов о здоровье и демо
"""

import requests
import json
import time
from datetime import datetime

class HealthDemoUnderstanding:
    """Класс для понимания здоровья и демо"""
    
    def __init__(self):
        self.servers = {
            'smart_dispatcher': {'port': 8080, 'endpoint': '/api/health'},
            'learning_server': {'port': 8091, 'endpoint': '/api/learning/health'},
            'controllers_server': {'port': 9000, 'endpoint': '/api/health'},
            'math_server': {'port': 8086, 'endpoint': '/api/health'}
        }
        
        self.health_keywords = [
            'здоров', 'работает', 'статус', 'сервер', 'модуль', 'система',
            'проблем', 'ошибк', 'неисправн', 'провер', 'диагност'
        ]
        
        self.demo_keywords = [
            'демо', 'демонстрация', 'показать', 'запустить', 'пример',
            'тест', 'провер', 'работает ли'
        ]
    
    def check_server_health(self, server_name):
        """Проверка здоровья сервера"""
        if server_name not in self.servers:
            return False, f"Неизвестный сервер: {server_name}"
        
        config = self.servers[server_name]
        url = f"http://localhost:{config['port']}{config['endpoint']}"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True, "Здоров"
            else:
                return False, f"Ошибка {response.status_code}"
        except Exception as e:
            return False, f"Недоступен: {str(e)[:30]}"
    
    def check_all_servers(self):
        """Проверка всех серверов"""
        results = {}
        healthy_count = 0
        
        for server_name in self.servers:
            is_healthy, status = self.check_server_health(server_name)
            results[server_name] = {'healthy': is_healthy, 'status': status}
            if is_healthy:
                healthy_count += 1
        
        return results, healthy_count, len(self.servers)
    
    def is_health_question(self, question):
        """Определяет, является ли вопрос о здоровье"""
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.health_keywords)
    
    def is_demo_question(self, question):
        """Определяет, является ли вопрос о демо"""
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.demo_keywords)
    
    def generate_health_response(self, question):
        """Генерирует ответ о здоровье"""
        results, healthy_count, total_count = self.check_all_servers()
        
        response = f"""🏥 **СТАТУС ЗДОРОВЬЯ RUBIN AI**

**📊 Общий статус:** {healthy_count}/{total_count} серверов работают

**🔍 Детальная проверка:**
"""
        
        for server_name, result in results.items():
            status_icon = "✅" if result['healthy'] else "❌"
            response += f"• {status_icon} {server_name}: {result['status']}\n"
        
        if healthy_count == total_count:
            response += f"""
**🎉 Все системы работают нормально!**
Rubin AI полностью здоров и готов к работе.
"""
        else:
            response += f"""
**⚠️ Обнаружены проблемы:**
{total_count - healthy_count} серверов не работают.
Рекомендуется проверить конфигурацию.
"""
        
        return response
    
    def generate_demo_response(self, question):
        """Генерирует ответ о демо"""
        return f"""🎬 **ДЕМОНСТРАЦИЯ RUBIN AI**

**🎯 Доступные демо:**
• `demo_solution_final.py` - Демонстрация решения проблемы шаблонных ответов
• `demo_rubin_self_learning.py` - Демонстрация самообучения Rubin AI
• `test_rubin_health_understanding.py` - Демонстрация понимания здоровья
• `vmb630_advanced_architecture.py` - Демонстрация паттернов проектирования

**🚀 Как запустить демо:**
```bash
python demo_solution_final.py
python demo_rubin_self_learning.py
python test_rubin_health_understanding.py
```

**📋 Что показывают демо:**
• Понимание контекста и избегание шаблонов
• Самообучение и запоминание опыта
• Диагностику здоровья системы
• Применение паттернов проектирования

**💡 Rubin AI может продемонстрировать:**
• Анализ PLC файлов и поиск ошибок
• Решение математических задач
• Модернизацию архитектуры систем
• Диагностику и исправление ошибок
"""

def test_health_demo_understanding():
    """Тестируем понимание здоровья и демо"""
    print("🏥 ТЕСТ ПОНИМАНИЯ ЗДОРОВЬЯ И ДЕМО")
    print("=" * 50)
    
    understanding = HealthDemoUnderstanding()
    
    # Тестовые вопросы
    test_questions = [
        "Как дела? Все работает?",
        "Проверь свое здоровье",
        "Какие серверы у тебя работают?",
        "Есть ли проблемы с системой?",
        "Что с демо? Работает ли демонстрация?",
        "Можешь запустить демо?",
        "Покажи мне демонстрацию работы"
    ]
    
    for question in test_questions:
        print(f"\n📝 Вопрос: {question}")
        print("-" * 40)
        
        if understanding.is_health_question(question):
            print("🏥 Определен как вопрос о здоровье")
            response = understanding.generate_health_response(question)
            print("📋 Ответ:")
            print(response[:300] + "..." if len(response) > 300 else response)
        
        elif understanding.is_demo_question(question):
            print("🎬 Определен как вопрос о демо")
            response = understanding.generate_demo_response(question)
            print("📋 Ответ:")
            print(response[:300] + "..." if len(response) > 300 else response)
        
        else:
            print("❓ Не определен тип вопроса")
            print("💡 Возможно, нужна дополнительная обработка")

if __name__ == "__main__":
    test_health_demo_understanding()










