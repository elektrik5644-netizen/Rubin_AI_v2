#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Скрипт обучения Rubin AI от Gemini
Автоматизированное обучение всех модулей системы Rubin AI
"""

import requests
import json
import time
from datetime import datetime

class RubinAITrainer:
    def __init__(self):
        self.gemini_bridge_url = "http://localhost:8082"
        self.rubin_url = "http://localhost:8080"
        self.training_sessions = []
        
    def check_system_health(self):
        """Проверяет состояние системы"""
        try:
            # Проверяем Gemini Bridge
            bridge_status = requests.get(f"{self.gemini_bridge_url}/api/gemini/status", timeout=5)
            print(f"🌉 Gemini Bridge: {'✅' if bridge_status.status_code == 200 else '❌'}")
            
            # Проверяем Smart Dispatcher
            dispatcher_status = requests.get(f"{self.rubin_url}/api/health", timeout=5)
            print(f"🎯 Smart Dispatcher: {'✅' if dispatcher_status.status_code == 200 else '❌'}")
            
            return bridge_status.status_code == 200 and dispatcher_status.status_code == 200
        except Exception as e:
            print(f"❌ Ошибка проверки системы: {e}")
            return False
    
    def teach_mathematics(self):
        """Обучение математическому модулю"""
        lessons = [
            {
                "topic": "алгебра",
                "instruction": "Объясни основы алгебры: переменные, уравнения, функции",
                "category": "mathematics"
            },
            {
                "topic": "геометрия", 
                "instruction": "Расскажи о геометрических фигурах и их свойствах",
                "category": "mathematics"
            },
            {
                "topic": "тригонометрия",
                "instruction": "Объясни тригонометрические функции и их применение",
                "category": "mathematics"
            }
        ]
        
        for lesson in lessons:
            self.teach_lesson(lesson)
            time.sleep(2)
    
    def teach_electrical(self):
        """Обучение электротехническому модулю"""
        lessons = [
            {
                "topic": "закон ома",
                "instruction": "Объясни закон Ома и его применение в электрических цепях",
                "category": "electrical"
            },
            {
                "topic": "транзисторы",
                "instruction": "Расскажи о принципе работы транзисторов и их типах",
                "category": "electrical"
            },
            {
                "topic": "конденсаторы",
                "instruction": "Объясни работу конденсаторов в электрических цепях",
                "category": "electrical"
            }
        ]
        
        for lesson in lessons:
            self.teach_lesson(lesson)
            time.sleep(2)
    
    def teach_programming(self):
        """Обучение модулю программирования"""
        lessons = [
            {
                "topic": "python",
                "instruction": "Объясни основы программирования на Python",
                "category": "programming"
            },
            {
                "topic": "алгоритмы",
                "instruction": "Расскажи о базовых алгоритмах и структурах данных",
                "category": "programming"
            },
            {
                "topic": "ооп",
                "instruction": "Объясни принципы объектно-ориентированного программирования",
                "category": "programming"
            }
        ]
        
        for lesson in lessons:
            self.teach_lesson(lesson)
            time.sleep(2)
    
    def teach_controllers(self):
        """Обучение модулю контроллеров"""
        lessons = [
            {
                "topic": "plc",
                "instruction": "Объясни принципы работы ПЛК и программирование лестничной логики",
                "category": "controllers"
            },
            {
                "topic": "pid",
                "instruction": "Расскажи о PID регуляторах и их настройке",
                "category": "controllers"
            },
            {
                "topic": "автоматизация",
                "instruction": "Объясни основы промышленной автоматизации",
                "category": "controllers"
            }
        ]
        
        for lesson in lessons:
            self.teach_lesson(lesson)
            time.sleep(2)
    
    def teach_gai(self):
        """Обучение GAI модулю"""
        lessons = [
            {
                "topic": "генерация текста",
                "instruction": "Объясни принципы генерации текста и языковых моделей",
                "category": "gai"
            },
            {
                "topic": "творчество",
                "instruction": "Расскажи о применении ИИ в творческих задачах",
                "category": "gai"
            }
        ]
        
        for lesson in lessons:
            self.teach_lesson(lesson)
            time.sleep(2)
    
    def teach_lesson(self, lesson):
        """Отправляет урок через Gemini Bridge"""
        try:
            payload = {
                "topic": lesson["topic"],
                "instruction": lesson["instruction"],
                "category": lesson["category"]
            }
            
            response = requests.post(
                f"{self.gemini_bridge_url}/api/gemini/teach",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Урок '{lesson['topic']}' успешно передан в модуль {lesson['category']}")
                print(f"   Ответ Rubin: {result.get('rubin_ai_response', '')[:100]}...")
                
                self.training_sessions.append({
                    "timestamp": datetime.now().isoformat(),
                    "lesson": lesson,
                    "success": True,
                    "response": result
                })
            else:
                print(f"❌ Ошибка обучения '{lesson['topic']}': {response.status_code}")
                self.training_sessions.append({
                    "timestamp": datetime.now().isoformat(),
                    "lesson": lesson,
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"❌ Исключение при обучении '{lesson['topic']}': {e}")
            self.training_sessions.append({
                "timestamp": datetime.now().isoformat(),
                "lesson": lesson,
                "success": False,
                "error": str(e)
            })
    
    def analyze_system(self):
        """Анализирует систему Rubin AI"""
        try:
            payload = {
                "type": "comprehensive",
                "query": "Проанализируй состояние всех модулей системы Rubin AI"
            }
            
            response = requests.post(
                f"{self.gemini_bridge_url}/api/gemini/analyze",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"📊 Анализ системы завершен")
                print(f"   Результат: {result.get('rubin_ai_response', '')[:200]}...")
                return result
            else:
                print(f"❌ Ошибка анализа: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Исключение при анализе: {e}")
            return None
    
    def run_comprehensive_training(self):
        """Запускает комплексное обучение"""
        print("🚀 Начинаем комплексное обучение Rubin AI от Gemini")
        print("=" * 60)
        
        # Проверяем систему
        if not self.check_system_health():
            print("❌ Система не готова к обучению")
            return
        
        print("\n📚 Начинаем обучение модулей...")
        
        # Обучаем все модули
        print("\n🧮 Обучение математического модуля...")
        self.teach_mathematics()
        
        print("\n⚡ Обучение электротехнического модуля...")
        self.teach_electrical()
        
        print("\n💻 Обучение модуля программирования...")
        self.teach_programming()
        
        print("\n🎛️ Обучение модуля контроллеров...")
        self.teach_controllers()
        
        print("\n🤖 Обучение GAI модуля...")
        self.teach_gai()
        
        # Анализируем результаты
        print("\n📊 Анализ результатов обучения...")
        self.analyze_system()
        
        # Выводим статистику
        self.print_training_summary()
    
    def print_training_summary(self):
        """Выводит сводку по обучению"""
        print("\n" + "=" * 60)
        print("📈 СВОДКА ОБУЧЕНИЯ")
        print("=" * 60)
        
        total_lessons = len(self.training_sessions)
        successful_lessons = sum(1 for session in self.training_sessions if session["success"])
        failed_lessons = total_lessons - successful_lessons
        
        print(f"📚 Всего уроков: {total_lessons}")
        print(f"✅ Успешных: {successful_lessons}")
        print(f"❌ Неудачных: {failed_lessons}")
        print(f"📊 Успешность: {(successful_lessons/total_lessons*100):.1f}%")
        
        if failed_lessons > 0:
            print("\n❌ Неудачные уроки:")
            for session in self.training_sessions:
                if not session["success"]:
                    print(f"   - {session['lesson']['topic']}: {session.get('error', 'Неизвестная ошибка')}")
        
        print(f"\n🕒 Обучение завершено: {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Основная функция"""
    trainer = RubinAITrainer()
    trainer.run_comprehensive_training()

if __name__ == "__main__":
    main()



