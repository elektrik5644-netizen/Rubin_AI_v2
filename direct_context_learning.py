#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 СИСТЕМА ПРЯМОГО ОБУЧЕНИЯ КОНТЕКСТУ RUBIN AI
==============================================
Радикальный подход к обучению понимания контекста
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectContextLearning:
    """Система прямого обучения контексту"""
    
    def __init__(self):
        self.context_memory = {
            "today_activities": [
                "Создали систему постоянного сканирования Rubin AI",
                "Исправили HTTP 500 ошибки через fallback механизм в Smart Dispatcher",
                "Модернизировали VMB630 с паттернами проектирования (Singleton, Observer, Factory, Strategy, Command)",
                "Проанализировали PLC файл и нашли ошибки (опечатка AXIS_DISCONNECTEP_TP_P, неправильный таймер)",
                "Создали систему автоматического исправления PLC ошибок",
                "Обучили Rubin AI пониманию процессов диагностики и модернизации"
            ],
            "current_session": "learning_context_understanding",
            "rubin_problems": [
                "Дает шаблонные ответы вместо понимания контекста",
                "Предлагает общие категории вместо конкретных ответов",
                "Не помнит предыдущие взаимодействия",
                "Не понимает, что 'обучение' относится к нашему взаимодействию"
            ],
            "expected_behavior": [
                "Понимать контекст вопросов",
                "Отвечать конкретно на основе нашего взаимодействия",
                "Помнить что мы делали сегодня",
                "Избегать шаблонных ответов"
            ]
        }
        self.smart_dispatcher_url = "http://localhost:8080/api/chat"
        
    def direct_context_teaching(self):
        """Прямое обучение пониманию контекста"""
        print("🎯 ПРЯМОЕ ОБУЧЕНИЕ ПОНИМАНИЮ КОНТЕКСТА RUBIN AI")
        print("=" * 60)
        
        # Проверяем доступность Smart Dispatcher
        try:
            response = requests.get('http://localhost:8080/api/health')
            if response.status_code == 200:
                print("✅ Smart Dispatcher подключен")
            else:
                print(f"❌ Smart Dispatcher недоступен (статус: {response.status_code})")
                return
        except Exception as e:
            print(f"❌ Ошибка подключения к Smart Dispatcher: {e}")
            return
        
        # Прямые инструкции для Rubin AI
        direct_instructions = [
            {
                "title": "Прямая инструкция по контексту",
                "message": "Программирование: Rubin, СЛУШАЙ ВНИМАТЕЛЬНО! Когда я спрашиваю 'как проходит твое обучение', я НЕ хочу слышать про категории или модули. Я хочу конкретный ответ о том, что мы делали СЕГОДНЯ: создали систему сканирования, исправили HTTP 500 ошибки, модернизировали VMB630, проанализировали PLC файлы. Отвечай КОНКРЕТНО о нашем взаимодействии!",
                "expected_response": "конкретно о сегодняшней работе"
            },
            {
                "title": "Запрет шаблонных ответов",
                "message": "Программирование: Rubin, ЗАПОМНИ! НЕ ДАВАЙ шаблонные ответы типа 'Для более точного ответа уточните тему' или списки категорий. Когда я спрашиваю о твоем обучении, отвечай ПРЯМО о том, что мы изучали: диагностику ошибок, fallback механизмы, паттерны проектирования, анализ PLC кода. БЕЗ ШАБЛОНОВ!",
                "expected_response": "без шаблонов, конкретно"
            },
            {
                "title": "Память о взаимодействии",
                "message": "Программирование: Rubin, ПОМНИ! Сегодня мы работали над: 1) Системой постоянного сканирования твоего обучения, 2) Исправлением HTTP 500 ошибок через fallback, 3) Модернизацией VMB630 с паттернами, 4) Анализом PLC файлов. Когда я спрашиваю о твоем обучении, отвечай именно об ЭТОМ!",
                "expected_response": "память о конкретных действиях"
            },
            {
                "title": "Понимание прогресса",
                "message": "Программирование: Rubin, ПОНИМАЙ! Твое обучение сегодня включало: изучение диагностики HTTP 500 ошибок, понимание fallback механизмов, изучение паттернов проектирования (Singleton, Observer, Factory), анализ PLC кода и исправление ошибок. Когда я спрашиваю о прогрессе, отвечай именно об ЭТОМ прогрессе!",
                "expected_response": "конкретный прогресс обучения"
            },
            {
                "title": "Контекстное мышление",
                "message": "Программирование: Rubin, ДУМАЙ КОНТЕКСТНО! Когда я спрашиваю 'как проходит твое обучение', я имею в виду НАШЕ взаимодействие сегодня, а НЕ общие принципы обучения. Отвечай о том, что ты изучил в нашем конкретном взаимодействии: диагностику, исправление ошибок, модернизацию, анализ кода. КОНТЕКСТ ВАЖЕН!",
                "expected_response": "контекстное понимание"
            }
        ]
        
        for i, instruction in enumerate(direct_instructions, 1):
            print(f"\n🎯 ИНСТРУКЦИЯ {i}: {instruction['title']}")
            print("-" * 50)
            print(f"📝 Сообщение: {instruction['message']}")
            
            try:
                response = requests.post('http://localhost:8080/api/chat', 
                                      json={'message': instruction['message']})
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        explanation = data['response'].get('explanation', data['response'].get('response', 'Нет объяснения'))
                        print(f"\n🤖 ОТВЕТ RUBIN AI:")
                        print(f"📋 {explanation[:400]}..." if len(explanation) > 400 else f"📋 {explanation}")
                        
                        # Анализируем ответ
                        context_score = self._analyze_context_understanding(explanation, instruction['expected_response'])
                        print(f"📊 Оценка понимания контекста: {context_score}/10")
                        
                    else:
                        print("❌ Ошибка в ответе Rubin AI")
                else:
                    print(f"❌ HTTP ошибка: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Ошибка запроса: {e}")
            
            time.sleep(2)  # Пауза между инструкциями
    
    def _analyze_context_understanding(self, response: str, expected: str) -> int:
        """Анализируем понимание контекста"""
        response_lower = response.lower()
        
        # Проверяем на шаблонные ответы (штраф)
        template_penalty = 0
        if "уточните тему" in response_lower:
            template_penalty += 5
        if "категория" in response_lower or "модуль" in response_lower:
            template_penalty += 3
        if "электротехника" in response_lower or "математика" in response_lower:
            template_penalty += 4
        
        # Проверяем на конкретность (бонус)
        specificity_bonus = 0
        if "сегодня" in response_lower:
            specificity_bonus += 2
        if "конкретно" in response_lower or "конкретный" in response_lower:
            specificity_bonus += 2
        if "наш" in response_lower or "наше" in response_lower:
            specificity_bonus += 1
        if "взаимодействие" in response_lower:
            specificity_bonus += 2
        
        # Проверяем на упоминание конкретных тем
        topic_bonus = 0
        if "http 500" in response_lower or "500" in response_lower:
            topic_bonus += 1
        if "fallback" in response_lower:
            topic_bonus += 1
        if "vmb630" in response_lower:
            topic_bonus += 1
        if "plc" in response_lower:
            topic_bonus += 1
        if "паттерн" in response_lower:
            topic_bonus += 1
        
        # Итоговая оценка
        base_score = 5  # Базовый балл
        total_score = base_score - template_penalty + specificity_bonus + topic_bonus
        
        return min(10, max(0, int(total_score)))
    
    def test_context_understanding(self):
        """Тестируем понимание контекста"""
        print("\n🧪 ТЕСТ ПОНИМАНИЯ КОНТЕКСТА")
        print("=" * 40)
        
        test_questions = [
            "Как проходит твое обучение?",
            "Что ты изучил сегодня?", 
            "Как ты понимаешь наш процесс работы?",
            "Расскажи о нашем взаимодействии",
            "Что мы делали сегодня?"
        ]
        
        total_score = 0
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 ТЕСТ {i}: {question}")
            
            try:
                response = requests.post('http://localhost:8080/api/chat', 
                                      json={'message': question})
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        explanation = data['response'].get('explanation', data['response'].get('response', 'Нет объяснения'))
                        print(f"📋 Ответ: {explanation[:300]}..." if len(explanation) > 300 else f"📋 Ответ: {explanation}")
                        
                        # Оцениваем ответ
                        score = self._evaluate_context_response(explanation)
                        print(f"📊 Оценка: {score}/10")
                        total_score += score
                        
                    else:
                        print("❌ Ошибка в ответе Rubin AI")
                else:
                    print(f"❌ HTTP ошибка: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Ошибка запроса: {e}")
            
            time.sleep(1)
        
        average_score = total_score / len(test_questions)
        print(f"\n📊 ОБЩАЯ ОЦЕНКА ПОНИМАНИЯ КОНТЕКСТА: {average_score:.1f}/10")
        
        if average_score >= 7:
            print("✅ Rubin AI понимает контекст")
        elif average_score >= 5:
            print("⚠️ Rubin AI частично понимает контекст")
        else:
            print("❌ Rubin AI НЕ понимает контекст")
    
    def _evaluate_context_response(self, response: str) -> int:
        """Оцениваем ответ на понимание контекста"""
        response_lower = response.lower()
        
        # Штрафы за шаблонные ответы
        penalty = 0
        if "уточните тему" in response_lower:
            penalty += 5
        if "категория" in response_lower:
            penalty += 3
        if "модуль" in response_lower:
            penalty += 3
        if "электротехника" in response_lower or "математика" in response_lower:
            penalty += 4
        
        # Бонусы за конкретность
        bonus = 0
        if "сегодня" in response_lower:
            bonus += 2
        if "конкретно" in response_lower:
            bonus += 2
        if "наш" in response_lower or "наше" in response_lower:
            bonus += 1
        if "взаимодействие" in response_lower:
            bonus += 2
        
        # Бонусы за упоминание конкретных тем
        if "http 500" in response_lower or "500" in response_lower:
            bonus += 1
        if "fallback" in response_lower:
            bonus += 1
        if "vmb630" in response_lower:
            bonus += 1
        if "plc" in response_lower:
            bonus += 1
        if "паттерн" in response_lower:
            bonus += 1
        if "сканирование" in response_lower:
            bonus += 1
        
        # Итоговая оценка
        score = 5 - penalty + bonus
        return min(10, max(0, int(score)))
    
    def generate_context_report(self):
        """Генерируем отчет о понимании контекста"""
        print("\n📊 ОТЧЕТ О ПОНИМАНИИ КОНТЕКСТА:")
        print("=" * 40)
        
        print("🎯 ПРОБЛЕМЫ RUBIN AI:")
        for problem in self.context_memory["rubin_problems"]:
            print(f"  ❌ {problem}")
        
        print("\n✅ ОЖИДАЕМОЕ ПОВЕДЕНИЕ:")
        for behavior in self.context_memory["expected_behavior"]:
            print(f"  ✅ {behavior}")
        
        print("\n📋 ЧТО МЫ ДЕЛАЛИ СЕГОДНЯ:")
        for activity in self.context_memory["today_activities"]:
            print(f"  📌 {activity}")
        
        # Сохраняем отчет
        report = {
            "timestamp": datetime.now().isoformat(),
            "context_memory": self.context_memory,
            "problems_identified": self.context_memory["rubin_problems"],
            "expected_behavior": self.context_memory["expected_behavior"],
            "today_activities": self.context_memory["today_activities"],
            "recommendations": [
                "Продолжить прямое обучение контексту",
                "Использовать более конкретные примеры",
                "Повторять инструкции до понимания",
                "Тестировать понимание после каждого урока"
            ]
        }
        
        try:
            with open('DIRECT_CONTEXT_LEARNING_REPORT.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 Отчет сохранен: DIRECT_CONTEXT_LEARNING_REPORT.json")
        except Exception as e:
            print(f"❌ Ошибка сохранения отчета: {e}")

def main():
    """Основная функция прямого обучения контексту"""
    print("🎯 СИСТЕМА ПРЯМОГО ОБУЧЕНИЯ КОНТЕКСТУ RUBIN AI")
    print("=" * 70)
    print("Цель: Научить Rubin AI понимать контекст и избегать шаблонных ответов")
    print("=" * 70)
    
    # Создаем систему прямого обучения
    learning_system = DirectContextLearning()
    
    # Прямое обучение контексту
    learning_system.direct_context_teaching()
    
    # Тестируем понимание контекста
    learning_system.test_context_understanding()
    
    # Генерируем отчет
    learning_system.generate_context_report()
    
    print("\n🎉 ПРЯМОЕ ОБУЧЕНИЕ КОНТЕКСТУ ЗАВЕРШЕНО!")
    print("=" * 40)
    print("✅ Rubin AI получил прямые инструкции по пониманию контекста")
    print("📊 Система тестирования понимания контекста активна")
    print("🧠 Rubin AI должен лучше понимать контекст вопросов")

if __name__ == "__main__":
    main()










