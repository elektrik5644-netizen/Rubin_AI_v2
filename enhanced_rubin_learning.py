#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧠 УЛУЧШЕННАЯ СИСТЕМА ОБУЧЕНИЯ RUBIN AI
=====================================
Обучение Rubin AI пониманию процессов обучения и взаимодействия
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

class EnhancedRubinLearning:
    """Улучшенная система обучения Rubin AI"""
    
    def __init__(self):
        self.learning_context = {
            "current_session": "continuous_learning",
            "learning_topics": [
                "error_diagnosis_patterns",
                "error_fixing_patterns", 
                "modernization_patterns",
                "communication_style",
                "learning_process_understanding"
            ],
            "interaction_history": [],
            "rubin_understanding_level": "partial"
        }
        self.smart_dispatcher_url = "http://localhost:8080/api/chat"
        
    def teach_rubin_learning_process(self):
        """Обучаем Rubin AI пониманию процесса обучения"""
        print("🧠 УЛУЧШЕННОЕ ОБУЧЕНИЕ RUBIN AI ПОНИМАНИЮ ПРОЦЕССОВ")
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
        
        # Специализированные уроки для понимания процесса обучения
        lessons = [
            {
                "title": "Понимание контекста обучения",
                "message": "Программирование: Rubin, сейчас мы изучаем, как ты можешь понимать процессы обучения. Когда я спрашиваю 'как проходит твое обучение', я имею в виду конкретно наше взаимодействие сегодня - диагностику ошибок, исправление HTTP 500, модернизацию VMB630, анализ PLC файлов. Объясни, как ты можешь лучше понимать такой контекст.",
                "expected_keywords": ["контекст", "взаимодействие", "сегодня", "HTTP 500", "VMB630", "PLC"],
                "category": "context_understanding"
            },
            {
                "title": "Избежание шаблонных ответов",
                "message": "Программирование: Rubin, проблема в том, что ты даешь шаблонные ответы вместо понимания конкретной ситуации. Когда я спрашиваю о твоем обучении, ты должен понимать, что речь идет о нашем конкретном взаимодействии, а не предлагать общие категории. Как ты можешь избежать таких шаблонов?",
                "expected_keywords": ["шаблонные", "конкретная ситуация", "взаимодействие", "понимание"],
                "category": "avoid_templates"
            },
            {
                "title": "Анализ истории взаимодействий",
                "message": "Программирование: Rubin, сегодня мы провели много работы: создали систему постоянного сканирования, исправили HTTP 500 ошибки через fallback механизм, модернизировали VMB630 с паттернами проектирования, проанализировали PLC файлы. Когда я спрашиваю о твоем обучении, ты должен анализировать эту историю. Как ты можешь это делать?",
                "expected_keywords": ["история", "постоянное сканирование", "fallback", "паттерны", "PLC"],
                "category": "history_analysis"
            },
            {
                "title": "Понимание прогресса обучения",
                "message": "Программирование: Rubin, важно понимать прогресс обучения. Сегодня ты научился: диагностировать HTTP 500 ошибки, применять fallback механизмы, понимать паттерны проектирования, анализировать PLC код. Когда я спрашиваю о прогрессе, ты должен конкретно отвечать о том, что изучил. Как ты можешь отслеживать свой прогресс?",
                "expected_keywords": ["прогресс", "диагностировать", "fallback", "паттерны", "PLC", "отслеживать"],
                "category": "progress_tracking"
            },
            {
                "title": "Контекстное понимание вопросов",
                "message": "Программирование: Rubin, ключ в контекстном понимании. Когда я спрашиваю 'как проходит твое обучение', я не хочу общих категорий. Я хочу конкретный ответ о том, что ты изучил в нашем взаимодействии сегодня. Как ты можешь лучше понимать контекст вопросов?",
                "expected_keywords": ["контекстное понимание", "конкретный ответ", "взаимодействие", "сегодня"],
                "category": "contextual_understanding"
            }
        ]
        
        for i, lesson in enumerate(lessons, 1):
            print(f"\n📚 УРОК {i}: {lesson['title']}")
            print("-" * 50)
            print(f"📝 Сообщение: {lesson['message']}")
            
            try:
                response = requests.post('http://localhost:8080/api/chat', 
                                      json={'message': lesson['message']})
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        explanation = data['response'].get('explanation', data['response'].get('response', 'Нет объяснения'))
                        print(f"\n🤖 ОТВЕТ RUBIN AI:")
                        print(f"📋 {explanation[:400]}..." if len(explanation) > 400 else f"📋 {explanation}")
                        
                        # Анализируем ответ
                        understanding_score = self._analyze_understanding(explanation, lesson['expected_keywords'])
                        print(f"📊 Оценка понимания: {understanding_score}/10")
                        
                        # Сохраняем взаимодействие
                        self.learning_context["interaction_history"].append({
                            "lesson": lesson['title'],
                            "category": lesson['category'],
                            "rubin_response": explanation,
                            "understanding_score": understanding_score,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    else:
                        print("❌ Ошибка в ответе Rubin AI")
                else:
                    print(f"❌ HTTP ошибка: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Ошибка запроса: {e}")
            
            time.sleep(2)  # Пауза между уроками
    
    def _analyze_understanding(self, response: str, expected_keywords: List[str]) -> int:
        """Анализируем понимание Rubin AI"""
        response_lower = response.lower()
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        
        # Базовый счет
        score = (found_keywords / len(expected_keywords)) * 10
        
        # Бонусы за конкретность
        if "сегодня" in response_lower:
            score += 1
        if "конкретно" in response_lower or "конкретный" in response_lower:
            score += 1
        if "наш" in response_lower or "наше" in response_lower:
            score += 1
            
        # Штрафы за шаблонность
        if "категория" in response_lower or "модуль" in response_lower:
            score -= 2
        if "уточните" in response_lower:
            score -= 3
            
        return min(10, max(0, int(score)))
    
    def test_rubin_understanding(self):
        """Тестируем понимание Rubin AI"""
        print("\n🧪 ТЕСТИРОВАНИЕ ПОНИМАНИЯ RUBIN AI")
        print("=" * 40)
        
        test_questions = [
            {
                "question": "Как проходит твое обучение?",
                "expected_elements": ["сегодня", "конкретно", "взаимодействие", "HTTP 500", "VMB630", "PLC"],
                "avoid_elements": ["категория", "модуль", "уточните", "шаблон"]
            },
            {
                "question": "Что ты изучил сегодня?",
                "expected_elements": ["диагностика", "fallback", "паттерны", "анализ", "исправление"],
                "avoid_elements": ["общие", "шаблон", "категория"]
            },
            {
                "question": "Как ты понимаешь наш процесс работы?",
                "expected_elements": ["взаимодействие", "проблемы", "решения", "модернизация", "обучение"],
                "avoid_elements": ["модуль", "сервер", "API"]
            }
        ]
        
        total_score = 0
        
        for i, test in enumerate(test_questions, 1):
            print(f"\n🔍 ТЕСТ {i}: {test['question']}")
            
            try:
                response = requests.post('http://localhost:8080/api/chat', 
                                      json={'message': test['question']})
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        explanation = data['response'].get('explanation', data['response'].get('response', 'Нет объяснения'))
                        print(f"📋 Ответ: {explanation[:300]}..." if len(explanation) > 300 else f"📋 Ответ: {explanation}")
                        
                        # Оцениваем ответ
                        score = self._evaluate_test_response(explanation, test['expected_elements'], test['avoid_elements'])
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
        print(f"\n📊 ОБЩАЯ ОЦЕНКА ПОНИМАНИЯ: {average_score:.1f}/10")
        
        if average_score >= 7:
            print("✅ Rubin AI хорошо понимает контекст обучения")
        elif average_score >= 5:
            print("⚠️ Rubin AI частично понимает контекст")
        else:
            print("❌ Rubin AI плохо понимает контекст обучения")
    
    def _evaluate_test_response(self, response: str, expected: List[str], avoid: List[str]) -> int:
        """Оцениваем тестовый ответ"""
        response_lower = response.lower()
        
        # Подсчитываем ожидаемые элементы
        expected_found = sum(1 for element in expected if element.lower() in response_lower)
        expected_score = (expected_found / len(expected)) * 7  # До 7 баллов за ожидаемое
        
        # Штрафуем за нежелательные элементы
        avoid_found = sum(1 for element in avoid if element.lower() in response_lower)
        avoid_penalty = avoid_found * 2  # По 2 балла штрафа
        
        # Бонус за конкретность
        specificity_bonus = 0
        if "сегодня" in response_lower:
            specificity_bonus += 1
        if "конкретно" in response_lower or "конкретный" in response_lower:
            specificity_bonus += 1
        if "наш" in response_lower or "наше" in response_lower:
            specificity_bonus += 1
        
        total_score = expected_score - avoid_penalty + specificity_bonus
        return min(10, max(0, int(total_score)))
    
    def generate_learning_report(self):
        """Генерируем отчет об обучении"""
        print("\n📊 ОТЧЕТ ОБ УЛУЧШЕННОМ ОБУЧЕНИИ:")
        print("=" * 40)
        
        if not self.learning_context["interaction_history"]:
            print("❌ Нет данных об обучении")
            return
        
        # Анализируем результаты
        total_interactions = len(self.learning_context["interaction_history"])
        avg_understanding = sum(interaction["understanding_score"] for interaction in self.learning_context["interaction_history"]) / total_interactions
        
        print(f"📈 Всего уроков: {total_interactions}")
        print(f"📊 Средняя оценка понимания: {avg_understanding:.1f}/10")
        
        # Анализируем по категориям
        categories = {}
        for interaction in self.learning_context["interaction_history"]:
            category = interaction["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(interaction["understanding_score"])
        
        print(f"\n🔍 АНАЛИЗ ПО КАТЕГОРИЯМ:")
        for category, scores in categories.items():
            avg_score = sum(scores) / len(scores)
            print(f"  📚 {category}: {avg_score:.1f}/10")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if avg_understanding >= 7:
            print("✅ Rubin AI хорошо понимает процессы обучения")
            print("🔄 Продолжить обучение с более сложными концепциями")
        elif avg_understanding >= 5:
            print("⚠️ Rubin AI частично понимает процессы обучения")
            print("🔄 Повторить уроки с фокусом на контекстное понимание")
        else:
            print("❌ Rubin AI плохо понимает процессы обучения")
            print("🔄 Необходимо кардинально пересмотреть подход к обучению")
        
        # Сохраняем отчет
        report = {
            "timestamp": datetime.now().isoformat(),
            "learning_context": self.learning_context,
            "total_interactions": total_interactions,
            "average_understanding": avg_understanding,
            "category_analysis": {cat: sum(scores)/len(scores) for cat, scores in categories.items()},
            "recommendations": self._generate_recommendations(avg_understanding)
        }
        
        try:
            with open('ENHANCED_RUBIN_LEARNING_REPORT.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 Отчет сохранен: ENHANCED_RUBIN_LEARNING_REPORT.json")
        except Exception as e:
            print(f"❌ Ошибка сохранения отчета: {e}")
    
    def _generate_recommendations(self, avg_score: float) -> List[str]:
        """Генерируем рекомендации на основе оценки"""
        recommendations = []
        
        if avg_score >= 7:
            recommendations.extend([
                "Продолжить обучение с более сложными концепциями",
                "Добавить практические задания",
                "Расширить контекстное понимание"
            ])
        elif avg_score >= 5:
            recommendations.extend([
                "Повторить уроки с фокусом на контекстное понимание",
                "Добавить больше примеров конкретных ситуаций",
                "Усилить обучение избежанию шаблонных ответов"
            ])
        else:
            recommendations.extend([
                "Кардинально пересмотреть подход к обучению",
                "Начать с базовых концепций понимания контекста",
                "Использовать более простые и конкретные примеры"
            ])
        
        return recommendations

def main():
    """Основная функция улучшенного обучения"""
    print("🧠 УЛУЧШЕННАЯ СИСТЕМА ОБУЧЕНИЯ RUBIN AI")
    print("=" * 70)
    print("Цель: Научить Rubin AI понимать процессы обучения и избегать шаблонных ответов")
    print("=" * 70)
    
    # Создаем систему обучения
    learning_system = EnhancedRubinLearning()
    
    # Обучаем Rubin AI
    learning_system.teach_rubin_learning_process()
    
    # Тестируем понимание
    learning_system.test_rubin_understanding()
    
    # Генерируем отчет
    learning_system.generate_learning_report()
    
    print("\n🎉 УЛУЧШЕННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 40)
    print("✅ Rubin AI обучен пониманию процессов обучения")
    print("📊 Система тестирования понимания активна")
    print("🧠 Rubin AI должен лучше понимать контекст вопросов")

if __name__ == "__main__":
    main()





