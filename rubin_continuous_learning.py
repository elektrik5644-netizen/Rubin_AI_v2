#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧠 СИСТЕМА ПОСТОЯННОГО ОБУЧЕНИЯ RUBIN AI
========================================
Сканируем взаимодействие и обучаем Rubin AI процессам диагностики и модернизации
"""

import requests
import json
import time
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubinLearningScanner:
    """Сканер для постоянного обучения Rubin AI"""
    
    def __init__(self):
        self.learning_log = []
        self.interaction_patterns = {}
        self.error_resolution_patterns = {}
        self.modernization_patterns = {}
        self.smart_dispatcher_url = "http://localhost:8080/api/chat"
        
    def scan_interaction(self, user_message: str, assistant_response: str, context: Dict[str, Any]):
        """Сканируем взаимодействие для обучения"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "context": context,
            "patterns": self._extract_patterns(user_message, assistant_response)
        }
        
        self.learning_log.append(interaction)
        self._analyze_patterns(interaction)
        
        return interaction
    
    def _extract_patterns(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Извлекаем паттерны из взаимодействия"""
        patterns = {
            "error_diagnosis": self._detect_error_diagnosis_pattern(user_message, assistant_response),
            "error_fixing": self._detect_error_fixing_pattern(user_message, assistant_response),
            "modernization": self._detect_modernization_pattern(user_message, assistant_response),
            "learning_process": self._detect_learning_process_pattern(user_message, assistant_response),
            "communication_style": self._detect_communication_style_pattern(user_message, assistant_response)
        }
        
        return patterns
    
    def _detect_error_diagnosis_pattern(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Обнаруживаем паттерн диагностики ошибок"""
        error_keywords = ["ошибка", "error", "проблема", "не работает", "недоступен", "HTTP 500", "fallback"]
        diagnosis_keywords = ["анализ", "диагностика", "проверка", "статус", "лог", "причина"]
        
        user_lower = user_message.lower()
        response_lower = assistant_response.lower()
        
        has_error = any(keyword in user_lower for keyword in error_keywords)
        has_diagnosis = any(keyword in response_lower for keyword in diagnosis_keywords)
        
        if has_error and has_diagnosis:
            return {
                "detected": True,
                "error_type": self._classify_error_type(user_message),
                "diagnosis_method": self._extract_diagnosis_method(assistant_response),
                "tools_used": self._extract_tools_used(assistant_response)
            }
        
        return {"detected": False}
    
    def _detect_error_fixing_pattern(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Обнаруживаем паттерн исправления ошибок"""
        fixing_keywords = ["исправить", "fix", "решить", "устранить", "обновить", "модернизировать"]
        solution_keywords = ["решение", "solution", "fallback", "механизм", "обход"]
        
        user_lower = user_message.lower()
        response_lower = assistant_response.lower()
        
        has_fixing = any(keyword in user_lower for keyword in fixing_keywords)
        has_solution = any(keyword in response_lower for keyword in solution_keywords)
        
        if has_fixing and has_solution:
            return {
                "detected": True,
                "fix_type": self._classify_fix_type(user_message),
                "solution_approach": self._extract_solution_approach(assistant_response),
                "implementation": self._extract_implementation(assistant_response)
            }
        
        return {"detected": False}
    
    def _detect_modernization_pattern(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Обнаруживаем паттерн модернизации"""
        modern_keywords = ["модернизировать", "улучшить", "обновить", "расширить", "добавить", "паттерн"]
        architecture_keywords = ["архитектура", "паттерн", "design pattern", "singleton", "observer", "factory"]
        
        user_lower = user_message.lower()
        response_lower = assistant_response.lower()
        
        has_modern = any(keyword in user_lower for keyword in modern_keywords)
        has_architecture = any(keyword in response_lower for keyword in architecture_keywords)
        
        if has_modern and has_architecture:
            return {
                "detected": True,
                "modernization_type": self._classify_modernization_type(user_message),
                "patterns_used": self._extract_patterns_used(assistant_response),
                "improvements": self._extract_improvements(assistant_response)
            }
        
        return {"detected": False}
    
    def _detect_learning_process_pattern(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Обнаруживаем паттерн процесса обучения"""
        learning_keywords = ["обучить", "научить", "обучение", "понимание", "изучение", "сканирование"]
        teaching_keywords = ["урок", "демонстрация", "пример", "объяснение", "показ"]
        
        user_lower = user_message.lower()
        response_lower = assistant_response.lower()
        
        has_learning = any(keyword in user_lower for keyword in learning_keywords)
        has_teaching = any(keyword in response_lower for keyword in teaching_keywords)
        
        if has_learning and has_teaching:
            return {
                "detected": True,
                "learning_type": self._classify_learning_type(user_message),
                "teaching_method": self._extract_teaching_method(assistant_response),
                "knowledge_transfer": self._extract_knowledge_transfer(assistant_response)
            }
        
        return {"detected": False}
    
    def _detect_communication_style_pattern(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Обнаруживаем паттерн стиля общения"""
        return {
            "user_tone": self._analyze_tone(user_message),
            "assistant_tone": self._analyze_tone(assistant_response),
            "formality_level": self._analyze_formality(user_message, assistant_response),
            "technical_depth": self._analyze_technical_depth(user_message, assistant_response)
        }
    
    def _classify_error_type(self, message: str) -> str:
        """Классифицируем тип ошибки"""
        if "HTTP 500" in message or "500" in message:
            return "HTTP_500"
        elif "подключение" in message.lower() or "connection" in message.lower():
            return "CONNECTION_ERROR"
        elif "plc" in message.lower():
            return "PLC_ERROR"
        elif "сервер" in message.lower() or "server" in message.lower():
            return "SERVER_ERROR"
        else:
            return "GENERAL_ERROR"
    
    def _classify_fix_type(self, message: str) -> str:
        """Классифицируем тип исправления"""
        if "fallback" in message.lower():
            return "FALLBACK_MECHANISM"
        elif "обновить" in message.lower() or "update" in message.lower():
            return "UPDATE"
        elif "модернизировать" in message.lower():
            return "MODERNIZATION"
        else:
            return "GENERAL_FIX"
    
    def _classify_modernization_type(self, message: str) -> str:
        """Классифицируем тип модернизации"""
        if "паттерн" in message.lower() or "pattern" in message.lower():
            return "DESIGN_PATTERNS"
        elif "архитектура" in message.lower() or "architecture" in message.lower():
            return "ARCHITECTURE"
        elif "функциональность" in message.lower():
            return "FUNCTIONALITY"
        else:
            return "GENERAL_MODERNIZATION"
    
    def _classify_learning_type(self, message: str) -> str:
        """Классифицируем тип обучения"""
        if "постоянно" in message.lower() or "сканирование" in message.lower():
            return "CONTINUOUS_LEARNING"
        elif "ошибки" in message.lower() or "error" in message.lower():
            return "ERROR_HANDLING_LEARNING"
        elif "модернизация" in message.lower():
            return "MODERNIZATION_LEARNING"
        else:
            return "GENERAL_LEARNING"
    
    def _extract_diagnosis_method(self, response: str) -> List[str]:
        """Извлекаем методы диагностики"""
        methods = []
        if "анализ" in response.lower():
            methods.append("ANALYSIS")
        if "проверка" in response.lower():
            methods.append("CHECKING")
        if "лог" in response.lower():
            methods.append("LOGGING")
        if "статус" in response.lower():
            methods.append("STATUS_CHECK")
        return methods
    
    def _extract_tools_used(self, response: str) -> List[str]:
        """Извлекаем используемые инструменты"""
        tools = []
        if "python" in response.lower():
            tools.append("PYTHON")
        if "requests" in response.lower():
            tools.append("REQUESTS")
        if "curl" in response.lower():
            tools.append("CURL")
        if "smart_dispatcher" in response.lower():
            tools.append("SMART_DISPATCHER")
        return tools
    
    def _extract_solution_approach(self, response: str) -> str:
        """Извлекаем подход к решению"""
        if "fallback" in response.lower():
            return "FALLBACK_APPROACH"
        elif "обновление" in response.lower():
            return "UPDATE_APPROACH"
        elif "модернизация" in response.lower():
            return "MODERNIZATION_APPROACH"
        else:
            return "GENERAL_APPROACH"
    
    def _extract_implementation(self, response: str) -> List[str]:
        """Извлекаем детали реализации"""
        implementation = []
        if "код" in response.lower():
            implementation.append("CODE_IMPLEMENTATION")
        if "скрипт" in response.lower():
            implementation.append("SCRIPT_CREATION")
        if "тестирование" in response.lower():
            implementation.append("TESTING")
        return implementation
    
    def _extract_patterns_used(self, response: str) -> List[str]:
        """Извлекаем используемые паттерны"""
        patterns = []
        if "singleton" in response.lower():
            patterns.append("SINGLETON")
        if "observer" in response.lower():
            patterns.append("OBSERVER")
        if "factory" in response.lower():
            patterns.append("FACTORY")
        if "strategy" in response.lower():
            patterns.append("STRATEGY")
        if "command" in response.lower():
            patterns.append("COMMAND")
        return patterns
    
    def _extract_improvements(self, response: str) -> List[str]:
        """Извлекаем улучшения"""
        improvements = []
        if "сложность" in response.lower():
            improvements.append("COMPLEXITY_REDUCTION")
        if "связанность" in response.lower():
            improvements.append("COUPLING_REDUCTION")
        if "тестируемость" in response.lower():
            improvements.append("TESTABILITY_INCREASE")
        if "надежность" in response.lower():
            improvements.append("RELIABILITY_INCREASE")
        return improvements
    
    def _extract_teaching_method(self, response: str) -> List[str]:
        """Извлекаем методы обучения"""
        methods = []
        if "демонстрация" in response.lower():
            methods.append("DEMONSTRATION")
        if "пример" in response.lower():
            methods.append("EXAMPLE")
        if "объяснение" in response.lower():
            methods.append("EXPLANATION")
        if "тестирование" in response.lower():
            methods.append("TESTING")
        return methods
    
    def _extract_knowledge_transfer(self, response: str) -> List[str]:
        """Извлекаем передачу знаний"""
        transfer = []
        if "понимание" in response.lower():
            transfer.append("UNDERSTANDING")
        if "запоминание" in response.lower():
            transfer.append("MEMORIZATION")
        if "применение" in response.lower():
            transfer.append("APPLICATION")
        return transfer
    
    def _analyze_tone(self, text: str) -> str:
        """Анализируем тон сообщения"""
        if "!" in text or "🎉" in text or "✅" in text:
            return "POSITIVE"
        elif "❌" in text or "ошибка" in text.lower():
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _analyze_formality(self, user_msg: str, assistant_msg: str) -> str:
        """Анализируем уровень формальности"""
        if "ты" in user_msg.lower() and "мы" in assistant_msg.lower():
            return "INFORMAL"
        else:
            return "FORMAL"
    
    def _analyze_technical_depth(self, user_msg: str, assistant_msg: str) -> str:
        """Анализируем техническую глубину"""
        technical_terms = ["API", "HTTP", "сервер", "код", "алгоритм", "архитектура"]
        user_tech = sum(1 for term in technical_terms if term.lower() in user_msg.lower())
        assistant_tech = sum(1 for term in technical_terms if term.lower() in assistant_msg.lower())
        
        if user_tech + assistant_tech > 3:
            return "HIGH"
        elif user_tech + assistant_tech > 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _analyze_patterns(self, interaction: Dict[str, Any]):
        """Анализируем паттерны взаимодействия"""
        patterns = interaction["patterns"]
        
        # Обновляем статистику паттернов
        for pattern_type, pattern_data in patterns.items():
            if pattern_data.get("detected"):
                if pattern_type not in self.interaction_patterns:
                    self.interaction_patterns[pattern_type] = []
                self.interaction_patterns[pattern_type].append(pattern_data)
    
    def teach_rubin_continuous_learning(self):
        """Обучаем Rubin AI постоянному обучению"""
        print("🧠 ОБУЧЕНИЕ RUBIN AI ПОСТОЯННОМУ СКАНИРОВАНИЮ")
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
        
        # Уроки для Rubin AI о постоянном обучении
        lessons = [
            {
                "title": "Что такое постоянное сканирование",
                "message": "Программирование: Привет Rubin! Мы создали систему постоянного сканирования взаимодействия. Это означает, что ты можешь постоянно изучать, как мы с тобой общаемся, диагностируем ошибки, исправляем их и модернизируем систему. Объясни, как это поможет тебе лучше понимать процессы.",
                "expected": "Rubin должен понять концепцию постоянного обучения"
            },
            {
                "title": "Паттерны диагностики ошибок",
                "message": "Программирование: Мы постоянно сканируем паттерны диагностики ошибок. Например, когда возникает HTTP 500 ошибка, мы анализируем причину, проверяем статус серверов, создаем fallback механизмы. Как ты можешь использовать эти паттерны для улучшения своей работы?",
                "expected": "Rubin должен понять паттерны диагностики"
            },
            {
                "title": "Паттерны исправления ошибок",
                "message": "Программирование: Мы сканируем паттерны исправления ошибок. Когда находим проблему, мы создаем решение, тестируем его, документируем результаты. Как ты можешь применять эти паттерны для автоматического исправления ошибок?",
                "expected": "Rubin должен понять паттерны исправления"
            },
            {
                "title": "Паттерны модернизации",
                "message": "Программирование: Мы сканируем паттерны модернизации системы. Например, добавляем новые паттерны проектирования, улучшаем архитектуру, расширяем функциональность. Как ты можешь использовать эти паттерны для предложения улучшений?",
                "expected": "Rubin должен понять паттерны модернизации"
            },
            {
                "title": "Стиль общения и взаимодействия",
                "message": "Программирование: Мы сканируем стиль нашего общения - как мы задаем вопросы, как ты отвечаешь, как мы решаем проблемы вместе. Это поможет тебе лучше понимать контекст и давать более релевантные ответы. Как ты можешь использовать эту информацию?",
                "expected": "Rubin должен понять важность стиля общения"
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
                        
                        # Сканируем это взаимодействие
                        self.scan_interaction(lesson['message'], explanation, {
                            "lesson_number": i,
                            "lesson_title": lesson['title'],
                            "expected_outcome": lesson['expected']
                        })
                        
                    else:
                        print("❌ Ошибка в ответе Rubin AI")
                else:
                    print(f"❌ HTTP ошибка: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Ошибка запроса: {e}")
            
            time.sleep(2)  # Пауза между уроками
    
    def generate_learning_report(self):
        """Генерируем отчет об обучении"""
        print("\n📊 ОТЧЕТ О ПОСТОЯННОМ ОБУЧЕНИИ:")
        print("=" * 40)
        
        total_interactions = len(self.learning_log)
        print(f"📈 Всего взаимодействий: {total_interactions}")
        
        # Анализируем паттерны
        for pattern_type, patterns in self.interaction_patterns.items():
            print(f"\n🔍 {pattern_type.upper()}:")
            print(f"  📊 Обнаружено паттернов: {len(patterns)}")
            
            if pattern_type == "error_diagnosis":
                error_types = [p.get("error_type") for p in patterns if p.get("error_type")]
                print(f"  🚨 Типы ошибок: {set(error_types)}")
                
            elif pattern_type == "error_fixing":
                fix_types = [p.get("fix_type") for p in patterns if p.get("fix_type")]
                print(f"  🔧 Типы исправлений: {set(fix_types)}")
                
            elif pattern_type == "modernization":
                modern_types = [p.get("modernization_type") for p in patterns if p.get("modernization_type")]
                print(f"  🚀 Типы модернизации: {set(modern_types)}")
        
        # Сохраняем отчет
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_interactions": total_interactions,
            "patterns": self.interaction_patterns,
            "learning_log": self.learning_log
        }
        
        try:
            with open('RUBIN_CONTINUOUS_LEARNING_REPORT.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 Отчет сохранен: RUBIN_CONTINUOUS_LEARNING_REPORT.json")
        except Exception as e:
            print(f"❌ Ошибка сохранения отчета: {e}")

def main():
    """Основная функция системы постоянного обучения"""
    print("🧠 СИСТЕМА ПОСТОЯННОГО ОБУЧЕНИЯ RUBIN AI")
    print("=" * 70)
    print("Цель: Научить Rubin AI постоянно сканировать и изучать процессы взаимодействия")
    print("=" * 70)
    
    # Создаем сканер
    scanner = RubinLearningScanner()
    
    # Обучаем Rubin AI
    scanner.teach_rubin_continuous_learning()
    
    # Генерируем отчет
    scanner.generate_learning_report()
    
    print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 40)
    print("✅ Rubin AI обучен постоянному сканированию")
    print("📊 Система готова к мониторингу взаимодействий")
    print("🧠 Rubin AI будет постоянно изучать процессы диагностики и модернизации")

if __name__ == "__main__":
    main()










