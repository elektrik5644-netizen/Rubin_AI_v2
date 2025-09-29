#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📡 СИСТЕМА МОНИТОРИНГА ВЗАИМОДЕЙСТВИЯ В РЕАЛЬНОМ ВРЕМЕНИ
========================================================
Постоянно сканируем и анализируем взаимодействие с Rubin AI
"""

import requests
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import queue

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeInteractionMonitor:
    """Монитор взаимодействия в реальном времени"""
    
    def __init__(self):
        self.interaction_queue = queue.Queue()
        self.learning_patterns = {
            "error_diagnosis": [],
            "error_fixing": [],
            "modernization": [],
            "learning_process": [],
            "communication_style": []
        }
        self.rubin_knowledge_base = {}
        self.monitoring_active = False
        self.smart_dispatcher_url = "http://localhost:8080/api/chat"
        
    def start_monitoring(self):
        """Запускаем мониторинг в реальном времени"""
        print("📡 ЗАПУСК МОНИТОРИНГА ВЗАИМОДЕЙСТВИЯ В РЕАЛЬНОМ ВРЕМЕНИ")
        print("=" * 60)
        
        self.monitoring_active = True
        
        # Запускаем потоки мониторинга
        monitor_thread = threading.Thread(target=self._monitor_interactions)
        analyzer_thread = threading.Thread(target=self._analyze_patterns)
        teacher_thread = threading.Thread(target=self._teach_rubin_continuously)
        
        monitor_thread.daemon = True
        analyzer_thread.daemon = True
        teacher_thread.daemon = True
        
        monitor_thread.start()
        analyzer_thread.start()
        teacher_thread.start()
        
        print("✅ Мониторинг запущен")
        print("📊 Анализ паттернов активен")
        print("🧠 Непрерывное обучение Rubin AI активно")
        
        return monitor_thread, analyzer_thread, teacher_thread
    
    def _monitor_interactions(self):
        """Мониторим взаимодействия"""
        while self.monitoring_active:
            try:
                # Симулируем получение взаимодействий
                # В реальной системе здесь был бы API для получения логов
                interaction = self._simulate_interaction()
                if interaction:
                    self.interaction_queue.put(interaction)
                
                time.sleep(5)  # Проверяем каждые 5 секунд
                
            except Exception as e:
                logger.error(f"Ошибка мониторинга: {e}")
                time.sleep(10)
    
    def _simulate_interaction(self) -> Optional[Dict[str, Any]]:
        """Симулируем взаимодействие для демонстрации"""
        # В реальной системе здесь был бы API для получения реальных взаимодействий
        interactions = [
            {
                "user_message": "HTTP 500 ошибка при анализе PLC файла",
                "assistant_response": "Анализирую проблему: сервер controllers недоступен, применяю fallback механизм",
                "context": {"error_type": "HTTP_500", "file_type": "PLC"}
            },
            {
                "user_message": "Как модернизировать архитектуру VMB630?",
                "assistant_response": "Предлагаю использовать паттерны Singleton, Observer, Factory для улучшения архитектуры",
                "context": {"topic": "modernization", "system": "VMB630"}
            },
            {
                "user_message": "Обучи Rubin AI пониманию ошибок",
                "assistant_response": "Создаю систему обучения с примерами диагностики и исправления ошибок",
                "context": {"learning_type": "error_handling", "target": "Rubin_AI"}
            }
        ]
        
        # Возвращаем случайное взаимодействие
        import random
        if random.random() < 0.3:  # 30% вероятность
            return random.choice(interactions)
        return None
    
    def _analyze_patterns(self):
        """Анализируем паттерны взаимодействий"""
        while self.monitoring_active:
            try:
                if not self.interaction_queue.empty():
                    interaction = self.interaction_queue.get()
                    self._process_interaction(interaction)
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Ошибка анализа паттернов: {e}")
                time.sleep(5)
    
    def _process_interaction(self, interaction: Dict[str, Any]):
        """Обрабатываем взаимодействие"""
        user_msg = interaction["user_message"]
        assistant_msg = interaction["assistant_response"]
        context = interaction["context"]
        
        # Анализируем паттерны
        patterns = self._extract_patterns(user_msg, assistant_msg, context)
        
        # Обновляем базу знаний
        self._update_knowledge_base(patterns, context)
        
        # Логируем для обучения
        logger.info(f"Обработано взаимодействие: {patterns}")
    
    def _extract_patterns(self, user_msg: str, assistant_msg: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекаем паттерны из взаимодействия"""
        patterns = {
            "timestamp": datetime.now().isoformat(),
            "error_diagnosis": self._detect_error_diagnosis(user_msg, assistant_msg),
            "error_fixing": self._detect_error_fixing(user_msg, assistant_msg),
            "modernization": self._detect_modernization(user_msg, assistant_msg),
            "learning_process": self._detect_learning_process(user_msg, assistant_msg),
            "communication_style": self._analyze_communication_style(user_msg, assistant_msg),
            "context": context
        }
        
        return patterns
    
    def _detect_error_diagnosis(self, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        """Обнаруживаем диагностику ошибок"""
        error_keywords = ["ошибка", "error", "проблема", "не работает", "HTTP 500"]
        diagnosis_keywords = ["анализ", "диагностика", "проверка", "причина"]
        
        has_error = any(keyword in user_msg.lower() for keyword in error_keywords)
        has_diagnosis = any(keyword in assistant_msg.lower() for keyword in diagnosis_keywords)
        
        if has_error and has_diagnosis:
            return {
                "detected": True,
                "error_type": self._classify_error(user_msg),
                "diagnosis_method": self._extract_diagnosis_method(assistant_msg),
                "confidence": 0.8
            }
        
        return {"detected": False}
    
    def _detect_error_fixing(self, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        """Обнаруживаем исправление ошибок"""
        fixing_keywords = ["исправить", "fix", "решить", "устранить", "fallback"]
        solution_keywords = ["решение", "solution", "механизм", "обход"]
        
        has_fixing = any(keyword in user_msg.lower() for keyword in fixing_keywords)
        has_solution = any(keyword in assistant_msg.lower() for keyword in solution_keywords)
        
        if has_fixing and has_solution:
            return {
                "detected": True,
                "fix_type": self._classify_fix(user_msg),
                "solution_approach": self._extract_solution_approach(assistant_msg),
                "confidence": 0.7
            }
        
        return {"detected": False}
    
    def _detect_modernization(self, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        """Обнаруживаем модернизацию"""
        modern_keywords = ["модернизировать", "улучшить", "обновить", "паттерн", "архитектура"]
        pattern_keywords = ["singleton", "observer", "factory", "strategy", "command"]
        
        has_modern = any(keyword in user_msg.lower() for keyword in modern_keywords)
        has_patterns = any(keyword in assistant_msg.lower() for keyword in pattern_keywords)
        
        if has_modern and has_patterns:
            return {
                "detected": True,
                "modernization_type": self._classify_modernization(user_msg),
                "patterns_used": self._extract_patterns_used(assistant_msg),
                "confidence": 0.9
            }
        
        return {"detected": False}
    
    def _detect_learning_process(self, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        """Обнаруживаем процесс обучения"""
        learning_keywords = ["обучить", "научить", "обучение", "понимание", "сканирование"]
        teaching_keywords = ["урок", "демонстрация", "пример", "объяснение"]
        
        has_learning = any(keyword in user_msg.lower() for keyword in learning_keywords)
        has_teaching = any(keyword in assistant_msg.lower() for keyword in teaching_keywords)
        
        if has_learning and has_teaching:
            return {
                "detected": True,
                "learning_type": self._classify_learning(user_msg),
                "teaching_method": self._extract_teaching_method(assistant_msg),
                "confidence": 0.8
            }
        
        return {"detected": False}
    
    def _analyze_communication_style(self, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        """Анализируем стиль общения"""
        return {
            "user_tone": self._analyze_tone(user_msg),
            "assistant_tone": self._analyze_tone(assistant_msg),
            "technical_level": self._analyze_technical_level(user_msg, assistant_msg),
            "formality": self._analyze_formality(user_msg, assistant_msg)
        }
    
    def _classify_error(self, message: str) -> str:
        """Классифицируем ошибку"""
        if "HTTP 500" in message:
            return "HTTP_500"
        elif "plc" in message.lower():
            return "PLC_ERROR"
        elif "сервер" in message.lower():
            return "SERVER_ERROR"
        else:
            return "GENERAL_ERROR"
    
    def _classify_fix(self, message: str) -> str:
        """Классифицируем исправление"""
        if "fallback" in message.lower():
            return "FALLBACK_MECHANISM"
        elif "обновить" in message.lower():
            return "UPDATE"
        else:
            return "GENERAL_FIX"
    
    def _classify_modernization(self, message: str) -> str:
        """Классифицируем модернизацию"""
        if "паттерн" in message.lower():
            return "DESIGN_PATTERNS"
        elif "архитектура" in message.lower():
            return "ARCHITECTURE"
        else:
            return "GENERAL_MODERNIZATION"
    
    def _classify_learning(self, message: str) -> str:
        """Классифицируем обучение"""
        if "постоянно" in message.lower():
            return "CONTINUOUS_LEARNING"
        elif "ошибки" in message.lower():
            return "ERROR_HANDLING_LEARNING"
        else:
            return "GENERAL_LEARNING"
    
    def _extract_diagnosis_method(self, response: str) -> List[str]:
        """Извлекаем методы диагностики"""
        methods = []
        if "анализ" in response.lower():
            methods.append("ANALYSIS")
        if "проверка" in response.lower():
            methods.append("CHECKING")
        if "статус" in response.lower():
            methods.append("STATUS_CHECK")
        return methods
    
    def _extract_solution_approach(self, response: str) -> str:
        """Извлекаем подход к решению"""
        if "fallback" in response.lower():
            return "FALLBACK_APPROACH"
        elif "обновление" in response.lower():
            return "UPDATE_APPROACH"
        else:
            return "GENERAL_APPROACH"
    
    def _extract_patterns_used(self, response: str) -> List[str]:
        """Извлекаем используемые паттерны"""
        patterns = []
        if "singleton" in response.lower():
            patterns.append("SINGLETON")
        if "observer" in response.lower():
            patterns.append("OBSERVER")
        if "factory" in response.lower():
            patterns.append("FACTORY")
        return patterns
    
    def _extract_teaching_method(self, response: str) -> List[str]:
        """Извлекаем методы обучения"""
        methods = []
        if "демонстрация" in response.lower():
            methods.append("DEMONSTRATION")
        if "пример" in response.lower():
            methods.append("EXAMPLE")
        if "объяснение" in response.lower():
            methods.append("EXPLANATION")
        return methods
    
    def _analyze_tone(self, text: str) -> str:
        """Анализируем тон"""
        if "!" in text or "🎉" in text:
            return "POSITIVE"
        elif "❌" in text or "ошибка" in text.lower():
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _analyze_technical_level(self, user_msg: str, assistant_msg: str) -> str:
        """Анализируем технический уровень"""
        tech_terms = ["API", "HTTP", "сервер", "код", "алгоритм"]
        total_tech = sum(1 for term in tech_terms if term.lower() in (user_msg + assistant_msg).lower())
        
        if total_tech > 3:
            return "HIGH"
        elif total_tech > 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _analyze_formality(self, user_msg: str, assistant_msg: str) -> str:
        """Анализируем формальность"""
        if "ты" in user_msg.lower() and "мы" in assistant_msg.lower():
            return "INFORMAL"
        else:
            return "FORMAL"
    
    def _update_knowledge_base(self, patterns: Dict[str, Any], context: Dict[str, Any]):
        """Обновляем базу знаний Rubin AI"""
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and pattern_data.get("detected"):
                if pattern_type not in self.rubin_knowledge_base:
                    self.rubin_knowledge_base[pattern_type] = []
                
                knowledge_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "pattern": pattern_data,
                    "context": context,
                    "confidence": pattern_data.get("confidence", 0.5)
                }
                
                self.rubin_knowledge_base[pattern_type].append(knowledge_entry)
                
                # Ограничиваем размер базы знаний
                if len(self.rubin_knowledge_base[pattern_type]) > 100:
                    self.rubin_knowledge_base[pattern_type] = self.rubin_knowledge_base[pattern_type][-100:]
    
    def _teach_rubin_continuously(self):
        """Непрерывно обучаем Rubin AI"""
        while self.monitoring_active:
            try:
                # Проверяем, есть ли новые знания для обучения
                if self._has_new_knowledge():
                    self._teach_rubin_new_patterns()
                
                time.sleep(30)  # Обучаем каждые 30 секунд
                
            except Exception as e:
                logger.error(f"Ошибка непрерывного обучения: {e}")
                time.sleep(60)
    
    def _has_new_knowledge(self) -> bool:
        """Проверяем, есть ли новые знания"""
        total_patterns = sum(len(patterns) for patterns in self.rubin_knowledge_base.values())
        return total_patterns > 0
    
    def _teach_rubin_new_patterns(self):
        """Обучаем Rubin AI новым паттернам"""
        try:
            # Создаем сообщение для обучения
            learning_message = self._create_learning_message()
            
            if learning_message:
                response = requests.post(self.smart_dispatcher_url, 
                                      json={'message': learning_message})
                
                if response.status_code == 200:
                    logger.info("Rubin AI обучен новым паттернам")
                else:
                    logger.warning(f"Ошибка обучения Rubin AI: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Ошибка обучения Rubin AI: {e}")
    
    def _create_learning_message(self) -> Optional[str]:
        """Создаем сообщение для обучения"""
        if not self.rubin_knowledge_base:
            return None
        
        # Анализируем последние паттерны
        recent_patterns = []
        for pattern_type, patterns in self.rubin_knowledge_base.items():
            if patterns:
                recent_patterns.append({
                    "type": pattern_type,
                    "count": len(patterns),
                    "latest": patterns[-1]
                })
        
        if recent_patterns:
            message = "Программирование: Rubin, я наблюдаю новые паттерны в нашем взаимодействии: "
            for pattern in recent_patterns:
                message += f"{pattern['type']} ({pattern['count']} случаев), "
            
            message += "Как ты можешь использовать эти паттерны для улучшения своей работы?"
            return message
        
        return None
    
    def stop_monitoring(self):
        """Останавливаем мониторинг"""
        self.monitoring_active = False
        print("🛑 Мониторинг остановлен")
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Получаем отчет об обучении"""
        return {
            "timestamp": datetime.now().isoformat(),
            "knowledge_base": self.rubin_knowledge_base,
            "total_patterns": sum(len(patterns) for patterns in self.rubin_knowledge_base.values()),
            "pattern_types": list(self.rubin_knowledge_base.keys())
        }

def main():
    """Основная функция мониторинга"""
    print("📡 СИСТЕМА МОНИТОРИНГА ВЗАИМОДЕЙСТВИЯ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 70)
    print("Цель: Постоянно сканировать и анализировать взаимодействие с Rubin AI")
    print("=" * 70)
    
    # Создаем монитор
    monitor = RealTimeInteractionMonitor()
    
    try:
        # Запускаем мониторинг
        threads = monitor.start_monitoring()
        
        print("\n🔄 Мониторинг активен...")
        print("Нажмите Ctrl+C для остановки")
        
        # Ждем завершения
        for thread in threads:
            thread.join()
            
    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки")
        monitor.stop_monitoring()
        
        # Генерируем финальный отчет
        report = monitor.get_learning_report()
        print(f"\n📊 ФИНАЛЬНЫЙ ОТЧЕТ:")
        print(f"📈 Всего паттернов: {report['total_patterns']}")
        print(f"🔍 Типы паттернов: {report['pattern_types']}")
        
        # Сохраняем отчет
        try:
            with open('REAL_TIME_MONITORING_REPORT.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print("📄 Отчет сохранен: REAL_TIME_MONITORING_REPORT.json")
        except Exception as e:
            print(f"❌ Ошибка сохранения отчета: {e}")

if __name__ == "__main__":
    main()










