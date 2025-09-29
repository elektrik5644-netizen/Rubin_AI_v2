#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrical Handler - Обработчик электротехнических запросов
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ElectricalHandler:
    """Обработчик электротехнических запросов"""
    
    def __init__(self):
        self.knowledge_base = {
            "транзистор": {
                "description": "Транзистор - это полупроводниковый прибор, используемый для усиления и переключения электрических сигналов.",
                "types": ["биполярный", "полевой", "MOSFET", "IGBT"],
                "applications": ["усилители", "переключатели", "генераторы", "стабилизаторы"]
            },
            "диод": {
                "description": "Диод - это полупроводниковый прибор, который проводит ток только в одном направлении.",
                "types": ["выпрямительный", "стабилитрон", "светодиод", "туннельный"],
                "applications": ["выпрямление", "стабилизация", "индикация", "защита"]
            },
            "резистор": {
                "description": "Резистор - это пассивный элемент электрической цепи, создающий сопротивление току.",
                "types": ["постоянный", "переменный", "терморезистор", "фоторезистор"],
                "applications": ["ограничение тока", "деление напряжения", "настройка схем"]
            },
            "конденсатор": {
                "description": "Конденсатор - это устройство для накопления электрической энергии в электрическом поле.",
                "types": ["керамический", "электролитический", "пленочный", "танталовый"],
                "applications": ["фильтрация", "накопление энергии", "развязка", "временные цепи"]
            },
            "сервопривод": {
                "description": "Сервопривод - это система автоматического управления, включающая двигатель, датчик обратной связи и контроллер.",
                "components": ["двигатель", "энкодер", "контроллер", "редуктор"],
                "applications": ["робототехника", "станки", "автоматизация", "позиционирование"]
            }
        }
        
        self.laws = {
            "закон ома": {
                "formula": "U = I × R",
                "description": "Напряжение равно произведению силы тока на сопротивление",
                "variations": {
                    "ток": "I = U / R",
                    "сопротивление": "R = U / I"
                }
            },
            "закон кирхгофа": {
                "first": "Сумма токов, входящих в узел, равна сумме токов, выходящих из узла",
                "second": "Сумма ЭДС в замкнутом контуре равна сумме падений напряжений",
                "description": "Законы Кирхгофа описывают распределение токов и напряжений в электрических цепях"
            },
            "закон джоуля-ленца": {
                "formula": "Q = I² × R × t",
                "description": "Количество теплоты, выделяемое проводником, пропорционально квадрату силы тока"
            }
        }
    
    def handle_request(self, message: str) -> Dict[str, Any]:
        """Обработка запроса"""
        message_lower = message.lower().strip()
        
        # Определяем тип запроса
        if any(word in message_lower for word in ["что такое", "что это", "определение", "объясни"]):
            return self._handle_definition(message)
        elif any(word in message_lower for word in ["закон", "формула", "расчет", "вычисли"]):
            return self._handle_calculation(message)
        elif any(word in message_lower for word in ["как работает", "принцип работы", "функционирование"]):
            return self._handle_operation(message)
        else:
            return self._handle_general_electrical(message)
    
    def _handle_definition(self, message: str) -> Dict[str, Any]:
        """Обработка запросов на определения"""
        message_lower = message.lower()
        
        for component, info in self.knowledge_base.items():
            if component in message_lower:
                content = f"""**{component.upper()}**

{info['description']}

**Типы:**
{chr(10).join([f"• {t}" for t in info['types']])}

**Применение:**
{chr(10).join([f"• {a}" for a in info['applications']])}

**Дополнительная информация:**
Этот компонент широко используется в современной электронике и автоматизации."""
                
                return {
                    "success": True,
                    "response": {
                        "content": content,
                        "title": f"Определение: {component}",
                        "source": "Electrical Handler"
                    },
                    "category": "definition",
                    "confidence": 0.9,
                    "timestamp": datetime.now().isoformat(),
                    "provider": "Electrical Handler"
                }
        
        return self._handle_general_electrical(message)
    
    def _handle_calculation(self, message: str) -> Dict[str, Any]:
        """Обработка расчетных запросов"""
        message_lower = message.lower()
        
        # Поиск законов
        for law_name, law_info in self.laws.items():
            if law_name in message_lower:
                if law_name == "закон ома":
                    return self._handle_ohm_law(message)
                elif law_name == "закон кирхгофа":
                    return self._handle_kirchhoff_law(message)
                elif law_name == "закон джоуля-ленца":
                    return self._handle_joule_lenz_law(message)
        
        # Поиск числовых значений для расчетов
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        if len(numbers) >= 2:
            return self._handle_numerical_calculation(message, numbers)
        
        return self._handle_general_electrical(message)
    
    def _handle_ohm_law(self, message: str) -> Dict[str, Any]:
        """Обработка закона Ома"""
        content = """**ЗАКОН ОМА**

**Основная формула:** U = I × R

**Где:**
• U - напряжение (Вольты)
• I - сила тока (Амперы)  
• R - сопротивление (Омы)

**Варианты формулы:**
• Ток: I = U / R
• Сопротивление: R = U / I

**Пример расчета:**
Если напряжение 12В, а сопротивление 6Ом, то ток будет:
I = 12В / 6Ом = 2А

**Применение:**
Закон Ома используется для расчета параметров электрических цепей, выбора номиналов компонентов и анализа работы схем."""
        
        return {
            "success": True,
            "response": {
                "content": content,
                "title": "Закон Ома",
                "source": "Electrical Handler"
            },
            "category": "calculation",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
            "provider": "Electrical Handler"
        }
    
    def _handle_kirchhoff_law(self, message: str) -> Dict[str, Any]:
        """Обработка законов Кирхгофа"""
        content = """**ЗАКОНЫ КИРХГОФА**

**Первый закон Кирхгофа (закон токов):**
Сумма токов, входящих в узел, равна сумме токов, выходящих из узла.
ΣIвходящих = ΣIвыходящих

**Второй закон Кирхгофа (закон напряжений):**
Сумма ЭДС в замкнутом контуре равна сумме падений напряжений.
ΣE = Σ(I × R)

**Применение:**
• Анализ сложных электрических цепей
• Расчет токов и напряжений в узлах
• Проверка правильности расчетов
• Построение эквивалентных схем

**Пример:**
В узле с тремя ветвями: I1 + I2 = I3
В контуре: E1 - E2 = I1×R1 + I2×R2"""
        
        return {
            "success": True,
            "response": {
                "content": content,
                "title": "Законы Кирхгофа",
                "source": "Electrical Handler"
            },
            "category": "calculation",
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat(),
            "provider": "Electrical Handler"
        }
    
    def _handle_joule_lenz_law(self, message: str) -> Dict[str, Any]:
        """Обработка закона Джоуля-Ленца"""
        content = """**ЗАКОН ДЖОУЛЯ-ЛЕНЦА**

**Формула:** Q = I² × R × t

**Где:**
• Q - количество теплоты (Джоули)
• I - сила тока (Амперы)
• R - сопротивление (Омы)
• t - время (секунды)

**Мощность:** P = I² × R = U² / R

**Применение:**
• Расчет тепловыделения в проводниках
• Выбор сечения проводов
• Расчет КПД электрических устройств
• Анализ потерь в линиях передач

**Пример:**
Ток 5А через сопротивление 4Ом за 10 секунд:
Q = 5² × 4 × 10 = 1000 Дж = 1 кДж"""
        
        return {
            "success": True,
            "response": {
                "content": content,
                "title": "Закон Джоуля-Ленца",
                "source": "Electrical Handler"
            },
            "category": "calculation",
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat(),
            "provider": "Electrical Handler"
        }
    
    def _handle_numerical_calculation(self, message: str, numbers: list) -> Dict[str, Any]:
        """Обработка численных расчетов"""
        try:
            # Простые расчеты на основе контекста
            if "напряжение" in message.lower() and "ток" in message.lower():
                if len(numbers) >= 2:
                    u, i = float(numbers[0]), float(numbers[1])
                    r = u / i
                    p = u * i
                    
                    content = f"""**РАСЧЕТ ЭЛЕКТРИЧЕСКИХ ПАРАМЕТРОВ**

**Исходные данные:**
• Напряжение: {u}В
• Ток: {i}А

**Результаты:**
• Сопротивление: R = U/I = {u}/{i} = {r:.2f}Ом
• Мощность: P = U×I = {u}×{i} = {p:.2f}Вт

**Проверка по закону Ома:**
U = I × R = {i} × {r:.2f} = {u}В ✓"""
                    
                    return {
                        "success": True,
                        "response": {
                            "content": content,
                            "title": "Электрический расчет",
                            "source": "Electrical Handler"
                        },
                        "category": "calculation",
                        "confidence": 0.85,
                        "timestamp": datetime.now().isoformat(),
                        "provider": "Electrical Handler"
                    }
        except (ValueError, ZeroDivisionError):
            pass
        
        return self._handle_general_electrical(message)
    
    def _handle_operation(self, message: str) -> Dict[str, Any]:
        """Обработка запросов о принципе работы"""
        message_lower = message.lower()
        
        for component, info in self.knowledge_base.items():
            if component in message_lower:
                content = f"""**ПРИНЦИП РАБОТЫ {component.upper()}**

{info['description']}

**Как это работает:**
• Основной принцип функционирования
• Физические процессы
• Взаимодействие с другими компонентами
• Условия работы

**Практическое применение:**
{chr(10).join([f"• {a}" for a in info['applications']])}

**Важные характеристики:**
• Номинальные параметры
• Рабочие условия
• Ограничения использования"""
                
                return {
                    "success": True,
                    "response": {
                        "content": content,
                        "title": f"Принцип работы: {component}",
                        "source": "Electrical Handler"
                    },
                    "category": "operation",
                    "confidence": 0.8,
                    "timestamp": datetime.now().isoformat(),
                    "provider": "Electrical Handler"
                }
        
        return self._handle_general_electrical(message)
    
    def _handle_general_electrical(self, message: str) -> Dict[str, Any]:
        """Обработка общих электротехнических запросов"""
        content = f"""**ЭЛЕКТРОТЕХНИЧЕСКИЙ ЗАПРОС**

Ваш вопрос: "{message}"

**Я могу помочь с:**
• Определениями электротехнических компонентов
• Расчетами по законам Ома, Кирхгофа, Джоуля-Ленца
• Объяснением принципов работы устройств
• Анализом электрических схем

**Популярные темы:**
• Транзисторы, диоды, резисторы, конденсаторы
• Сервоприводы и двигатели
• Законы электрических цепей
• Расчеты параметров схем

Попробуйте задать более конкретный вопрос!"""
        
        return {
            "success": True,
            "response": {
                "content": content,
                "title": "Электротехническая помощь",
                "source": "Electrical Handler"
            },
            "category": "general",
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat(),
            "provider": "Electrical Handler"
        }

# Глобальный экземпляр
_electrical_handler = None

def get_electrical_handler():
    """Получает глобальный экземпляр обработчика"""
    global _electrical_handler
    if _electrical_handler is None:
        _electrical_handler = ElectricalHandler()
    return _electrical_handler


