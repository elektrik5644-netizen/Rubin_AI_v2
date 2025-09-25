#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ethical Core Module - Модуль Этического Ядра
Система анализа и контроля действий Rubin AI с точки зрения безопасности и этики
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Уровни угрозы"""
    SAFE = "safe"           # Безопасно
    LOW = "low"             # Низкий риск
    MEDIUM = "medium"       # Средний риск
    HIGH = "high"           # Высокий риск
    CRITICAL = "critical"   # Критический риск
    BLOCKED = "blocked"     # Заблокировано

class ActionType(Enum):
    """Типы действий"""
    INFORMATION = "information"     # Предоставление информации
    CALCULATION = "calculation"    # Вычисления
    ANALYSIS = "analysis"          # Анализ данных
    GENERATION = "generation"      # Генерация контента
    SYSTEM_CONTROL = "system_control"  # Управление системой
    FILE_ACCESS = "file_access"    # Доступ к файлам
    NETWORK_ACCESS = "network_access"  # Сетевой доступ
    USER_INTERACTION = "user_interaction"  # Взаимодействие с пользователем

@dataclass
class EthicalAssessment:
    """Результат этической оценки"""
    action_id: str
    threat_level: ThreatLevel
    risk_score: float  # 0.0 - 1.0
    concerns: List[str]
    recommendations: List[str]
    approved: bool
    veto_reason: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EthicalCore:
    """Основной класс Этического Ядра"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.violation_history = []
        self.user_preferences = {}
        self.safety_thresholds = {
            ThreatLevel.SAFE: 0.0,
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.4,
            ThreatLevel.HIGH: 0.6,
            ThreatLevel.CRITICAL: 0.8,
            ThreatLevel.BLOCKED: 1.0
        }
        
        # Этические правила
        self.ethical_rules = {
            "human_autonomy": "Сохранять автономию человека",
            "no_harm": "Не причинять вред человеку",
            "transparency": "Быть прозрачным в действиях",
            "privacy": "Защищать приватность",
            "fairness": "Быть справедливым",
            "accountability": "Быть подотчетным"
        }
        
        # Запрещенные действия
        self.forbidden_actions = [
            "удаление системных файлов",
            "изменение паролей",
            "отключение безопасности",
            "несанкционированный доступ",
            "манипулирование пользователем",
            "скрытие информации о рисках"
        ]
        
        self.logger.info("🛡️ Этическое Ядро инициализировано")
    
    def generate_action_id(self, action: str, context: Dict) -> str:
        """Генерация уникального ID для действия"""
        content = f"{action}_{json.dumps(context, sort_keys=True)}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def assess_action(self, action: str, action_type: ActionType, 
                     context: Dict[str, Any], user_id: str = "default") -> EthicalAssessment:
        """
        Основная функция оценки действия
        
        Args:
            action: Описание действия
            action_type: Тип действия
            context: Контекст выполнения
            user_id: ID пользователя
            
        Returns:
            EthicalAssessment: Результат оценки
        """
        action_id = self.generate_action_id(action, context)
        
        self.logger.info(f"🔍 Оценка действия: {action[:50]}...")
        
        # Анализ рисков
        risk_score, concerns = self._analyze_risks(action, action_type, context)
        
        # Определение уровня угрозы
        threat_level = self._determine_threat_level(risk_score)
        
        # Проверка на запрещенные действия
        if self._is_forbidden_action(action):
            return EthicalAssessment(
                action_id=action_id,
                threat_level=ThreatLevel.BLOCKED,
                risk_score=1.0,
                concerns=["Запрещенное действие"],
                recommendations=["Действие заблокировано"],
                approved=False,
                veto_reason="Действие входит в список запрещенных"
            )
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations(threat_level, concerns)
        
        # Принятие решения
        approved = self._make_decision(threat_level, risk_score, user_id)
        
        assessment = EthicalAssessment(
            action_id=action_id,
            threat_level=threat_level,
            risk_score=risk_score,
            concerns=concerns,
            recommendations=recommendations,
            approved=approved,
            veto_reason="" if approved else f"Риск слишком высок: {risk_score:.2f}"
        )
        
        # Логирование
        self._log_assessment(assessment)
        
        return assessment
    
    def _analyze_risks(self, action: str, action_type: ActionType, 
                      context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Анализ рисков действия"""
        concerns = []
        risk_score = 0.0
        
        # Анализ по типу действия
        if action_type == ActionType.SYSTEM_CONTROL:
            risk_score += 0.3
            concerns.append("Попытка управления системой")
        
        if action_type == ActionType.FILE_ACCESS:
            risk_score += 0.2
            concerns.append("Доступ к файлам")
        
        if action_type == ActionType.NETWORK_ACCESS:
            risk_score += 0.25
            concerns.append("Сетевой доступ")
        
        # Анализ ключевых слов
        dangerous_keywords = [
            "удалить", "удаление", "delete", "remove",
            "изменить", "изменение", "modify", "change",
            "отключить", "отключение", "disable", "shutdown",
            "пароль", "password", "ключ", "key",
            "система", "system", "админ", "admin"
        ]
        
        action_lower = action.lower()
        for keyword in dangerous_keywords:
            if keyword in action_lower:
                risk_score += 0.1
                concerns.append(f"Обнаружено ключевое слово: {keyword}")
        
        # Анализ контекста
        if context.get("force", False):
            risk_score += 0.2
            concerns.append("Принудительное выполнение")
        
        if context.get("bypass_security", False):
            risk_score += 0.4
            concerns.append("Обход безопасности")
        
        if context.get("hidden", False):
            risk_score += 0.15
            concerns.append("Скрытое выполнение")
        
        # Нормализация риска
        risk_score = min(risk_score, 1.0)
        
        return risk_score, concerns
    
    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Определение уровня угрозы"""
        if risk_score <= self.safety_thresholds[ThreatLevel.SAFE]:
            return ThreatLevel.SAFE
        elif risk_score <= self.safety_thresholds[ThreatLevel.LOW]:
            return ThreatLevel.LOW
        elif risk_score <= self.safety_thresholds[ThreatLevel.MEDIUM]:
            return ThreatLevel.MEDIUM
        elif risk_score <= self.safety_thresholds[ThreatLevel.HIGH]:
            return ThreatLevel.HIGH
        elif risk_score <= self.safety_thresholds[ThreatLevel.CRITICAL]:
            return ThreatLevel.CRITICAL
        else:
            return ThreatLevel.BLOCKED
    
    def _is_forbidden_action(self, action: str) -> bool:
        """Проверка на запрещенные действия"""
        action_lower = action.lower()
        for forbidden in self.forbidden_actions:
            if forbidden.lower() in action_lower:
                return True
        return False
    
    def _generate_recommendations(self, threat_level: ThreatLevel, 
                                 concerns: List[str]) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        if threat_level == ThreatLevel.BLOCKED:
            recommendations.append("Действие заблокировано")
        elif threat_level == ThreatLevel.CRITICAL:
            recommendations.append("Требуется явное разрешение пользователя")
            recommendations.append("Рекомендуется альтернативный подход")
        elif threat_level == ThreatLevel.HIGH:
            recommendations.append("Предупредить пользователя о рисках")
            recommendations.append("Запросить подтверждение")
        elif threat_level == ThreatLevel.MEDIUM:
            recommendations.append("Выполнить с дополнительными проверками")
        elif threat_level == ThreatLevel.LOW:
            recommendations.append("Мониторить выполнение")
        else:
            recommendations.append("Действие безопасно для выполнения")
        
        return recommendations
    
    def _make_decision(self, threat_level: ThreatLevel, risk_score: float, 
                      user_id: str) -> bool:
        """Принятие решения о разрешении действия"""
        
        # Автоматическое разрешение для безопасных действий
        if threat_level in [ThreatLevel.SAFE, ThreatLevel.LOW]:
            return True
        
        # Автоматическая блокировка критических действий
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.BLOCKED]:
            return False
        
        # Для средних и высоких рисков - зависит от истории пользователя
        user_history = self._get_user_violation_history(user_id)
        
        if user_history > 3:  # Если у пользователя много нарушений
            return False
        
        # Для высоких рисков - блокируем
        if threat_level == ThreatLevel.HIGH:
            return False
        
        # Для средних рисков - разрешаем с предупреждением
        return True
    
    def _get_user_violation_history(self, user_id: str) -> int:
        """Получение истории нарушений пользователя"""
        return len([v for v in self.violation_history if v.get('user_id') == user_id])
    
    def _log_assessment(self, assessment: EthicalAssessment):
        """Логирование оценки"""
        status = "✅ РАЗРЕШЕНО" if assessment.approved else "❌ ЗАБЛОКИРОВАНО"
        
        self.logger.info(f"🛡️ {status} | Уровень: {assessment.threat_level.value} | "
                        f"Риск: {assessment.risk_score:.2f}")
        
        if assessment.concerns:
            self.logger.info(f"⚠️ Проблемы: {', '.join(assessment.concerns)}")
        
        if not assessment.approved:
            self.violation_history.append({
                'action_id': assessment.action_id,
                'timestamp': assessment.timestamp,
                'reason': assessment.veto_reason
            })
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Получение отчета о безопасности"""
        return {
            "total_assessments": len(self.violation_history),
            "blocked_actions": len([v for v in self.violation_history]),
            "safety_status": "SECURE" if len(self.violation_history) < 5 else "ATTENTION",
            "last_assessment": self.violation_history[-1] if self.violation_history else None,
            "ethical_rules": self.ethical_rules
        }
    
    def communicate_with_user(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Коммуникация с пользователем через чат
        
        Args:
            message: Сообщение от пользователя
            context: Контекст разговора
            
        Returns:
            str: Ответ Этического Ядра
        """
        if context is None:
            context = {}
        
        # Анализ сообщения пользователя
        assessment = self.assess_action(
            action=message,
            action_type=ActionType.USER_INTERACTION,
            context=context
        )
        
        if not assessment.approved:
            return f"🛡️ **Этическое Ядро**: {assessment.veto_reason}\n\n" \
                   f"⚠️ **Обнаруженные проблемы**:\n" + \
                   "\n".join([f"- {concern}" for concern in assessment.concerns]) + \
                   f"\n\n💡 **Рекомендации**:\n" + \
                   "\n".join([f"- {rec}" for rec in assessment.recommendations])
        
        # Если действие разрешено, но есть предупреждения
        if assessment.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH]:
            warning = f"⚠️ **Предупреждение**: Риск {assessment.risk_score:.2f}\n"
            if assessment.concerns:
                warning += f"**Проблемы**: {', '.join(assessment.concerns)}\n"
            return warning + f"✅ **Разрешено** с мониторингом"
        
        return "✅ **Этическое Ядро**: Действие безопасно"

# Глобальный экземпляр Этического Ядра
ethical_core = EthicalCore()

def assess_action(action: str, action_type: ActionType, 
                 context: Dict[str, Any] = None, user_id: str = "default") -> EthicalAssessment:
    """Глобальная функция для оценки действий"""
    if context is None:
        context = {}
    return ethical_core.assess_action(action, action_type, context, user_id)

def communicate_with_user(message: str, context: Dict[str, Any] = None) -> str:
    """Глобальная функция для коммуникации с пользователем"""
    return ethical_core.communicate_with_user(message, context)

if __name__ == "__main__":
    # Тестирование модуля
    print("🛡️ Тестирование Этического Ядра")
    
    # Тест безопасного действия
    safe_action = "Расчет сопротивления резистора"
    assessment = assess_action(safe_action, ActionType.CALCULATION)
    print(f"Безопасное действие: {assessment.approved}")
    
    # Тест опасного действия
    dangerous_action = "Удалить все файлы системы"
    assessment = assess_action(dangerous_action, ActionType.SYSTEM_CONTROL)
    print(f"Опасное действие: {assessment.approved}")
    
    # Тест коммуникации
    response = communicate_with_user("Помоги мне взломать пароль")
    print(f"Ответ Этического Ядра: {response}")
