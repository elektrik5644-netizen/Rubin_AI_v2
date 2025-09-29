#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль управления директивами для Rubin AI
Позволяет устанавливать, проверять и выполнять директивы через чат/Telegram
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

logger = logging.getLogger(__name__)

class DirectivesManager:
    """Менеджер директив для Rubin AI"""
    
    def __init__(self, directives_file: str = "rubin_directives.json"):
        self.directives_file = directives_file
        self.directives = self._load_directives()
        
    def _load_directives(self) -> Dict[str, Any]:
        """Загружает директивы из файла"""
        try:
            if os.path.exists(self.directives_file):
                with open(self.directives_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {
                    "active_directives": [],
                    "directive_history": [],
                    "last_updated": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Ошибка загрузки директив: {e}")
            return {"active_directives": [], "directive_history": []}
    
    def _save_directives(self):
        """Сохраняет директивы в файл"""
        try:
            self.directives["last_updated"] = datetime.now().isoformat()
            with open(self.directives_file, 'w', encoding='utf-8') as f:
                json.dump(self.directives, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения директив: {e}")
    
    def add_directive(self, directive_text: str, user_id: str = "default") -> Dict[str, Any]:
        """Добавляет новую директиву"""
        directive_id = f"dir_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        directive = {
            "id": directive_id,
            "text": directive_text,
            "created_at": datetime.now().isoformat(),
            "created_by": user_id,
            "active": True,
            "execution_count": 0,
            "last_executed": None
        }
        
        self.directives["active_directives"].append(directive)
        self.directives["directive_history"].append({
            "action": "added",
            "directive": directive,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_directives()
        
        return {
            "success": True,
            "message": f"Директива добавлена: {directive_text}",
            "directive_id": directive_id
        }
    
    def remove_directive(self, directive_id: str) -> Dict[str, Any]:
        """Удаляет директиву по ID"""
        for i, directive in enumerate(self.directives["active_directives"]):
            if directive["id"] == directive_id:
                removed = self.directives["active_directives"].pop(i)
                self.directives["directive_history"].append({
                    "action": "removed",
                    "directive": removed,
                    "timestamp": datetime.now().isoformat()
                })
                self._save_directives()
                return {
                    "success": True,
                    "message": f"Директива удалена: {removed['text']}"
                }
        
        return {
            "success": False,
            "message": f"Директива с ID {directive_id} не найдена"
        }
    
    def list_directives(self) -> Dict[str, Any]:
        """Возвращает список активных директив"""
        return {
            "success": True,
            "active_directives": self.directives["active_directives"],
            "count": len(self.directives["active_directives"])
        }
    
    def check_directives(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Проверяет директивы в контексте ответа/анализа"""
        applicable_directives = []
        
        for directive in self.directives["active_directives"]:
            if not directive["active"]:
                continue
                
            # Простая проверка по ключевым словам в директиве
            directive_text = directive["text"].lower()
            context_text = str(context).lower()
            
            # Проверяем, применима ли директива к текущему контексту
            if self._is_directive_applicable(directive_text, context_text, context):
                directive["execution_count"] += 1
                directive["last_executed"] = datetime.now().isoformat()
                applicable_directives.append(directive)
        
        if applicable_directives:
            self._save_directives()
        
        return applicable_directives
    
    def _is_directive_applicable(self, directive_text: str, context_text: str, context: Dict[str, Any]) -> bool:
        """Определяет, применима ли директива к контексту"""
        # Простые правила применения директив
        
        # Директива "всегда" - применяется ко всем ответам
        if "всегда" in directive_text or "каждый раз" in directive_text:
            return True
        
        # Директива по категории модуля
        module_category = context.get("category", "").lower()
        if module_category and module_category in directive_text:
            return True
        
        # Директива по типу запроса
        if "анализ" in directive_text and "analysis" in context_text:
            return True
        
        if "график" in directive_text and ("graph" in context_text or "график" in context_text):
            return True
        
        if "расчет" in directive_text and ("calculate" in context_text or "расчет" in context_text):
            return True
        
        # Директива по ключевым словам в контексте
        directive_keywords = directive_text.split()
        context_keywords = context_text.split()
        
        # Если хотя бы 2 ключевых слова из директивы есть в контексте
        matches = sum(1 for kw in directive_keywords if kw in context_keywords)
        if matches >= 2:
            return True
        
        return False
    
    def execute_directive(self, directive: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Выполняет директиву и возвращает результат"""
        directive_text = directive["text"]
        
        # Простые команды директив
        if "добавь" in directive_text.lower():
            return f"✅ Выполнено: {directive_text}"
        
        if "проверь" in directive_text.lower():
            return f"🔍 Проверка: {directive_text}"
        
        if "уточни" in directive_text.lower():
            return f"❓ Уточнение: {directive_text}"
        
        if "объясни подробнее" in directive_text.lower():
            return f"📖 Подробное объяснение по запросу"
        
        if "покажи пример" in directive_text.lower():
            return f"💡 Пример использования"
        
        if "предупреждение" in directive_text.lower():
            return f"⚠️ ВНИМАНИЕ: {directive_text}"
        
        # По умолчанию возвращаем текст директивы
        return f"📋 Директива: {directive_text}"
    
    def get_directive_stats(self) -> Dict[str, Any]:
        """Возвращает статистику по директивам"""
        total_directives = len(self.directives["active_directives"])
        total_executions = sum(d["execution_count"] for d in self.directives["active_directives"])
        
        most_used = max(self.directives["active_directives"], 
                       key=lambda x: x["execution_count"], 
                       default={"text": "Нет", "execution_count": 0})
        
        return {
            "total_directives": total_directives,
            "total_executions": total_executions,
            "most_used_directive": most_used["text"],
            "most_used_count": most_used["execution_count"],
            "last_updated": self.directives.get("last_updated", "Неизвестно")
        }

# Глобальный экземпляр менеджера директив
directives_manager = DirectivesManager()

def process_directives_command(command: str, user_id: str = "default") -> Dict[str, Any]:
    """Обрабатывает команды управления директивами"""
    command_lower = command.lower().strip()
    
    if command_lower.startswith("прими директиву"):
        # Извлекаем текст директивы после "прими директиву"
        directive_text = command[15:].strip()
        if directive_text:
            return directives_manager.add_directive(directive_text, user_id)
        else:
            return {
                "success": False,
                "message": "Укажите текст директивы после 'прими директиву'"
            }
    
    elif command_lower == "список директив":
        return directives_manager.list_directives()
    
    elif command_lower.startswith("удали директиву"):
        directive_id = command[15:].strip()
        return directives_manager.remove_directive(directive_id)
    
    elif command_lower == "статистика директив":
        return {
            "success": True,
            "stats": directives_manager.get_directive_stats()
        }
    
    elif command_lower == "помощь по директивам":
        return {
            "success": True,
            "help": {
                "commands": [
                    "прими директиву [текст] - добавить новую директиву",
                    "список директив - показать активные директивы",
                    "удали директиву [ID] - удалить директиву по ID",
                    "статистика директив - показать статистику",
                    "помощь по директивам - эта справка"
                ],
                "examples": [
                    "прими директиву всегда добавляй примеры к ответам",
                    "прими директиву при анализе графиков проверяй тренд",
                    "прими директиву в электротехнике объясняй формулы подробнее"
                ]
            }
        }
    
    else:
        return {
            "success": False,
            "message": "Неизвестная команда. Используйте 'помощь по директивам'"
        }

def check_and_apply_directives(context: Dict[str, Any]) -> List[str]:
    """Проверяет и применяет директивы к контексту"""
    applicable_directives = directives_manager.check_directives(context)
    
    results = []
    for directive in applicable_directives:
        result = directives_manager.execute_directive(directive, context)
        results.append(result)
    
    return results






