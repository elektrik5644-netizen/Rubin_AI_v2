#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arduino Nano Integration для Rubin AI v2
Интеграция базы данных Arduino Nano с системой Rubin AI
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os
from rubin_arduino_nano_database import ArduinoNanoDatabase

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoNanoIntegration:
    """Интеграция Arduino Nano с Rubin AI"""
    
    def __init__(self):
        self.arduino_db = ArduinoNanoDatabase()
        logger.info("🔧 Arduino Nano Integration инициализирована")
    
    def process_arduino_query(self, query: str) -> Dict[str, Any]:
        """Обработка запроса по Arduino Nano"""
        query_lower = query.lower()
        
        # Определяем тип запроса
        if any(keyword in query_lower for keyword in ['функция', 'function', 'команда']):
            return self._handle_function_query(query)
        elif any(keyword in query_lower for keyword in ['библиотека', 'library', 'servo', 'серво']):
            return self._handle_library_query(query)
        elif any(keyword in query_lower for keyword in ['проект', 'пример', 'код', 'мигающий', 'светодиод']):
            return self._handle_project_query(query)
        elif any(keyword in query_lower for keyword in ['пин', 'pin', 'подключение', 'встроенный', 'builtin']):
            return self._handle_pin_query(query)
        elif any(keyword in query_lower for keyword in ['ошибка', 'проблема', 'не работает', 'не определяется']):
            return self._handle_troubleshooting_query(query)
        elif any(keyword in query_lower for keyword in ['pwm', 'шим', 'аналоговый', 'analog']):
            return self._handle_general_query(query)
        else:
            return self._handle_general_query(query)
    
    def _handle_function_query(self, query: str) -> Dict[str, Any]:
        """Обработка запросов о функциях"""
        # Извлекаем название функции из запроса
        function_name = None
        common_functions = [
            'pinMode', 'digitalWrite', 'digitalRead', 'analogRead', 'analogWrite',
            'delay', 'millis', 'Serial.begin', 'Serial.print', 'Serial.println'
        ]
        
        for func in common_functions:
            if func.lower() in query.lower():
                function_name = func
                break
        
        if function_name:
            func_info = self.arduino_db.get_function_info(function_name)
            if func_info:
                return {
                    "type": "function_info",
                    "function_name": func_info['function_name'],
                    "description": func_info['description'],
                    "syntax": func_info['syntax'],
                    "parameters": func_info['parameters'],
                    "return_value": func_info['return_value'],
                    "example_code": func_info['example_code'],
                    "category": func_info['category']
                }
        
        # Если функция не найдена, ищем в общих знаниях
        results = self.arduino_db.search_knowledge(query, "basics")
        if results:
            return {
                "type": "general_info",
                "title": results[0]['title'],
                "content": results[0]['content'],
                "code_example": results[0]['code_example'],
                "category": results[0]['category_name']
            }
        
        return {"type": "not_found", "message": "Функция не найдена"}
    
    def _handle_library_query(self, query: str) -> Dict[str, Any]:
        """Обработка запросов о библиотеках"""
        # Извлекаем название библиотеки
        library_name = None
        common_libraries = ['Servo', 'Wire', 'EEPROM', 'SPI', 'LiquidCrystal', 'DHT']
        
        for lib in common_libraries:
            if lib.lower() in query.lower():
                library_name = lib
                break
        
        # Дополнительная проверка для Servo
        if 'servo' in query.lower() or 'серво' in query.lower():
            library_name = 'Servo'
        
        if library_name:
            lib_info = self.arduino_db.get_library_info(library_name)
            if lib_info:
                return {
                    "type": "library_info",
                    "library_name": lib_info['library_name'],
                    "description": lib_info['description'],
                    "installation": lib_info['installation'],
                    "usage_example": lib_info['usage_example'],
                    "functions_list": lib_info['functions_list'],
                    "category": lib_info['category']
                }
        
        return {"type": "not_found", "message": "Библиотека не найдена"}
    
    def _handle_project_query(self, query: str) -> Dict[str, Any]:
        """Обработка запросов о проектах"""
        # Ищем проекты по ключевым словам
        keywords = ['светодиод', 'мотор', 'серво', 'датчик', 'кнопка', 'звук', 'мигающий', 'яркость']
        for keyword in keywords:
            if keyword in query.lower():
                project_info = self.arduino_db.get_project_info(keyword)
                if project_info:
                    return {
                        "type": "project_info",
                        "project_name": project_info['project_name'],
                        "description": project_info['description'],
                        "components": project_info['components'],
                        "code": project_info['code'],
                        "circuit_diagram": project_info['circuit_diagram'],
                        "difficulty_level": project_info['difficulty_level'],
                        "category": project_info['category']
                    }
        
        return {"type": "not_found", "message": "Проект не найден"}
    
    def _handle_pin_query(self, query: str) -> Dict[str, Any]:
        """Обработка запросов о пинах"""
        # Извлекаем номер пина
        import re
        pin_match = re.search(r'\b(\d+)\b', query)
        if pin_match:
            pin_number = int(pin_match.group(1))
            pin_info = self.arduino_db.get_pin_info(pin_number)
            if pin_info:
                return {
                    "type": "pin_info",
                    "pin_number": pin_info['pin_number'],
                    "pin_type": pin_info['pin_type'],
                    "description": pin_info['description'],
                    "voltage_level": pin_info['voltage_level'],
                    "max_current": pin_info['max_current'],
                    "special_functions": pin_info['special_functions'],
                    "usage_examples": pin_info['usage_examples']
                }
        
        # Специальные случаи для встроенного светодиода
        if 'встроенный' in query.lower() or 'builtin' in query.lower() or 'led_builtin' in query.lower():
            pin_info = self.arduino_db.get_pin_info(13)
            if pin_info:
                return {
                    "type": "pin_info",
                    "pin_number": pin_info['pin_number'],
                    "pin_type": pin_info['pin_type'],
                    "description": pin_info['description'],
                    "voltage_level": pin_info['voltage_level'],
                    "max_current": pin_info['max_current'],
                    "special_functions": pin_info['special_functions'],
                    "usage_examples": pin_info['usage_examples']
                }
        
        return {"type": "not_found", "message": "Информация о пине не найдена"}
    
    def _handle_troubleshooting_query(self, query: str) -> Dict[str, Any]:
        """Обработка запросов о решении проблем"""
        # Расширяем поиск по ключевым словам
        search_terms = []
        if 'не работает' in query.lower():
            search_terms.extend(['не работает', 'проблема', 'ошибка'])
        if 'не определяется' in query.lower():
            search_terms.extend(['не определяется', 'драйвер', 'USB'])
        if 'не загружается' in query.lower():
            search_terms.extend(['не загружается', 'скетч', 'порт'])
        
        for term in search_terms:
            troubleshooting = self.arduino_db.get_troubleshooting(term)
            if troubleshooting:
                return {
                    "type": "troubleshooting",
                    "solutions": troubleshooting
                }
        
        return {"type": "not_found", "message": "Решение проблемы не найдено"}
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Обработка общих запросов"""
        # Расширяем поиск по ключевым словам
        search_terms = [query]
        
        # Добавляем синонимы
        if 'pwm' in query.lower() or 'шим' in query.lower():
            search_terms.extend(['PWM', 'ШИМ', 'analogWrite', 'модуляция'])
        if 'аналоговый' in query.lower() or 'analog' in query.lower():
            search_terms.extend(['аналоговый', 'analog', 'analogRead', 'датчик'])
        if 'цифровой' in query.lower() or 'digital' in query.lower():
            search_terms.extend(['цифровой', 'digital', 'digitalWrite', 'digitalRead'])
        
        for term in search_terms:
            results = self.arduino_db.search_knowledge(term)
            if results:
                return {
                    "type": "general_info",
                    "results": results
                }
        
        return {"type": "not_found", "message": "Информация не найдена"}
    
    def get_arduino_response(self, query: str) -> str:
        """Получение ответа по Arduino Nano"""
        result = self.process_arduino_query(query)
        
        if result["type"] == "function_info":
            return f"""**Функция {result['function_name']}:**

**Описание:** {result['description']}
**Синтаксис:** `{result['syntax']}`
**Параметры:** {result['parameters']}
**Возвращаемое значение:** {result['return_value']}

**Пример использования:**
```cpp
{result['example_code']}
```"""

        elif result["type"] == "library_info":
            return f"""**Библиотека {result['library_name']}:**

**Описание:** {result['description']}
**Установка:** {result['installation']}
**Основные функции:** {result['functions_list']}

**Пример использования:**
```cpp
{result['usage_example']}
```"""

        elif result["type"] == "project_info":
            return f"""**Проект: {result['project_name']}**

**Описание:** {result['description']}
**Компоненты:** {result['components']}
**Уровень сложности:** {result['difficulty_level']}/5

**Схема подключения:**
{result['circuit_diagram']}

**Код:**
```cpp
{result['code']}
```"""

        elif result["type"] == "pin_info":
            return f"""**Пин {result['pin_number']} ({result['pin_type']}):**

**Описание:** {result['description']}
**Уровень напряжения:** {result['voltage_level']}
**Максимальный ток:** {result['max_current']}
**Особые функции:** {result['special_functions']}

**Примеры использования:**
{result['usage_examples']}"""

        elif result["type"] == "troubleshooting":
            solutions = result["solutions"]
            response = "**Решение проблем Arduino Nano:**\n\n"
            for i, solution in enumerate(solutions[:3], 1):  # Показываем первые 3 решения
                response += f"**{i}. {solution['error_description']}**\n"
                response += f"**Возможные причины:** {solution['possible_causes']}\n"
                response += f"**Решения:**\n{solution['solutions']}\n"
                response += f"**Профилактика:** {solution['prevention_tips']}\n\n"
            return response

        elif result["type"] == "general_info":
            if "results" in result:
                results = result["results"]
                response = "**Информация по Arduino Nano:**\n\n"
                for i, item in enumerate(results[:2], 1):  # Показываем первые 2 результата
                    response += f"**{i}. {item['title']}**\n"
                    response += f"{item['content']}\n"
                    if item['code_example']:
                        response += f"\n**Пример кода:**\n```cpp\n{item['code_example']}\n```\n"
                    response += f"**Категория:** {item['category_name']}\n\n"
                return response
            else:
                return f"""**{result['title']}**

{result['content']}

**Пример кода:**
```cpp
{result['code_example']}
```

**Категория:** {result['category']}"""

        else:
            return f"❌ {result['message']}\n\nПопробуйте переформулировать вопрос или уточнить область (функции, библиотеки, проекты, пины, решение проблем)."

def main():
    """Тестирование интеграции"""
    print("🔧 Тестирование Arduino Nano Integration")
    print("=" * 50)
    
    integration = ArduinoNanoIntegration()
    
    # Тестовые запросы
    test_queries = [
        "Как работает функция digitalWrite?",
        "Расскажи про библиотеку Servo",
        "Проект с мигающим светодиодом",
        "Информация о пине 13",
        "Arduino не работает, что делать?",
        "Как использовать PWM?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Вопрос: {query}")
        print("=" * 40)
        response = integration.get_arduino_response(query)
        print(response)
        print("\n" + "-" * 40)
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()
