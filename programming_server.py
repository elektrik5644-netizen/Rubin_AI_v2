#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
💻 PROGRAMMING SERVER
====================
Сервер для программирования и разработки
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/programming/explain', methods=['GET', 'POST'])
def explain_programming():
    """Объяснение программистских концепций"""
    try:
        if request.method == 'GET':
            concept = request.args.get('concept', '')
        else:
            data = request.get_json()
            concept = data.get('concept', '')
        
        logger.info(f"💻 Получен запрос программирования: {concept[:50]}...")
        
        # Простая логика объяснения
        if "python" in concept.lower():
            result = "Python - высокоуровневый язык программирования с простым синтаксисом"
        elif "алгоритм" in concept.lower() or "сортировка" in concept.lower():
            result = """**Алгоритмы сортировки:**

**Основные типы:**
- **Пузырьковая сортировка** - O(n²), простая реализация
- **Быстрая сортировка** - O(n log n), эффективная
- **Сортировка слиянием** - O(n log n), стабильная
- **Сортировка вставками** - O(n²), хороша для малых массивов

**Принцип работы:**
1. Сравнение элементов
2. Перестановка при необходимости
3. Повтор до полной упорядоченности

**Пример (Python):**
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```"""
        elif "функция" in concept.lower():
            result = "Функция - блок кода, который можно вызывать многократно"
        elif "класс" in concept.lower():
            result = "Класс - шаблон для создания объектов в ООП"
        else:
            result = f"Объяснение концепции '{concept}' требует дополнительного анализа"
        
        return jsonify({
            "module": "programming",
            "concept": concept,
            "explanation": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в programming: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/programming/analyze', methods=['POST'])
def analyze_code():
    """Анализ кода"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        logger.info(f"💻 Анализ кода: {len(code)} символов")
        
        # Простой анализ
        lines = code.count('\n') + 1
        functions = code.count('def ')
        classes = code.count('class ')
        
        result = f"Анализ кода: {lines} строк, {functions} функций, {classes} классов"
        
        return jsonify({
            "module": "programming",
            "analysis": result,
            "metrics": {
                "lines": lines,
                "functions": functions,
                "classes": classes
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа кода: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "programming",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

if __name__ == "__main__":
    print("💻 Programming Server запущен")
    print("URL: http://localhost:8088")
    print("Доступные эндпоинты:")
    print("  - POST /api/programming/explain - объяснение концепций")
    print("  - POST /api/programming/analyze - анализ кода")
    print("  - GET /api/health - проверка здоровья")
    app.run(host='0.0.0.0', port=8088, debug=False)





