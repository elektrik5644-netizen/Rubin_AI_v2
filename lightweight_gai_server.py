#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Lightweight GAI Server - Оптимизированная версия без тяжелых моделей
Минимальное потребление памяти для работы на слабых системах
"""

from flask import Flask, request, jsonify
import logging
import json
import random
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class LightweightGAI:
    """Легковесный генеративный ИИ без внешних зависимостей"""
    
    def __init__(self):
        self.templates = {
            'text': [
                "Основываясь на вашем запросе, я могу предложить следующее решение:",
                "Рассматривая данную проблему с точки зрения ИИ, можно выделить:",
                "Для эффективного решения этой задачи рекомендую:",
                "Анализируя предоставленную информацию, вижу следующие возможности:"
            ],
            'code': [
                "# Оптимизированное решение для вашей задачи:",
                "# Эффективный алгоритм с минимальным потреблением памяти:",
                "# Простое и надежное решение:",
                "# Масштабируемый подход к решению:"
            ],
            'diagram': [
                "Диаграмма архитектуры системы:",
                "Схема взаимодействия компонентов:",
                "Визуализация процесса:",
                "Структурная диаграмма решения:"
            ]
        }
    
    def generate_text(self, prompt, max_length=5000):
        """Генерация текста без использования тяжелых моделей"""
        try:
            template = random.choice(self.templates['text'])
            
            # Простая логика генерации на основе ключевых слов
            keywords = prompt.lower().split()
            response = template
            
            if any(word in keywords for word in ['алгоритм', 'код', 'программирование']):
                response += "\n\n1. Определите основные требования к системе"
                response += "\n2. Выберите подходящий алгоритм"
                response += "\n3. Реализуйте с учетом производительности"
            elif any(word in keywords for word in ['анализ', 'данные', 'статистика']):
                response += "\n\n1. Соберите релевантные данные"
                response += "\n2. Проведите предварительный анализ"
                response += "\n3. Примените статистические методы"
            else:
                response += "\n\n1. Изучите проблему детально"
                response += "\n2. Найдите аналогичные решения"
                response += "\n3. Адаптируйте под ваши условия"
            
            return response[:max_length]
        except Exception as e:
            return f"Ошибка генерации текста: {str(e)}"
    
    def generate_code(self, language, task, max_lines=50):
        """Генерация кода без использования тяжелых моделей"""
        try:
            template = random.choice(self.templates['code'])
            
            # Простые шаблоны кода
            code_templates = {
                'python': f"""# {task}
def solve_problem():
    # Инициализация
    result = []
    
    # Основная логика
    for i in range(10):
        result.append(i * 2)
    
    return result

# Использование
if __name__ == "__main__":
    solution = solve_problem()
    print(solution)""",
                
                'javascript': f"""// {task}
function solveProblem() {{
    // Инициализация
    const result = [];
    
    // Основная логика
    for (let i = 0; i < 10; i++) {{
        result.push(i * 2);
    }}
    
    return result;
}}

// Использование
const solution = solveProblem();
console.log(solution);""",
                
                'java': f"""// {task}
public class Solution {{
    public static int[] solveProblem() {{
        // Инициализация
        int[] result = new int[10];
        
        // Основная логика
        for (int i = 0; i < 10; i++) {{
            result[i] = i * 2;
        }}
        
        return result;
    }}
    
    public static void main(String[] args) {{
        int[] solution = solveProblem();
        System.out.println(Arrays.toString(solution));
    }}
}}"""
            }
            
            return code_templates.get(language.lower(), code_templates['python'])[:max_lines*50]
        except Exception as e:
            return f"# Ошибка генерации кода: {str(e)}"
    
    def generate_diagram(self, description):
        """Генерация описания диаграммы"""
        try:
            template = random.choice(self.templates['diagram'])
            
            # Простые шаблоны диаграмм
            diagram_templates = [
                f"{template}\n\n```\n┌─────────────┐    ┌─────────────┐\n│   Input     │───▶│  Process    │\n└─────────────┘    └─────────────┘\n                           │\n                           ▼\n                   ┌─────────────┐\n                   │   Output    │\n                   └─────────────┘\n```",
                
                f"{template}\n\n```\n┌─────────────┐\n│   Start     │\n└──────┬──────┘\n       │\n       ▼\n┌─────────────┐\n│   Process   │\n└──────┬──────┘\n       │\n       ▼\n┌─────────────┐\n│    End      │\n└─────────────┘\n```"
            ]
            
            return random.choice(diagram_templates)
        except Exception as e:
            return f"Ошибка генерации диаграммы: {str(e)}"

# Инициализация GAI
gai = LightweightGAI()

@app.route('/api/gai/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "status": "healthy",
        "service": "Lightweight GAI Server",
        "memory_usage": "optimized",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/gai/status', methods=['GET'])
def status():
    """Статус сервера"""
    return jsonify({
        "service": "Lightweight GAI Server",
        "version": "1.0.0",
        "memory_optimized": True,
        "models_loaded": False,
        "capabilities": [
            "text_generation",
            "code_generation", 
            "diagram_description"
        ]
    })

@app.route('/api/gai/generate_text', methods=['POST'])
def generate_text():
    """Генерация текста"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 5000)
        
        if not prompt:
            return jsonify({"error": "Промпт не может быть пустым"}), 400
        
        result = gai.generate_text(prompt, max_length)
        
        return jsonify({
            "generated_text": result,
            "prompt": prompt,
            "length": len(result),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации текста: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/gai/generate_code', methods=['POST'])
def generate_code():
    """Генерация кода"""
    try:
        data = request.get_json()
        language = data.get('language', 'python')
        task = data.get('task', '')
        max_lines = data.get('max_lines', 50)
        
        if not task:
            return jsonify({"error": "Задача не может быть пустой"}), 400
        
        result = gai.generate_code(language, task, max_lines)
        
        return jsonify({
            "generated_code": result,
            "language": language,
            "task": task,
            "lines": len(result.split('\n')),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации кода: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/gai/generate_diagram', methods=['POST'])
def generate_diagram():
    """Генерация диаграммы"""
    try:
        data = request.get_json()
        description = data.get('description', '')
        
        if not description:
            return jsonify({"error": "Описание не может быть пустым"}), 400
        
        result = gai.generate_diagram(description)
        
        return jsonify({
            "generated_diagram": result,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации диаграммы: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Lightweight GAI Server запущен")
    logger.info("URL: http://localhost:8104")
    logger.info("💾 Оптимизирован для минимального потребления памяти")
    logger.info("Endpoints:")
    logger.info("  - POST /api/gai/generate_text - Генерация текста")
    logger.info("  - POST /api/gai/generate_code - Генерация кода")
    logger.info("  - POST /api/gai/generate_diagram - Генерация диаграмм")
    logger.info("  - GET /api/gai/health - Проверка здоровья")
    logger.info("  - GET /api/gai/status - Статус сервера")
    
    app.run(host='0.0.0.0', port=8104, debug=False, threaded=True)

