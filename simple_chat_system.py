#!/usr/bin/env python3
"""
Простая система чата для Rubin AI без сложных зависимостей
"""

import json
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class SimpleRubinAI:
    """Простая система AI без сложных зависимостей"""
    
    def __init__(self):
        self.knowledge_base = self.load_knowledge_base()
        self.conversation_history = []
    
    def load_knowledge_base(self):
        """Загрузка базы знаний"""
        knowledge = {
            # Программирование
            'python': {
                'keywords': ['python', 'питон', 'программирование', 'код'],
                'responses': [
                    "Python - отличный язык программирования! Могу помочь с:\n• Синтаксисом и структурами данных\n• Библиотеками (numpy, pandas, matplotlib)\n• ООП и функциональным программированием\n• Отладкой и оптимизацией кода",
                    "Python широко используется в:\n• Веб-разработке (Django, Flask)\n• Data Science и машинном обучении\n• Автоматизации и скриптинге\n• Научных вычислениях"
                ]
            },
            
            # Электротехника
            'electronics': {
                'keywords': ['транзистор', 'диод', 'резистор', 'конденсатор', 'схема', 'электротехника'],
                'responses': [
                    "Электротехника - основа современной техники!\n• Транзисторы - ключевые элементы усилителей и переключателей\n• Диоды - выпрямляют переменный ток\n• Резисторы - ограничивают ток\n• Конденсаторы - накапливают энергию",
                    "Основные законы:\n• Закон Ома: U = I × R\n• Законы Кирхгофа\n• Мощность: P = U × I\n• Энергия: W = P × t"
                ]
            },
            
            # Промышленная автоматизация
            'automation': {
                'keywords': ['plc', 'scada', 'автоматизация', 'контроллер', 'pmac', 'чпу'],
                'responses': [
                    "Промышленная автоматизация включает:\n• PLC (программируемые логические контроллеры)\n• SCADA системы мониторинга\n• PMAC контроллеры для точного позиционирования\n• ЧПУ системы для станков",
                    "Основные протоколы:\n• Modbus RTU/TCP\n• Profinet\n• Ethernet/IP\n• OPC UA"
                ]
            },
            
            # Математика
            'mathematics': {
                'keywords': ['математика', 'вычислить', 'решить', 'формула', 'уравнение', '+', '-', '*', '/'],
                'responses': [
                    "Математика - язык науки!\n• Арифметика: сложение, вычитание, умножение, деление\n• Алгебра: уравнения, функции, графики\n• Геометрия: площади, объемы, углы\n• Тригонометрия: синусы, косинусы, тангенсы"
                ]
            },
            
            # Общие вопросы
            'general': {
                'keywords': ['привет', 'как дела', 'что умеешь', 'помощь', 'help'],
                'responses': [
                    "Привет! Я Rubin AI - ваш помощник по техническим вопросам.\n\nЯ специализируюсь на:\n• Программировании (Python, C++, PLC)\n• Электротехнике и схемотехнике\n• Промышленной автоматизации\n• Математике и физике\n\nЧем могу помочь?",
                    "Мои возможности:\n• Анализ и объяснение кода\n• Решение технических задач\n• Объяснение электронных схем\n• Помощь с автоматизацией\n• Математические вычисления"
                ]
            }
        }
        return knowledge
    
    def classify_message(self, message):
        """Простая классификация сообщения"""
        message_lower = message.lower()
        
        # Проверяем каждую категорию
        for category, data in self.knowledge_base.items():
            for keyword in data['keywords']:
                if keyword in message_lower:
                    return category
        
        return 'general'
    
    def solve_math(self, message):
        """Простое решение математических задач"""
        # Ищем арифметические выражения
        pattern = r'(\d+)\s*([+\-*/])\s*(\d+)'
        match = re.search(pattern, message)
        
        if match:
            num1 = int(match.group(1))
            op = match.group(2)
            num2 = int(match.group(3))
            
            if op == '+':
                result = num1 + num2
            elif op == '-':
                result = num1 - num2
            elif op == '*':
                result = num1 * num2
            elif op == '/':
                result = num1 / num2 if num2 != 0 else "Деление на ноль!"
            
            return f"🧮 **Решение:** {num1} {op} {num2} = {result}"
        
        return None
    
    def generate_response(self, message):
        """Генерация ответа на сообщение"""
        # Сохраняем в историю
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'category': None
        })
        
        # Проверяем на математические задачи
        math_result = self.solve_math(message)
        if math_result:
            return {
                'response': math_result,
                'category': 'mathematics',
                'confidence': 0.9,
                'source': 'math_solver'
            }
        
        # Классифицируем сообщение
        category = self.classify_message(message)
        
        # Выбираем ответ
        if category in self.knowledge_base:
            responses = self.knowledge_base[category]['responses']
            import random
            response = random.choice(responses)
        else:
            response = "Извините, я не понял ваш вопрос. Попробуйте переформулировать или спросите о программировании, электротехнике или автоматизации."
        
        # Обновляем историю
        self.conversation_history[-1]['category'] = category
        
        return {
            'response': response,
            'category': category,
            'confidence': 0.8,
            'source': 'knowledge_base'
        }

# Создаем экземпляр AI
rubin_ai = SimpleRubinAI()

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Основной endpoint для чата"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Генерируем ответ
        response = rubin_ai.generate_response(message)
        response['timestamp'] = datetime.now().isoformat()
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Ошибка обработки: {str(e)}',
            'response': 'Извините, произошла ошибка. Попробуйте еще раз.',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health')
def health():
    """Проверка здоровья системы"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': 'Simple Rubin AI',
        'conversations': len(rubin_ai.conversation_history)
    })

@app.route('/api/stats')
def stats():
    """Статистика системы"""
    return jsonify({
        'conversations': len(rubin_ai.conversation_history),
        'knowledge_categories': len(rubin_ai.knowledge_base),
        'system': 'Simple Rubin AI',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Запуск Simple Rubin AI...")
    print("📡 API доступен на: http://localhost:8085")
    print("🔗 Тестирование: http://localhost:8085/api/health")
    app.run(host='0.0.0.0', port=8085, debug=True)












