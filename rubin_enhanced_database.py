#!/usr/bin/env python3
"""
Улучшенная версия Rubin AI с полной базой знаний
"""

import sqlite3
import json
import re
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class RubinEnhancedDatabase:
    """Улучшенная версия Rubin AI с подключением к полной базе знаний"""
    
    def __init__(self):
        self.databases = {
            'main': 'rubin_ai_v2.db',
            'documents': 'rubin_ai_documents.db',
            'knowledge': 'rubin_knowledge_base.db',
            'learning': 'rubin_learning.db'
        }
        self.conversation_history = []
        self.load_test_documents()
        self.load_database_content()
    
    def load_test_documents(self):
        """Загрузка документов из test_documents"""
        self.test_documents = {}
        test_docs_dir = 'test_documents'
        
        if os.path.exists(test_docs_dir):
            for filename in os.listdir(test_docs_dir):
                if filename.endswith('.txt') or filename.endswith('.py'):
                    file_path = os.path.join(test_docs_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.test_documents[filename] = {
                                'content': content,
                                'size': len(content),
                                'category': self.detect_category(filename)
                            }
                    except Exception as e:
                        print(f"Ошибка загрузки {filename}: {e}")
    
    def load_database_content(self):
        """Загрузка содержимого из баз данных"""
        self.database_content = []
        
        # Загружаем из основной базы данных
        if os.path.exists('rubin_ai_v2.db'):
            try:
                conn = sqlite3.connect('rubin_ai_v2.db')
                cursor = conn.cursor()
                cursor.execute("SELECT filename, content FROM documents LIMIT 20")
                results = cursor.fetchall()
                for filename, content in results:
                    if content and len(content) > 50:  # Только значимый контент
                        self.database_content.append({
                            'filename': filename,
                            'content': content,
                            'category': self.detect_category(filename),
                            'source': 'database'
                        })
                conn.close()
            except Exception as e:
                print(f"Ошибка загрузки из базы данных: {e}")
    
    def detect_category(self, filename):
        """Определение категории по имени файла"""
        filename_lower = filename.lower()
        if 'python' in filename_lower or 'programming' in filename_lower:
            return 'programming'
        elif 'electrical' in filename_lower or 'circuit' in filename_lower:
            return 'electronics'
        elif 'controller' in filename_lower or 'automation' in filename_lower or 'pid' in filename_lower:
            return 'automation'
        elif 'radio' in filename_lower:
            return 'radiomechanics'
        elif 'math' in filename_lower or 'mathematical' in filename_lower:
            return 'mathematics'
        else:
            return 'general'
    
    def search_content(self, query):
        """Поиск в содержимом документов"""
        results = []
        query_lower = query.lower()
        
        # Поиск в тестовых документах
        for filename, doc_data in self.test_documents.items():
            content_lower = doc_data['content'].lower()
            if query_lower in content_lower:
                # Находим релевантные фрагменты
                lines = doc_data['content'].split('\n')
                relevant_lines = []
                for i, line in enumerate(lines):
                    if query_lower in line.lower():
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        relevant_lines.extend(lines[start:end])
                
                if relevant_lines:
                    results.append({
                        'filename': filename,
                        'content': '\n'.join(relevant_lines),
                        'category': doc_data['category'],
                        'source': 'test_documents',
                        'relevance': self.calculate_relevance(query_lower, content_lower)
                    })
        
        # Поиск в содержимом базы данных
        for item in self.database_content:
            content_lower = item['content'].lower()
            if query_lower in content_lower:
                # Обрезаем контент до релевантной части
                content = item['content']
                if len(content) > 500:
                    # Находим позицию запроса
                    pos = content_lower.find(query_lower)
                    start = max(0, pos - 250)
                    end = min(len(content), pos + 250)
                    content = content[start:end]
                
                results.append({
                    'filename': item['filename'],
                    'content': content,
                    'category': item['category'],
                    'source': 'database',
                    'relevance': self.calculate_relevance(query_lower, content_lower)
                })
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:5]  # Возвращаем топ-5 результатов
    
    def calculate_relevance(self, query, content):
        """Расчет релевантности"""
        query_words = query.split()
        relevance = 0
        
        for word in query_words:
            if len(word) > 2:  # Игнорируем короткие слова
                count = content.count(word)
                relevance += count * len(word)  # Длинные слова важнее
        
        return relevance
    
    def solve_math(self, message):
        """Решение математических задач"""
        try:
            # Импортируем улучшенный математический решатель
            from rubin_math_solver import solve_math_problem
            result = solve_math_problem(message)
            return result
        except ImportError:
            # Fallback к простому решателю
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
    
    def generate_smart_response(self, message, search_results):
        """Генерация умного ответа на основе найденных результатов"""
        if not search_results:
            return self.generate_fallback_response(message)
        
        # Группируем результаты по категориям
        categories = {}
        for result in search_results:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        response_parts = []
        sources = []
        
        # Формируем ответ по категориям
        for category, results in categories.items():
            if category == 'programming':
                response_parts.append("🐍 **ПРОГРАММИРОВАНИЕ:**")
            elif category == 'electronics':
                response_parts.append("⚡ **ЭЛЕКТРОТЕХНИКА:**")
            elif category == 'automation':
                response_parts.append("🤖 **АВТОМАТИЗАЦИЯ:**")
            elif category == 'radiomechanics':
                response_parts.append("📡 **РАДИОМЕХАНИКА:**")
            elif category == 'mathematics':
                response_parts.append("🧮 **МАТЕМАТИКА:**")
            else:
                response_parts.append(f"📚 **{category.upper()}:**")
            
            for result in results[:2]:  # Максимум 2 результата на категорию
                filename = result['filename']
                content = result['content']
                
                # Обрезаем контент
                if len(content) > 300:
                    content = content[:300] + "..."
                
                response_parts.append(f"📄 **{filename}:**\n{content}")
                sources.append(filename)
        
        response = "\n\n".join(response_parts)
        
        # Добавляем информацию об источниках
        if sources:
            response += f"\n\n📚 **Источники:** {', '.join(set(sources))}"
        
        return response
    
    def generate_fallback_response(self, message):
        """Генерация fallback ответа"""
        message_lower = message.lower()
        
        # Специальные ответы для общих вопросов
        if any(word in message_lower for word in ['привет', 'hello', 'hi']):
            return "Привет! Я Rubin AI - ваш помощник по техническим вопросам. Я специализируюсь на программировании, электротехнике, автоматизации и математике. Чем могу помочь?"
        
        elif any(word in message_lower for word in ['что умеешь', 'возможности', 'help']):
            return """🤖 **МОИ ВОЗМОЖНОСТИ:**

🐍 **Программирование:**
• Python, C++, JavaScript
• Анализ и объяснение кода
• Решение алгоритмических задач

⚡ **Электротехника:**
• Схемы и компоненты
• Транзисторы, диоды, резисторы
• Законы Ома и Кирхгофа

🤖 **Автоматизация:**
• PLC программирование
• PMAC контроллеры
• ПИД регуляторы

🧮 **Математика:**
• Арифметические вычисления
• Алгебра и геометрия
• Решение уравнений

📡 **Радиомеханика:**
• Антенны и передатчики
• Электромагнитные волны
• Радиотехнические устройства

Задайте конкретный вопрос, и я найду информацию в базе знаний!"""
        
        elif any(word in message_lower for word in ['python', 'питон']):
            return """🐍 **PYTHON ПРОГРАММИРОВАНИЕ:**

Python - высокоуровневый язык программирования с простым синтаксисом.

**Основные возможности:**
• Переменные и типы данных
• Управляющие структуры (if, for, while)
• Функции и классы
• Модули и пакеты

**Популярные библиотеки:**
• NumPy - численные вычисления
• Pandas - анализ данных
• Matplotlib - визуализация
• Django/Flask - веб-разработка

**Пример простой программы:**
```python
def greet(name):
    return f"Привет, {name}!"

print(greet("Мир"))
```

Хотите узнать что-то конкретное о Python?"""
        
        elif any(word in message_lower for word in ['транзистор', 'transistor']):
            return """⚡ **ТРАНЗИСТОР:**

Транзистор - полупроводниковый прибор для усиления и переключения электрических сигналов.

**Типы транзисторов:**
• **Биполярные (BJT)** - управляются током
• **Полевые (FET)** - управляются напряжением
• **MOSFET** - металл-оксид-полупроводник

**Принцип работы:**
• База управляет током коллектора
• Коэффициент усиления h21э
• Режимы: отсечка, активный, насыщение

**Применение:**
• Усилители сигналов
• Цифровые переключатели
• Генераторы колебаний
• Регуляторы напряжения

Нужна более детальная информация о конкретном типе транзистора?"""
        
        else:
            return f"""Я не нашел точную информацию по запросу '{message}' в базе знаний.

**Попробуйте:**
• Переформулировать вопрос
• Использовать ключевые слова: Python, транзистор, PLC, математика
• Задать более конкретный вопрос

**Доступные темы:**
• Программирование (Python, C++)
• Электротехника (схемы, компоненты)
• Автоматизация (PLC, PMAC)
• Математика (вычисления, уравнения)
• Радиомеханика (антенны, передатчики)"""
    
    def generate_response(self, message):
        """Генерация ответа с использованием полной базы знаний"""
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
                'source': 'math_solver',
                'database_used': False
            }
        
        # Поиск в содержимом
        search_results = self.search_content(message)
        
        if search_results:
            response = self.generate_smart_response(message, search_results)
            return {
                'response': response,
                'category': 'knowledge_search',
                'confidence': 0.8,
                'source': 'database_search',
                'database_used': True,
                'sources_found': len(search_results)
            }
        else:
            response = self.generate_fallback_response(message)
            return {
                'response': response,
                'category': 'fallback',
                'confidence': 0.6,
                'source': 'fallback',
                'database_used': True,
                'sources_found': 0
            }

# Создаем экземпляр AI
rubin_ai = RubinEnhancedDatabase()

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
        'system': 'Rubin AI Enhanced Database',
        'databases': list(rubin_ai.databases.keys()),
        'test_documents': len(rubin_ai.test_documents),
        'database_content': len(rubin_ai.database_content),
        'conversations': len(rubin_ai.conversation_history)
    })

@app.route('/api/stats')
def stats():
    """Статистика системы"""
    return jsonify({
        'conversations': len(rubin_ai.conversation_history),
        'databases': {name: os.path.exists(path) for name, path in rubin_ai.databases.items()},
        'test_documents': {name: doc['size'] for name, doc in rubin_ai.test_documents.items()},
        'database_content_items': len(rubin_ai.database_content),
        'system': 'Rubin AI Enhanced Database',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search')
def search():
    """Поиск в базе знаний"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Параметр q обязателен'}), 400
    
    # Поиск в содержимом
    results = rubin_ai.search_content(query)
    
    return jsonify({
        'query': query,
        'results': results,
        'total_found': len(results),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Запуск Rubin AI Enhanced Database...")
    print("📡 API доступен на: http://localhost:8087")
    print("🔗 Тестирование: http://localhost:8087/api/health")
    print("🔍 Поиск: http://localhost:8087/api/search?q=python")
    print(f"📚 Загружено {len(rubin_ai.test_documents)} тестовых документов")
    print(f"🗄️ Загружено {len(rubin_ai.database_content)} элементов из базы данных")
    app.run(host='0.0.0.0', port=8087, debug=True)
