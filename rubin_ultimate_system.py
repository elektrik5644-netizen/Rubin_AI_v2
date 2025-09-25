#!/usr/bin/env python3
"""
Ultimate версия Rubin AI с полной базой знаний и улучшенной математикой
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

class RubinUltimateSystem:
    """Ultimate версия Rubin AI с полной базой знаний"""
    
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
        self.initialize_math_solver()
    
    def initialize_math_solver(self):
        """Инициализация математического решателя"""
        try:
            from rubin_math_solver import solve_math_problem
            self.solve_math_problem = solve_math_problem
            self.math_solver_available = True
            print("✅ Улучшенный математический решатель загружен")
        except ImportError:
            self.math_solver_available = False
            print("⚠️ Улучшенный математический решатель недоступен")
    
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
                
                # Проверяем структуру таблицы
                cursor.execute("PRAGMA table_info(documents)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'filename' in columns and 'content' in columns:
                    cursor.execute("SELECT filename, content FROM documents LIMIT 20")
                    results = cursor.fetchall()
                    for filename, content in results:
                        if content and len(content) > 50:
                            self.database_content.append({
                                'filename': filename,
                                'content': content,
                                'category': self.detect_category(filename),
                                'source': 'database'
                            })
                else:
                    print("⚠️ Неподдерживаемая структура таблицы documents")
                
                conn.close()
            except Exception as e:
                print(f"Ошибка загрузки из базы данных: {e}")
        
        # Загружаем из базы знаний
        if os.path.exists('rubin_knowledge_base.db'):
            try:
                conn = sqlite3.connect('rubin_knowledge_base.db')
                cursor = conn.cursor()
                
                cursor.execute("SELECT title, content, category, subject FROM knowledge_entries")
                results = cursor.fetchall()
                for title, content, category, subject in results:
                    if content and len(content) > 50:
                        self.database_content.append({
                            'filename': f"{title} ({subject})",
                            'content': content,
                            'category': category,
                            'source': 'knowledge_base',
                            'subject': subject
                        })
                
                conn.close()
                print(f"✅ Загружено {len(results)} записей из базы знаний")
            except Exception as e:
                print(f"Ошибка загрузки из базы знаний: {e}")
    
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
        elif 'chemistry' in filename_lower or 'химия' in filename_lower:
            return 'chemistry'
        elif 'physics' in filename_lower or 'физика' in filename_lower:
            return 'physics'
        elif 'algebra' in filename_lower or 'алгебра' in filename_lower:
            return 'mathematics'
        elif 'geometry' in filename_lower or 'геометрия' in filename_lower:
            return 'mathematics'
        elif 'calculus' in filename_lower or 'анализ' in filename_lower or 'высшая математика' in filename_lower:
            return 'mathematics'
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
            relevance = 0
            
            # Поиск по полной фразе
            if query_lower in content_lower:
                relevance += 100
            
            # Поиск по отдельным словам
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    relevance += 10
            
            if relevance > 0:
                # Находим релевантные фрагменты
                lines = doc_data['content'].split('\n')
                relevant_lines = []
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(word in line_lower for word in query_words if len(word) > 2):
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        relevant_lines.extend(lines[start:end])
                
                if relevant_lines:
                    results.append({
                        'filename': filename,
                        'content': '\n'.join(relevant_lines),
                        'category': doc_data['category'],
                        'source': 'test_documents',
                        'relevance': relevance
                    })
        
        # Поиск в содержимом базы данных
        for item in self.database_content:
            content_lower = item['content'].lower()
            relevance = 0
            
            # Поиск по полной фразе
            if query_lower in content_lower:
                relevance += 100
            
            # Поиск по отдельным словам
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    relevance += 10
            
            if relevance > 0:
                # Обрезаем контент до релевантной части
                content = item['content']
                if len(content) > 500:
                    # Находим позицию первого найденного слова
                    pos = -1
                    for word in query_words:
                        if len(word) > 2:
                            word_pos = content_lower.find(word)
                            if word_pos != -1 and (pos == -1 or word_pos < pos):
                                pos = word_pos
                    
                    if pos != -1:
                        start = max(0, pos - 250)
                        end = min(len(content), pos + 250)
                        content = content[start:end]
                
                results.append({
                    'filename': item['filename'],
                    'content': content,
                    'category': item['category'],
                    'source': 'database',
                    'relevance': relevance
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
        if self.math_solver_available:
            return self.solve_math_problem(message)
        else:
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
        """Генерация естественного ответа на основе найденных результатов"""
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
        
        # Формируем естественный ответ по категориям
        for category, results in categories.items():
            if category == 'programming':
                response_parts.append("Программирование:")
            elif category == 'electronics':
                response_parts.append("Электротехника:")
            elif category == 'automation':
                response_parts.append("Автоматизация:")
            elif category == 'radiomechanics':
                response_parts.append("Радиомеханика:")
            elif category == 'chemistry':
                response_parts.append("Химия:")
            elif category == 'physics':
                response_parts.append("Физика:")
            elif category == 'mathematics':
                response_parts.append("Математика:")
            else:
                response_parts.append(f"{category.title()}:")
            
            for result in results[:2]:  # Максимум 2 результата на категорию
                filename = result['filename']
                content = result['content']
                
                # Обрезаем контент
                if len(content) > 300:
                    content = content[:300] + "..."
                
                response_parts.append(f"{filename}:\n{content}")
                sources.append(filename)
        
        response = "\n\n".join(response_parts)
        
        # Добавляем информацию об источниках
        if sources:
            response += f"\n\nИсточники: {', '.join(set(sources))}"
        
        return response
    
    def generate_fallback_response(self, message):
        """Генерация естественного ответа без шаблонов"""
        message_lower = message.lower()
        
        # Естественные ответы на общие вопросы
        if any(word in message_lower for word in ['привет', 'hello', 'hi']):
            return "Привет! Расскажите, что вас интересует."
        
        elif any(word in message_lower for word in ['что умеешь', 'возможности', 'help']):
            return """Я могу помочь с:
• Программированием - Python, C++, JavaScript, алгоритмы
• Электротехникой - схемы, компоненты, законы
• Автоматизацией - ПЛК, контроллеры, ПИД регуляторы
• Математикой - уравнения, вычисления, геометрия
• Химией - реакции, элементы, уравнения
• Физикой - механика, электричество, энергия
• Радиомеханикой - антенны, сигналы, передатчики

Задайте конкретный вопрос!"""
        
        elif any(word in message_lower for word in ['python', 'питон']):
            return """Python - язык программирования с простым синтаксисом.

Основы:
• Переменные и типы данных
• Управляющие структуры (if, for, while)
• Функции и классы
• Модули и пакеты

Популярные библиотеки:
• NumPy - численные вычисления
• Pandas - анализ данных
• Matplotlib - визуализация
• Django/Flask - веб-разработка

Что именно вас интересует в Python?"""
        
        elif any(word in message_lower for word in ['транзистор', 'transistor']):
            return """Транзистор - полупроводниковый прибор для усиления и переключения сигналов.

Типы:
• Биполярные (BJT) - управляются током
• Полевые (FET) - управляются напряжением
• MOSFET - металл-оксид-полупроводник

Принцип работы:
• База управляет током коллектора
• Коэффициент усиления h21э
• Режимы: отсечка, активный, насыщение

Применение:
• Усилители сигналов
• Переключатели
• Генераторы

Какой аспект транзисторов вас интересует?"""
        
        # Общий естественный ответ
        return f"Вы спросили: '{message}'. Расскажите больше о том, что вас интересует, и я постараюсь помочь."
    
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
rubin_ai = RubinUltimateSystem()

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
        'system': 'Rubin AI Ultimate System',
        'databases': list(rubin_ai.databases.keys()),
        'test_documents': len(rubin_ai.test_documents),
        'database_content': len(rubin_ai.database_content),
        'math_solver': rubin_ai.math_solver_available,
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
        'math_solver_available': rubin_ai.math_solver_available,
        'system': 'Rubin AI Ultimate System',
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
    print("🚀 Запуск Rubin AI Ultimate System...")
    print("📡 API доступен на: http://localhost:8088")
    print("🔗 Тестирование: http://localhost:8088/api/health")
    print("🔍 Поиск: http://localhost:8088/api/search?q=python")
    print(f"📚 Загружено {len(rubin_ai.test_documents)} тестовых документов")
    print(f"🗄️ Загружено {len(rubin_ai.database_content)} элементов из базы данных")
    print(f"🧮 Математический решатель: {'✅ Доступен' if rubin_ai.math_solver_available else '❌ Недоступен'}")
    app.run(host='0.0.0.0', port=8088, debug=True)
