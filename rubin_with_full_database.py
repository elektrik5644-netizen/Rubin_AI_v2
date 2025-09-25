#!/usr/bin/env python3
"""
Rubin AI с полной базой знаний
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

class RubinWithFullDatabase:
    """Rubin AI с подключением к полной базе знаний"""
    
    def __init__(self):
        self.databases = {
            'main': 'rubin_ai_v2.db',
            'documents': 'rubin_ai_documents.db',
            'knowledge': 'rubin_knowledge_base.db',
            'learning': 'rubin_learning.db'
        }
        self.conversation_history = []
        self.load_test_documents()
    
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
    
    def detect_category(self, filename):
        """Определение категории по имени файла"""
        filename_lower = filename.lower()
        if 'python' in filename_lower or 'programming' in filename_lower:
            return 'programming'
        elif 'electrical' in filename_lower or 'circuit' in filename_lower:
            return 'electronics'
        elif 'controller' in filename_lower or 'automation' in filename_lower:
            return 'automation'
        elif 'radio' in filename_lower:
            return 'radiomechanics'
        else:
            return 'general'
    
    def search_in_database(self, query, db_name='main'):
        """Поиск в базе данных"""
        if db_name not in self.databases:
            return []
        
        db_path = self.databases[db_name]
        if not os.path.exists(db_path):
            return []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Поиск в таблице documents
            cursor.execute("""
                SELECT filename, content, category 
                FROM documents 
                WHERE content LIKE ? OR filename LIKE ?
                LIMIT 5
            """, (f'%{query}%', f'%{query}%'))
            
            results = cursor.fetchall()
            conn.close()
            
            return [{'filename': r[0], 'content': r[1], 'category': r[2]} for r in results]
            
        except Exception as e:
            print(f"Ошибка поиска в {db_name}: {e}")
            return []
    
    def search_in_test_documents(self, query):
        """Поиск в тестовых документах"""
        results = []
        query_lower = query.lower()
        
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
                        'size': doc_data['size']
                    })
        
        return results[:3]  # Возвращаем топ-3 результата
    
    def solve_math(self, message):
        """Решение математических задач"""
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
        
        # Поиск в базах данных
        db_results = []
        for db_name in self.databases:
            results = self.search_in_database(message, db_name)
            db_results.extend(results)
        
        # Поиск в тестовых документах
        test_results = self.search_in_test_documents(message)
        
        # Объединяем результаты
        all_results = db_results + test_results
        
        if all_results:
            # Формируем ответ на основе найденной информации
            response_parts = []
            sources = []
            
            for result in all_results[:3]:  # Берем топ-3 результата
                filename = result.get('filename', 'Неизвестный файл')
                content = result.get('content', '')
                category = result.get('category', 'general')
                
                # Обрезаем контент до разумного размера
                if len(content) > 500:
                    content = content[:500] + "..."
                
                response_parts.append(f"📄 **{filename}** ({category}):\n{content}")
                sources.append(filename)
            
            response = "\n\n".join(response_parts)
            
            # Добавляем информацию об источниках
            response += f"\n\n📚 **Источники:** {', '.join(sources)}"
            
            return {
                'response': response,
                'category': 'knowledge_search',
                'confidence': 0.8,
                'source': 'database_search',
                'database_used': True,
                'sources_found': len(all_results)
            }
        else:
            # Fallback ответ
            return {
                'response': f"Я не нашел информацию по запросу '{message}' в базе знаний.\n\nПопробуйте:\n• Переформулировать вопрос\n• Использовать другие ключевые слова\n• Спросить о программировании, электротехнике или автоматизации",
                'category': 'no_results',
                'confidence': 0.3,
                'source': 'fallback',
                'database_used': True,
                'sources_found': 0
            }

# Создаем экземпляр AI
rubin_ai = RubinWithFullDatabase()

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
        'system': 'Rubin AI with Full Database',
        'databases': list(rubin_ai.databases.keys()),
        'test_documents': len(rubin_ai.test_documents),
        'conversations': len(rubin_ai.conversation_history)
    })

@app.route('/api/stats')
def stats():
    """Статистика системы"""
    return jsonify({
        'conversations': len(rubin_ai.conversation_history),
        'databases': {name: os.path.exists(path) for name, path in rubin_ai.databases.items()},
        'test_documents': {name: doc['size'] for name, doc in rubin_ai.test_documents.items()},
        'system': 'Rubin AI with Full Database',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search')
def search():
    """Поиск в базе знаний"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Параметр q обязателен'}), 400
    
    # Поиск в базах данных
    db_results = []
    for db_name in rubin_ai.databases:
        results = rubin_ai.search_in_database(query, db_name)
        db_results.extend(results)
    
    # Поиск в тестовых документах
    test_results = rubin_ai.search_in_test_documents(query)
    
    return jsonify({
        'query': query,
        'database_results': db_results,
        'test_document_results': test_results,
        'total_found': len(db_results) + len(test_results),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Запуск Rubin AI с полной базой знаний...")
    print("📡 API доступен на: http://localhost:8086")
    print("🔗 Тестирование: http://localhost:8086/api/health")
    print("🔍 Поиск: http://localhost:8086/api/search?q=python")
    print(f"📚 Загружено {len(rubin_ai.test_documents)} тестовых документов")
    print(f"🗄️ Доступно {len(rubin_ai.databases)} баз данных")
    app.run(host='0.0.0.0', port=8086, debug=True)












