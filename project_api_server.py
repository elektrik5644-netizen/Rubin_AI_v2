#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API для интеграции системы проектов с Rubin AI
Предоставляет REST API для анализа проектов и получения знаний
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from pathlib import Path
from rubin_project_integration import RubinProjectIntegration

app = Flask(__name__)
CORS(app)

# Инициализация интеграции
project_integration = RubinProjectIntegration()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/projects/analyze', methods=['POST'])
def analyze_project():
    """Анализ проекта пользователя"""
    try:
        data = request.get_json()
        
        if not data or 'project_path' not in data:
            return jsonify({
                'success': False,
                'error': 'Не указан путь к проекту'
            }), 400
            
        project_path = data['project_path']
        project_name = data.get('project_name')
        
        # Проверяем существование пути
        if not os.path.exists(project_path):
            return jsonify({
                'success': False,
                'error': f'Путь не найден: {project_path}'
            }), 404
            
        # Анализируем проект
        result = project_integration.analyze_user_project(project_path, project_name)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Ошибка анализа проекта: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Получение списка всех проектов"""
    try:
        projects = project_integration.project_reader.get_all_projects()
        
        formatted_projects = []
        for project in projects:
            formatted_projects.append({
                'id': project[0],
                'name': project[1],
                'path': project[2],
                'type': project[3],
                'description': project[4],
                'created_at': project[5],
                'last_analyzed': project[6],
                'total_files': project[8],
                'total_size': project[9],
                'language_stats': json.loads(project[10]) if project[10] else {},
                'framework_stats': json.loads(project[11]) if project[11] else {}
            })
            
        return jsonify({
            'success': True,
            'projects': formatted_projects,
            'total': len(formatted_projects)
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения проектов: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/<int:project_id>/insights', methods=['GET'])
def get_project_insights(project_id):
    """Получение инсайтов по проекту"""
    try:
        insights = project_integration.generate_project_insights(project_id)
        
        if 'error' in insights:
            return jsonify({
                'success': False,
                'error': insights['error']
            }), 404
            
        return jsonify({
            'success': True,
            'insights': insights
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения инсайтов: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/<int:project_id>/architecture', methods=['GET'])
def get_project_architecture(project_id):
    """Получение архитектуры проекта"""
    try:
        architecture = project_integration.get_project_architecture(project_id)
        
        return jsonify({
            'success': True,
            'architecture': architecture
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения архитектуры: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/<int:project_id>/patterns', methods=['GET'])
def get_project_patterns(project_id):
    """Получение паттернов проектирования"""
    try:
        patterns = project_integration.get_project_patterns(project_id)
        
        return jsonify({
            'success': True,
            'patterns': patterns
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения паттернов: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/search', methods=['POST'])
def search_in_projects():
    """Поиск в проектах"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Не указан поисковый запрос'
            }), 400
            
        query = data['query']
        project_id = data.get('project_id')
        
        # Поиск в проектах
        results = project_integration.search_project_knowledge(query, project_id)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/answer', methods=['POST'])
def answer_with_context():
    """Ответ на вопрос с использованием контекста проектов"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Не указан вопрос'
            }), 400
            
        question = data['question']
        project_id = data.get('project_id')
        
        # Получаем ответ с контекстом
        answer = project_integration.answer_with_project_context(question, project_id)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer['answer'],
            'sources': answer['sources'],
            'confidence': answer['confidence'],
            'total_results': answer['total_results']
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка формирования ответа: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/compare', methods=['POST'])
def compare_projects():
    """Сравнение проектов"""
    try:
        data = request.get_json()
        
        if not data or 'project_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'Не указаны ID проектов для сравнения'
            }), 400
            
        project_ids = data['project_ids']
        
        if len(project_ids) < 2:
            return jsonify({
                'success': False,
                'error': 'Необходимо указать минимум 2 проекта для сравнения'
            }), 400
            
        # Сравниваем проекты
        comparison = project_integration.get_project_comparison(project_ids)
        
        if 'error' in comparison:
            return jsonify({
                'success': False,
                'error': comparison['error']
            }), 500
            
        return jsonify({
            'success': True,
            'comparison': comparison
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка сравнения проектов: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/<int:project_id>/export', methods=['GET'])
def export_project_knowledge(project_id):
    """Экспорт знаний проекта"""
    try:
        format_type = request.args.get('format', 'json')
        
        if format_type not in ['json', 'text']:
            return jsonify({
                'success': False,
                'error': 'Неподдерживаемый формат экспорта'
            }), 400
            
        # Экспортируем знания
        export_data = project_integration.export_project_knowledge(project_id, format_type)
        
        if export_data.startswith('Ошибка'):
            return jsonify({
                'success': False,
                'error': export_data
            }), 500
            
        return jsonify({
            'success': True,
            'format': format_type,
            'data': export_data
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка экспорта: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/projects/<int:project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Удаление проекта"""
    try:
        # Удаляем проект из базы данных
        conn = sqlite3.connect(project_integration.project_db_path)
        cursor = conn.cursor()
        
        # Удаляем связанные записи
        cursor.execute("DELETE FROM project_knowledge WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM project_components WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM project_files WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Проект {project_id} удален'
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка удаления проекта: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка: {e}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка состояния API"""
    try:
        # Проверяем подключение к базе данных
        projects = project_integration.project_reader.get_all_projects()
        
        return jsonify({
            'status': 'healthy',
            'message': 'Project Integration API работает',
            'total_projects': len(projects),
            'database': 'connected'
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка проверки состояния: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """Документация API"""
    docs = {
        'title': 'Rubin AI Project Integration API',
        'version': '1.0.0',
        'description': 'API для анализа проектов и интеграции знаний с Rubin AI',
        'endpoints': {
            'POST /api/projects/analyze': {
                'description': 'Анализ проекта пользователя',
                'parameters': {
                    'project_path': 'string (required) - путь к проекту',
                    'project_name': 'string (optional) - имя проекта'
                }
            },
            'GET /api/projects': {
                'description': 'Получение списка всех проектов'
            },
            'GET /api/projects/{id}/insights': {
                'description': 'Получение инсайтов по проекту'
            },
            'GET /api/projects/{id}/architecture': {
                'description': 'Получение архитектуры проекта'
            },
            'GET /api/projects/{id}/patterns': {
                'description': 'Получение паттернов проектирования'
            },
            'POST /api/projects/search': {
                'description': 'Поиск в проектах',
                'parameters': {
                    'query': 'string (required) - поисковый запрос',
                    'project_id': 'integer (optional) - ID проекта для ограничения поиска'
                }
            },
            'POST /api/projects/answer': {
                'description': 'Ответ на вопрос с использованием контекста проектов',
                'parameters': {
                    'question': 'string (required) - вопрос',
                    'project_id': 'integer (optional) - ID проекта для ограничения контекста'
                }
            },
            'POST /api/projects/compare': {
                'description': 'Сравнение проектов',
                'parameters': {
                    'project_ids': 'array (required) - массив ID проектов'
                }
            },
            'GET /api/projects/{id}/export': {
                'description': 'Экспорт знаний проекта',
                'parameters': {
                    'format': 'string (optional) - формат экспорта (json, text)'
                }
            },
            'DELETE /api/projects/{id}': {
                'description': 'Удаление проекта'
            },
            'GET /api/health': {
                'description': 'Проверка состояния API'
            }
        }
    }
    
    return jsonify(docs), 200

if __name__ == '__main__':
    print("🚀 Запуск Rubin AI Project Integration API")
    print("=" * 50)
    print("📡 API доступен на: http://localhost:8091")
    print("📚 Документация: http://localhost:8091/api/docs")
    print("🔍 Проверка состояния: http://localhost:8091/api/health")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=8091, debug=True)





