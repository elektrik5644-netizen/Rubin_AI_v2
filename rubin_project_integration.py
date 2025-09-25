#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграция системы чтения проектов с Rubin AI
Позволяет Rubin AI использовать знания из других проектов для ответов
"""

import os
import sqlite3
import json
import re
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from project_folder_reader import ProjectFolderReader

class RubinProjectIntegration:
    """Интеграция системы проектов с Rubin AI"""
    
    def __init__(self, project_db_path="rubin_project_knowledge.db"):
        self.project_db_path = project_db_path
        self.setup_logging()
        self.project_reader = ProjectFolderReader(project_db_path)
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def analyze_user_project(self, project_path: str, project_name: str = None) -> Dict:
        """Анализ проекта пользователя"""
        try:
            self.logger.info(f"🔍 Анализируем проект пользователя: {project_path}")
            
            # Анализируем проект
            success = self.project_reader.analyze_project_folder(project_path, project_name)
            
            if success:
                # Получаем сводку по проекту
                projects = self.project_reader.get_all_projects()
                if projects:
                    latest_project = projects[0]  # Самый последний
                    project_id = latest_project[0]
                    summary = self.project_reader.get_project_summary(project_id)
                    
                    return {
                        'success': True,
                        'project_id': project_id,
                        'project_name': latest_project[1],
                        'project_type': latest_project[3],
                        'summary': summary,
                        'message': f"Проект '{latest_project[1]}' успешно проанализирован"
                    }
                    
            return {
                'success': False,
                'message': 'Ошибка анализа проекта'
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа проекта: {e}")
            return {
                'success': False,
                'message': f'Ошибка: {e}'
            }
            
    def search_project_knowledge(self, query: str, project_id: int = None) -> List[Dict]:
        """Поиск знаний в проектах"""
        try:
            # Поиск в файлах проектов
            results = self.project_reader.search_in_projects(query, [project_id] if project_id else None)
            
            # Форматируем результаты
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'file_name': result[3],
                    'file_path': result[2],
                    'project_name': result[16],
                    'language': result[10],
                    'framework': result[11],
                    'content_preview': result[8][:200] + '...' if len(result[8]) > 200 else result[8],
                    'complexity_score': result[12],
                    'line_count': result[13]
                })
                
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска знаний: {e}")
            return []
            
    def get_project_architecture(self, project_id: int) -> Dict:
        """Получение архитектуры проекта"""
        try:
            conn = sqlite3.connect(self.project_db_path)
            cursor = conn.cursor()
            
            # Получаем компоненты проекта
            cursor.execute('''
                SELECT component_type, component_name, description, complexity_score
                FROM project_components 
                WHERE project_id = ?
                ORDER BY component_type, component_name
            ''', (project_id,))
            
            components = cursor.fetchall()
            
            # Группируем по типам
            architecture = {}
            for comp_type, comp_name, description, complexity in components:
                if comp_type not in architecture:
                    architecture[comp_type] = []
                    
                architecture[comp_type].append({
                    'name': comp_name,
                    'description': description,
                    'complexity': complexity
                })
                
            conn.close()
            
            return architecture
            
        except Exception as e:
            self.logger.error(f"Ошибка получения архитектуры: {e}")
            return {}
            
    def get_project_patterns(self, project_id: int) -> List[Dict]:
        """Получение паттернов проектирования из проекта"""
        try:
            conn = sqlite3.connect(self.project_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, description, code_example, tags, confidence_score
                FROM project_knowledge 
                WHERE project_id = ? AND knowledge_type = 'pattern'
                ORDER BY confidence_score DESC
            ''', (project_id,))
            
            patterns = cursor.fetchall()
            
            formatted_patterns = []
            for title, description, code_example, tags, confidence in patterns:
                formatted_patterns.append({
                    'title': title,
                    'description': description,
                    'code_example': code_example,
                    'tags': json.loads(tags) if tags else [],
                    'confidence': confidence
                })
                
            conn.close()
            
            return formatted_patterns
            
        except Exception as e:
            self.logger.error(f"Ошибка получения паттернов: {e}")
            return []
            
    def generate_project_insights(self, project_id: int) -> Dict:
        """Генерация инсайтов по проекту"""
        try:
            # Получаем сводку проекта
            summary = self.project_reader.get_project_summary(project_id)
            if not summary:
                return {'error': 'Проект не найден'}
                
            project_info = summary['project']
            file_stats = summary['file_stats']
            component_stats = summary['component_stats']
            knowledge_stats = summary['knowledge_stats']
            
            # Анализируем архитектуру
            architecture = self.get_project_architecture(project_id)
            
            # Получаем паттерны
            patterns = self.get_project_patterns(project_id)
            
            # Генерируем инсайты
            insights = {
                'project_overview': {
                    'name': project_info[1],
                    'type': project_info[3],
                    'total_files': file_stats[0] if file_stats[0] else 0,
                    'total_size_mb': round((file_stats[1] if file_stats[1] else 0) / (1024 * 1024), 2),
                    'average_complexity': round(file_stats[2] if file_stats[2] else 0, 2)
                },
                'architecture_analysis': {
                    'total_components': sum(len(comps) for comps in architecture.values()),
                    'component_types': list(architecture.keys()),
                    'most_complex_components': self._find_most_complex_components(architecture)
                },
                'design_patterns': {
                    'total_patterns': len(patterns),
                    'patterns': patterns[:5],  # Топ-5 паттернов
                    'pattern_categories': list(set(tag for pattern in patterns for tag in pattern['tags']))
                },
                'recommendations': self._generate_recommendations(project_info, architecture, patterns)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации инсайтов: {e}")
            return {'error': f'Ошибка: {e}'}
            
    def _find_most_complex_components(self, architecture: Dict) -> List[Dict]:
        """Поиск наиболее сложных компонентов"""
        all_components = []
        for comp_type, components in architecture.items():
            for comp in components:
                comp['type'] = comp_type
                all_components.append(comp)
                
        # Сортируем по сложности
        all_components.sort(key=lambda x: x['complexity'], reverse=True)
        return all_components[:5]
        
    def _generate_recommendations(self, project_info: Tuple, architecture: Dict, patterns: List[Dict]) -> List[str]:
        """Генерация рекомендаций по проекту"""
        recommendations = []
        
        # Анализируем размер проекта
        total_files = project_info[6] if isinstance(project_info[6], int) else 0
        if total_files > 100:  # Больше 100 файлов
            recommendations.append("Проект довольно большой. Рассмотрите возможность разделения на модули.")
            
        # Анализируем архитектуру
        if 'class' in architecture and len(architecture['class']) > 20:
            recommendations.append("Много классов в проекте. Проверьте соблюдение принципа единственной ответственности.")
            
        if 'function' in architecture and len(architecture['function']) > 50:
            recommendations.append("Много функций. Рассмотрите возможность группировки в классы или модули.")
            
        # Анализируем паттерны
        if len(patterns) == 0:
            recommendations.append("Не обнаружено явных паттернов проектирования. Рассмотрите использование известных паттернов.")
            
        # Анализируем сложность
        high_complexity_components = [comp for comp_type in architecture.values() 
                                    for comp in comp_type if comp['complexity'] > 15]
        if high_complexity_components:
            recommendations.append(f"Обнаружены компоненты высокой сложности: {', '.join([comp['name'] for comp in high_complexity_components[:3]])}")
            
        return recommendations
        
    def answer_with_project_context(self, question: str, project_id: int = None) -> Dict:
        """Ответ на вопрос с использованием контекста проектов"""
        try:
            # Поиск релевантной информации в проектах
            search_results = self.search_project_knowledge(question, project_id)
            
            if not search_results:
                return {
                    'answer': 'Не найдено релевантной информации в проанализированных проектах.',
                    'sources': [],
                    'confidence': 0.0
                }
                
            # Формируем ответ на основе найденной информации
            answer_parts = []
            sources = []
            
            for result in search_results[:3]:  # Топ-3 результата
                sources.append({
                    'file': result['file_name'],
                    'project': result['project_name'],
                    'language': result['language'],
                    'preview': result['content_preview']
                })
                
                answer_parts.append(f"В проекте '{result['project_name']}' в файле '{result['file_name']}' найдено:")
                answer_parts.append(result['content_preview'])
                answer_parts.append("")
                
            # Формируем итоговый ответ
            answer = "\n".join(answer_parts)
            
            # Рассчитываем уверенность на основе количества найденных результатов
            confidence = min(0.9, len(search_results) * 0.3)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'total_results': len(search_results)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка формирования ответа: {e}")
            return {
                'answer': f'Ошибка при поиске информации: {e}',
                'sources': [],
                'confidence': 0.0
            }
            
    def get_project_comparison(self, project_ids: List[int]) -> Dict:
        """Сравнение проектов"""
        try:
            comparison = {
                'projects': [],
                'comparison_metrics': {},
                'recommendations': []
            }
            
            for project_id in project_ids:
                summary = self.project_reader.get_project_summary(project_id)
                if summary:
                    project_info = summary['project']
                    file_stats = summary['file_stats']
                    
                    comparison['projects'].append({
                        'id': project_id,
                        'name': project_info[1],
                        'type': project_info[3],
                        'file_count': file_stats[0] if file_stats[0] else 0,
                        'total_size': file_stats[1] if file_stats[1] else 0,
                        'avg_complexity': file_stats[2] if file_stats[2] else 0
                    })
                    
            # Сравниваем метрики
            if len(comparison['projects']) >= 2:
                file_counts = [p['file_count'] for p in comparison['projects']]
                complexities = [p['avg_complexity'] for p in comparison['projects']]
                
                comparison['comparison_metrics'] = {
                    'largest_project': max(comparison['projects'], key=lambda x: x['file_count']),
                    'most_complex_project': max(comparison['projects'], key=lambda x: x['avg_complexity']),
                    'file_count_range': f"{min(file_counts)} - {max(file_counts)}",
                    'complexity_range': f"{min(complexities):.1f} - {max(complexities):.1f}"
                }
                
                # Генерируем рекомендации
                if max(file_counts) > min(file_counts) * 3:
                    comparison['recommendations'].append("Значительная разница в размере проектов. Рассмотрите стандартизацию структуры.")
                    
                if max(complexities) > min(complexities) * 2:
                    comparison['recommendations'].append("Большая разница в сложности проектов. Проверьте качество кода.")
                    
            return comparison
            
        except Exception as e:
            self.logger.error(f"Ошибка сравнения проектов: {e}")
            return {'error': f'Ошибка: {e}'}
            
    def export_project_knowledge(self, project_id: int, format: str = 'json') -> str:
        """Экспорт знаний проекта"""
        try:
            # Получаем все данные проекта
            summary = self.project_reader.get_project_summary(project_id)
            architecture = self.get_project_architecture(project_id)
            patterns = self.get_project_patterns(project_id)
            
            export_data = {
                'project_summary': summary,
                'architecture': architecture,
                'design_patterns': patterns,
                'export_timestamp': str(datetime.now()),
                'export_format': format
            }
            
            if format == 'json':
                return json.dumps(export_data, ensure_ascii=False, indent=2)
            else:
                return str(export_data)
                
        except Exception as e:
            self.logger.error(f"Ошибка экспорта: {e}")
            return f'Ошибка экспорта: {e}'


def main():
    """Главная функция для тестирования интеграции"""
    print("🔗 ИНТЕГРАЦИЯ СИСТЕМЫ ПРОЕКТОВ С RUBIN AI")
    print("=" * 50)
    
    integration = RubinProjectIntegration()
    
    # Получаем список проектов
    projects = integration.project_reader.get_all_projects()
    
    if not projects:
        print("❌ Нет проанализированных проектов")
        print("Сначала проанализируйте проект с помощью project_folder_reader.py")
        return
        
    print(f"📋 Найдено проектов: {len(projects)}")
    for i, project in enumerate(projects):
        print(f"  {i+1}. {project[1]} ({project[3]}) - {project[8]} файлов")
        
    # Выбираем проект для демонстрации
    if projects:
        project_id = projects[0][0]  # Берем первый проект
        
        print(f"\n🔍 Демонстрация возможностей для проекта: {projects[0][1]}")
        
        # Генерируем инсайты
        insights = integration.generate_project_insights(project_id)
        print(f"\n📊 Инсайты проекта:")
        print(f"  - Тип: {insights['project_overview']['type']}")
        print(f"  - Файлов: {insights['project_overview']['total_files']}")
        print(f"  - Размер: {insights['project_overview']['total_size_mb']} MB")
        print(f"  - Средняя сложность: {insights['project_overview']['average_complexity']}")
        
        # Показываем архитектуру
        if insights['architecture_analysis']['component_types']:
            print(f"\n🏗️ Архитектура:")
            print(f"  - Компонентов: {insights['architecture_analysis']['total_components']}")
            print(f"  - Типы: {', '.join(insights['architecture_analysis']['component_types'])}")
            
        # Показываем рекомендации
        if insights['recommendations']:
            print(f"\n💡 Рекомендации:")
            for rec in insights['recommendations']:
                print(f"  - {rec}")
                
        # Демонстрируем поиск
        print(f"\n🔍 Демонстрация поиска:")
        search_query = "class"
        search_results = integration.search_project_knowledge(search_query, project_id)
        print(f"  Поиск '{search_query}': найдено {len(search_results)} результатов")
        
        if search_results:
            print(f"  Первый результат: {search_results[0]['file_name']} в проекте {search_results[0]['project_name']}")


if __name__ == "__main__":
    main()
